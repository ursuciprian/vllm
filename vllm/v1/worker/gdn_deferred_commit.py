# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred GDN checkpoints: replay the accepted prefix before a state copy.

With ``VLLM_GDN_DEFERRED_CHECKPOINTS=1`` the b12x Qwen GDN decode kernel stops
writing one full recurrent-state checkpoint per verified token. It writes one
base checkpoint into the running block plus compact per-token records into the
speculative blocks, and the next decode step replays the accepted prefix onto
the base as part of the state read it already performs. That removes four of
the six state snapshots the kernel moves per layer-step.

The consequence for this file: **the running block is no longer the committed
state**, and the speculative blocks are no longer checkpoints. Every reader
outside the decode kernel must first ask b12x to materialize the accepted
prefix. There are exactly two such readers under align mamba cache mode, both
block-boundary state copies:

* ``postprocess_mamba_fused_kernel`` -- same step, right after the sampler.
* ``precopy_mamba_align_fused_kernel`` -- next step, the CPU-metadata fallback.

Both are handled the same way, and the ordering is what makes it correct:

1. Run the copy kernel with ``DECISION_ONLY=True``. It emits, per request, the
   pre-advance running column and the accepted count, and copies nothing. The
   copy decision therefore still lives in exactly one place.
2. Gather that request's 1 + num_spec block ids into a b12x-shaped state-index
   table, and set each boundary request's commit destination to its own
   running block (an in-place commit).
3. Ask every GDN layer to commit. b12x reads the base from column 0, replays
   ``accepted - 1`` records out of columns 1.., and writes the result back to
   column 0.
4. Run the copy kernel for real with ``DEFERRED_TEMPORAL=True``, so its
   temporal half copies the now-committed running block with bias 0. The conv
   half keeps its accepted-token bias, which is why the shared copy helper
   takes the two biases separately.

A third reader exists and is refused rather than handled:
``checkpoint_mamba_states_kernel`` (request-boundary checkpoints) can ask for
up to three different capture biases for one request in one launch, and a
single accepted-prefix commit cannot express that. See
:func:`refuse_reasons`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import triton
import triton.language as tl

from vllm import envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def refuse_reasons(vllm_config: "VllmConfig") -> list[str]:
    """Why deferred GDN checkpoints cannot be used for this configuration.

    Empty means usable. This is deliberately fail-closed: the caller raises on
    a non-empty list when the env var asked for the feature, rather than
    silently running the shipped path, because the two differ in what the
    state pool means and a silent downgrade would be invisible in a benchmark.
    """
    reasons: list[str] = []
    cache_config = vllm_config.cache_config
    if getattr(cache_config, "mamba_cache_mode", "none") != "align":
        reasons.append(
            "requires --mamba-cache-mode align (the speculative columns are "
            "only exclusively owned in align mode)"
        )
    if getattr(vllm_config, "use_request_boundary_checkpoints", False):
        reasons.append(
            "is incompatible with request-boundary checkpoints: one capture "
            "can request several distinct biases for the same request and a "
            "single accepted-prefix commit cannot express that"
        )
    num_spec = 0
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None:
        num_spec = int(
            getattr(speculative_config, "num_speculative_tokens", 0) or 0
        )
    if num_spec < 1:
        reasons.append(
            "has nothing to defer without speculative tokens "
            f"(num_speculative_tokens={num_spec})"
        )
    return reasons


def requested() -> bool:
    return bool(envs.VLLM_GDN_DEFERRED_CHECKPOINTS)


def resolve(vllm_config: "VllmConfig") -> bool:
    """Return whether deferred checkpoints are on, raising if asked and unusable."""
    if not requested():
        return False
    reasons = refuse_reasons(vllm_config)
    if reasons:
        raise ValueError(
            "VLLM_GDN_DEFERRED_CHECKPOINTS=1 but the deferred GDN checkpoint "
            "path " + "; ".join(reasons) + ". Unset it or fix the configuration."
        )
    logger.info_once(
        "GDN deferred checkpoints enabled: the b12x decode kernel writes one "
        "base checkpoint plus per-token records, and block-boundary copies go "
        "through an accepted-prefix commit."
    )
    return True


@triton.jit(do_not_specialize=["num_reqs"])
def gather_gdn_commit_windows_kernel(
    commit_src_col_ptr,  # [max_reqs] pre-advance running column, -1 = skip
    block_table_ptr,  # [max_reqs, max_blocks] int32, the GDN group's table
    block_table_stride_req,
    out_state_indices_ptr,  # [max_reqs, COLUMNS] int32
    out_destination_ptr,  # [max_reqs] int32, -1 = skip
    num_reqs,
    COLUMNS: tl.constexpr,
):
    """Shape one request's state window the way b12x's commit expects it.

    ``out_state_indices[r, j] = block_table[r, src_col + j]`` so column 0 is the
    base checkpoint and columns 1.. are that step's record blocks, exactly the
    window the decode kernel wrote. The destination is column 0 again: the
    commit runs in place and the existing block copy then moves the committed
    state to the new window with a zero temporal bias.
    """
    req_idx = tl.program_id(0)
    if req_idx >= num_reqs:
        return
    src_col = tl.load(commit_src_col_ptr + req_idx)
    columns = tl.arange(0, COLUMNS)
    if src_col < 0:
        tl.store(out_destination_ptr + req_idx, -1)
        tl.store(out_state_indices_ptr + req_idx * COLUMNS + columns, 0)
        return
    row = block_table_ptr + req_idx.to(tl.int64) * block_table_stride_req
    blocks = tl.load(row + src_col + columns)
    tl.store(out_state_indices_ptr + req_idx * COLUMNS + columns, blocks)
    tl.store(out_destination_ptr + req_idx, tl.load(row + src_col))


class GdnDeferredCommit:
    """Per-step buffers and the commit call for one GDN mamba group.

    The buffers are allocated once and never reallocated, so their addresses
    are stable for CUDA graph capture, exactly like the other align-mode
    per-request buffers.
    """

    def __init__(
        self,
        *,
        max_num_reqs: int,
        state_index_columns: int,
        device: torch.device,
    ) -> None:
        self.max_num_reqs = int(max_num_reqs)
        self.state_index_columns = int(state_index_columns)
        factory = dict(dtype=torch.int32, device=device)
        self.src_col = torch.full((self.max_num_reqs,), -1, **factory)
        self.accepted = torch.ones((self.max_num_reqs,), **factory)
        self.state_indices = torch.zeros(
            (self.max_num_reqs, self.state_index_columns), **factory
        )
        self.destination = torch.full((self.max_num_reqs,), -1, **factory)
        self.num_seqs = torch.zeros((1,), **factory)
        self._layers: list[Any] = []

    def bind_layers(self, layers: list[Any]) -> None:
        """Record the GDN layers whose state must be committed."""
        self._layers = [
            layer
            for layer in layers
            if getattr(layer, "b12x_gdn_deferred_checkpoints", False)
        ]

    @property
    def active(self) -> bool:
        return bool(self._layers)

    def reset(self, num_reqs: int) -> None:
        self.src_col[:num_reqs].fill_(-1)
        self.accepted[:num_reqs].fill_(1)
        self.num_seqs.fill_(int(num_reqs))

    def commit(
        self,
        *,
        num_reqs: int,
        block_table: torch.Tensor,
    ) -> None:
        """Replay each boundary request's accepted prefix into its own block.

        ``block_table`` is the GDN group's device block table in request-slot
        order, i.e. the same rows ``commit_src_col`` was written with.
        """
        if not self.active or num_reqs == 0:
            return
        gather_gdn_commit_windows_kernel[(num_reqs,)](
            self.src_col,
            block_table,
            block_table.stride(0),
            self.state_indices,
            self.destination,
            num_reqs,
            COLUMNS=self.state_index_columns,
        )
        for layer in self._layers:
            layer.commit_b12x_gdn_deferred(
                state_indices=self.state_indices,
                num_accepted_tokens=self.accepted,
                num_seqs=self.num_seqs,
                destination_indices=self.destination,
            )

    def precompile(self) -> None:
        """Warm the commit kernel so it may be used inside a graph capture."""
        for layer in self._layers:
            layer.precompile_b12x_gdn_deferred_commit()
