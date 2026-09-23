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


def refuse_reasons(
    vllm_config: "VllmConfig",
    *,
    decode_kernel: str | None = "b12x",
    prefill_backend: str | None = "b12x",
) -> list[str]:
    """Why deferred GDN checkpoints cannot be used for this configuration.

    Empty means usable. This is deliberately fail-closed: the caller raises on
    a non-empty list when the env var asked for the feature, rather than
    silently running the shipped path, because the two differ in what the
    state pool means and a silent downgrade would be invisible in a benchmark.
    """
    reasons: list[str] = []
    if decode_kernel != "b12x":
        reasons.append(
            f"requires the b12x GDN decode kernel (got {decode_kernel!r}); "
            "only it writes and replays the per-token records"
        )
    if prefill_backend != "b12x":
        reasons.append(
            f"requires the b12x GDN prefill backend (got {prefill_backend!r}); "
            "another prefill kernel could read a record block as a state"
        )
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


def resolve(
    vllm_config: "VllmConfig",
    *,
    decode_kernel: str | None,
    prefill_backend: str | None,
) -> bool:
    """Return whether deferred checkpoints are on, raising if asked and unusable.

    Called for every Qwen GDN layer, b12x or not, so the flag can never be
    silently ignored.
    """
    if not requested():
        return False
    reasons = refuse_reasons(
        vllm_config, decode_kernel=decode_kernel, prefill_backend=prefill_backend
    )
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


@triton.jit
def gather_gdn_commit_windows_kernel(
    commit_src_col_ptr,  # [MAX_REQS] pre-advance running column, -1 = none
    commit_dst_col_ptr,  # [MAX_REQS] block-table column the commit writes
    block_table_ptr,  # [>= MAX_REQS, max_blocks] int32, one mamba group
    block_table_stride_req,
    out_state_indices_ptr,  # [MAX_REQS, COLUMNS] int32
    out_destination_ptr,  # [MAX_REQS] int32, -1 = skip
    out_num_seqs_ptr,  # [1] int32: MAX_REQS if any row commits, else 0
    MAX_REQS: tl.constexpr,
    BLOCK: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    """Shape one mamba group's commit windows the way b12x's commit expects.

    One program over every request row, so the launch never depends on the
    live batch size and can sit in a CUDA graph. Rows are in batch order, the
    order the decision kernels write and the block table uses.
    ``out_state_indices[r, j] = block_table[r, src_col + j]`` (column 0 is the
    base, 1.. are that step's record blocks) and the destination is
    ``block_table[r, dst_col]``. With no committing row, ``num_seqs`` is 0 and
    b12x's commit exits without touching the pool.
    """
    rows = tl.arange(0, BLOCK)
    in_range = rows < MAX_REQS
    src = tl.load(commit_src_col_ptr + rows, mask=in_range, other=-1)
    dst = tl.load(commit_dst_col_ptr + rows, mask=in_range, other=-1)
    live = in_range & (src >= 0)
    row_base = block_table_ptr + rows.to(tl.int64) * block_table_stride_req
    for column in tl.static_range(COLUMNS):
        block = tl.load(row_base + src + column, mask=live, other=0)
        tl.store(out_state_indices_ptr + rows * COLUMNS + column, block, mask=in_range)
    destination = tl.load(row_base + dst, mask=live, other=-1)
    tl.store(
        out_destination_ptr + rows, tl.where(live, destination, -1), mask=in_range
    )
    any_live = tl.max(live.to(tl.int32), axis=0)
    tl.store(out_num_seqs_ptr, any_live * MAX_REQS)


class _GroupCommit:
    """Per mamba block-table group: its own table, windows and layers."""

    def __init__(self, block_table, layers, max_num_reqs, columns, device):
        factory = dict(dtype=torch.int32, device=device)
        self.block_table = block_table
        self.layers = layers
        self.state_indices = torch.zeros((max_num_reqs, columns), **factory)
        self.destination = torch.full((max_num_reqs,), -1, **factory)
        self.num_seqs = torch.zeros((1,), **factory)


class GdnDeferredCommit:
    """Commit buffers and the commit call for every GDN mamba group.

    A hybrid model splits its GDN layers over several mamba block-table groups
    (Qwen3.8-Flash-Next: 36 GDN + attention layers give three), each with its
    own physical block ids. The copy decision is per request and shared; the
    windows, destinations and layers are per group.

    Per step the drivers launch one decision kernel and then call
    :meth:`commit`, which replays one CUDA graph holding every group's gather,
    every layer's commit and the decision-buffer reset. All buffers have fixed
    addresses, and the live batch size never enters a launch.
    """

    def __init__(
        self,
        *,
        max_num_reqs: int,
        state_index_columns: int,
        device: torch.device,
        groups: list[tuple[torch.Tensor, list[Any]]],
    ) -> None:
        self.max_num_reqs = int(max_num_reqs)
        self.state_index_columns = int(state_index_columns)
        factory = dict(dtype=torch.int32, device=device)
        # Decision buffers, batch-row order, shared by every group. The commit
        # resets them to -1 after use, so rows the decision kernel does not
        # write never commit.
        self.src_col = torch.full((self.max_num_reqs,), -1, **factory)
        self.dst_col = torch.full((self.max_num_reqs,), -1, **factory)
        self.accepted = torch.ones((self.max_num_reqs,), **factory)
        self.groups = [
            _GroupCommit(
                table,
                [
                    layer
                    for layer in layers
                    if getattr(layer, "b12x_gdn_deferred_checkpoints", False)
                ],
                self.max_num_reqs,
                self.state_index_columns,
                device,
            )
            for table, layers in groups
        ]
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_failed = False

    @property
    def active(self) -> bool:
        return any(group.layers for group in self.groups)

    def _launch(self) -> None:
        block = triton.next_power_of_2(self.max_num_reqs)
        for group in self.groups:
            if not group.layers:
                continue
            gather_gdn_commit_windows_kernel[(1,)](
                self.src_col,
                self.dst_col,
                group.block_table,
                group.block_table.stride(0),
                group.state_indices,
                group.destination,
                group.num_seqs,
                MAX_REQS=self.max_num_reqs,
                BLOCK=block,
                COLUMNS=self.state_index_columns,
            )
            for layer in group.layers:
                layer.commit_b12x_gdn_deferred(
                    state_indices=group.state_indices,
                    num_accepted_tokens=self.accepted,
                    num_seqs=group.num_seqs,
                    destination_indices=group.destination,
                )
        self.src_col.fill_(-1)
        self.dst_col.fill_(-1)

    def commit(self) -> None:
        """Replay each committing request's accepted prefix, then reset."""
        if not self.active:
            return
        if self._graph is not None:
            self._graph.replay()
            return
        # First use runs eagerly (compiles the gather, warms the commit), then
        # captures the same launches for every later step.
        self._launch()
        self._capture()

    def _capture(self) -> None:
        if self._graph_failed or torch.cuda.is_current_stream_capturing():
            return
        import gc

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        gc_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            with torch.cuda.graph(graph, stream=stream):
                self._launch()
        except Exception:
            # Correctness does not depend on the graph; only host time does.
            self._graph_failed = True
            logger.warning(
                "GDN deferred commit could not be captured in a CUDA graph; "
                "running it eagerly every step.",
                exc_info=True,
            )
            return
        finally:
            if gc_enabled:
                gc.enable()
        torch.cuda.current_stream().wait_stream(stream)
        self._graph = graph


def precompile_all(forward_context: dict[str, Any]) -> int:
    """Compile and warm every deferred GDN layer's commit before serving.

    Strict: a layer that cannot compile its commit fails the boot, instead of
    compiling lazily on the first boundary crossing (possibly under a frozen
    b12x session). Returns the number of layers warmed.
    """
    count = 0
    for layer in forward_context.values():
        if getattr(layer, "b12x_gdn_deferred_checkpoints", False):
            layer.precompile_b12x_gdn_deferred_commit()
            count += 1
    return count
