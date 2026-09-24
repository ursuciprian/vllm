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
prefix. There are three such readers under align mamba cache mode, two
block-boundary state copies and one export:

* ``postprocess_mamba_fused_kernel`` -- same step, right after the sampler.
* ``precopy_mamba_align_fused_kernel`` -- next step, the CPU-metadata fallback.

Both run decide -> commit -> copy; they differ in where the commit writes:

1. Run the copy kernel with ``DECISION_ONLY=True``. It emits, per request in
   batch-row order, the pre-advance running column, the accepted count and
   the destination column, and copies nothing. The copy decision therefore
   still lives in exactly one place.
2. Gather each mamba group's 1 + num_spec block ids into a b12x-shaped
   state-index table and resolve the destination block:

   * post-sampler: the aligned block ``bt[dest]`` (``dest <= src``), so a
     running window that stays live keeps its base and records;
   * pre-forward: ``bt[src]`` in place (the old window is dead afterwards),
     skipped when the bias is 0 because nothing needs replaying.
3. Ask every GDN layer to commit. b12x reads the base from column 0, replays
   ``accepted - 1`` records out of columns 1.. and writes the destination.
4. Run the copy kernel for real with ``DEFERRED_TEMPORAL=True``:

   * post-sampler: the temporal copy is skipped (``temporal_bias < 0``), since
     the commit already wrote ``bt[dest]``;
   * pre-forward: the temporal half copies the committed ``bt[src]`` with
     bias 0.

   The conv half keeps its accepted-token bias either way, which is why the
   shared copy helper takes the two biases separately. Only temporal states
   flagged in ``state_temporal_deferred`` (deferred GDN layers) change; conv
   states and other mamba layers (e.g. the PLE short conv) keep the shipped
   copy, so they behave identically with the flag on or off.

The third reader is ``checkpoint_mamba_states_kernel``: request-boundary
checkpoints (``--recurrent-checkpoint-policy auto`` resolves to them for this
model) export a request's committed state at the prompt, instruction and
response endpoints into budgeted blocks outside its window. It runs right after
the sampler, before the block-boundary postprocess, and exports without
touching the window (commit-then-export):

* Prompt and instruction captures always ask for bias 0 (they happen on a
  prefill step, see ``prepare_boundary_capture``), and so does a response
  that ends on the first verified token. Column 0 already holds exactly that
  state in both modes -- prefill writes a full state there and the deferred
  decode kernel writes its full base checkpoint there -- so the shipped copy
  stays.
* A response capture with bias ``b > 0`` would select speculative column
  ``b``, which is a record now. :func:`boundary_export_decision_kernel` turns
  those rows into commits of ``b + 1`` accepted tokens straight into the
  capture's own destination block (one per mamba group), and the copy kernel
  skips the temporal copy for deferred GDN states with ``b > 0``. The commit
  reads the window and writes only the destination, so the window's base and
  records are intact for the block-boundary commit that follows in the same
  step, and for the next decode step.
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
    alias_errors_ptr,  # [1] int32, accumulated: destinations naming a record
    MAX_REQS: tl.constexpr,
    BLOCK: tl.constexpr,
    COLUMNS: tl.constexpr,
    EXPLICIT_DST: tl.constexpr = False,
    explicit_destination_ptr=None,  # [MAX_REQS] block ids, read iff EXPLICIT_DST
):
    """Shape one mamba group's commit windows the way b12x's commit expects.

    One program over every request row, so the launch never depends on the
    live batch size and can sit in a CUDA graph. Rows are in batch order, the
    order the decision kernels write and the block table uses.
    ``out_state_indices[r, j] = block_table[r, src_col + j]`` (column 0 is the
    base, 1.. are that step's record blocks) and the destination is
    ``block_table[r, dst_col]``, or ``explicit_destination[r]`` (a block id,
    -1 = skip) under ``EXPLICIT_DST`` -- the boundary export, whose
    destination is a capture block outside the table. With no committing row,
    ``num_seqs`` is 0 and b12x's commit exits without touching the pool.
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
    if EXPLICIT_DST:
        destination = tl.load(explicit_destination_ptr + rows, mask=live, other=-1)
        live = live & (destination >= 0)
    else:
        destination = tl.load(row_base + dst, mask=live, other=-1)
    # b12x refuses (silently skips) a destination that names one of the
    # row's record blocks. dst <= src makes that impossible unless the block
    # table repeats an id; count it on device so it is observable without a
    # host sync on the hot path (GdnDeferredCommit.alias_errors).
    aliased = tl.zeros([BLOCK], dtype=tl.int32)
    for column in tl.static_range(1, COLUMNS):
        record = tl.load(row_base + src + column, mask=live, other=-2)
        aliased += (live & (record == destination)).to(tl.int32)
    tl.atomic_add(alias_errors_ptr, tl.sum(aliased, axis=0))
    tl.store(
        out_destination_ptr + rows, tl.where(live, destination, -1), mask=in_range
    )
    any_live = tl.max(live.to(tl.int32), axis=0)
    tl.store(out_num_seqs_ptr, any_live * MAX_REQS)


@triton.jit
def boundary_export_decision_kernel(
    idx_mapping_ptr,  # [num_reqs] batch row -> request slot, -1 = masked
    state_idx_ptr,  # [max_reqs] running column per slot
    capture_tokens_ptr,  # [num_reqs, NUM_CAPTURES]
    capture_bias_ptr,  # [num_reqs, NUM_CAPTURES]
    destination_blocks_ptr,  # [max_reqs, NUM_CAPTURES, NUM_GROUPS]
    commit_src_col_ptr,  # [MAX_REQS] out
    commit_accepted_ptr,  # [MAX_REQS] out
    export_destination_ptr,  # [NUM_GROUPS, MAX_REQS] out
    MAX_REQS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    NUM_CAPTURES: tl.constexpr,
    KIND: tl.constexpr,
):
    """Turn a capture of speculative column ``b > 0`` into an export commit.

    One program per batch row. Only ``KIND`` (the response slot) can carry a
    bias; the other captures are bias 0 and keep the shipped copy.
    """
    batch_idx = tl.program_id(0)
    offset = batch_idx * NUM_CAPTURES + KIND
    if tl.load(capture_tokens_ptr + offset) <= 0:
        return
    bias = tl.load(capture_bias_ptr + offset)
    if bias <= 0:
        return
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0:
        return
    tl.store(commit_src_col_ptr + batch_idx, tl.load(state_idx_ptr + req_idx))
    tl.store(commit_accepted_ptr + batch_idx, bias + 1)
    for group in tl.static_range(NUM_GROUPS):
        block = tl.load(
            destination_blocks_ptr
            + (req_idx * NUM_CAPTURES + KIND) * NUM_GROUPS
            + group
        )
        tl.store(
            export_destination_ptr + group * MAX_REQS + batch_idx,
            tl.where(block > 0, block, -1),
        )


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
    addresses, and the live batch size never enters a launch. The boundary
    export uses the same buffers with explicit per-group destinations
    (``export_destination``) and its own graph.
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
        # Device-side count of commits b12x would refuse; read it when
        # debugging, never on the hot path.
        self.alias_errors = torch.zeros((1,), **factory)
        # [group, row] capture block ids for commit(export=True), -1 = skip.
        self.export_destination = torch.full(
            (len(groups), self.max_num_reqs), -1, **factory
        )
        self.groups = [
            _GroupCommit(
                table,
                [
                    layer
                    for layer in layers
                    if getattr(layer, "b12x_gdn_deferred_checkpoints", False) is True
                ],
                self.max_num_reqs,
                self.state_index_columns,
                device,
            )
            for table, layers in groups
        ]
        self._graphs: dict[bool, torch.cuda.CUDAGraph] = {}
        self._graph_failed = False

    @property
    def active(self) -> bool:
        return any(group.layers for group in self.groups)

    @property
    def _graph(self) -> torch.cuda.CUDAGraph | None:
        return self._graphs.get(False)

    def _launch(self, export: bool) -> None:
        block = triton.next_power_of_2(self.max_num_reqs)
        for index, group in enumerate(self.groups):
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
                self.alias_errors,
                MAX_REQS=self.max_num_reqs,
                BLOCK=block,
                COLUMNS=self.state_index_columns,
                EXPLICIT_DST=export,
                explicit_destination_ptr=self.export_destination[index],
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

    def commit(self, export: bool = False) -> None:
        """Replay each committing request's accepted prefix, then reset.

        ``export=True`` writes ``export_destination`` instead of the block
        table's ``dst_col`` block and leaves the window untouched.
        """
        if not self.active:
            return
        graph = self._graphs.get(export)
        if graph is not None:
            graph.replay()
            return
        # First use runs eagerly (compiles the gather, warms the commit), then
        # captures the same launches for every later step.
        self._launch(export)
        self._capture(export)

    def _capture(self, export: bool) -> None:
        if (
            self._graph_failed
            or self.src_col.device.type != "cuda"
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        import gc

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        gc_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            # thread_local: other threads (async output, RoCE allreduce
            # helper) may make CUDA calls while this mid-serving capture runs.
            with torch.cuda.graph(
                graph, stream=stream, capture_error_mode="thread_local"
            ):
                self._launch(export)
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
        self._graphs[export] = graph


def precompile_all(forward_context: dict[str, Any]) -> int:
    """Compile and warm every deferred GDN layer's commit before serving.

    Strict: a layer that cannot compile its commit fails the boot, instead of
    compiling lazily on the first boundary crossing (possibly under a frozen
    b12x session). Returns the number of layers warmed.
    """
    count = 0
    for layer in forward_context.values():
        if getattr(layer, "b12x_gdn_deferred_checkpoints", False) is True:
            layer.precompile_b12x_gdn_deferred_commit()
            count += 1
    return count
