# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for deferred GDN checkpoints (VLLM_GDN_DEFERRED_CHECKPOINTS).

CPU: gating, plus the real Triton kernels under ``TRITON_INTERPRET=1`` in a
subprocess (needs Triton, no GPU). GPU-marked: the decide -> commit -> copy
protocol against the shipped copy, with a stand-in for b12x's commit.
"""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.v1.worker import gdn_deferred_commit as gdc


def _config(*, mode="align", boundary=False, num_spec=4):
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode=mode),
        use_request_boundary_checkpoints=boundary,
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec),
    )


def test_flag_is_declared_and_feeds_the_compile_cache_key(monkeypatch):
    # It changes which b12x plans a boot declares, so a warm torch AOT cache
    # must not be shared between on and off.
    assert "VLLM_GDN_DEFERRED_CHECKPOINTS" in envs.environment_variables
    monkeypatch.setenv("VLLM_GDN_DEFERRED_CHECKPOINTS", "1")
    on = envs.compile_factors()
    monkeypatch.setenv("VLLM_GDN_DEFERRED_CHECKPOINTS", "0")
    off = envs.compile_factors()
    assert on.get("VLLM_GDN_DEFERRED_CHECKPOINTS") != off.get(
        "VLLM_GDN_DEFERRED_CHECKPOINTS"
    )


def test_refuse_reasons_cover_every_unsupported_mode():
    assert gdc.refuse_reasons(_config()) == []
    assert len(gdc.refuse_reasons(_config(mode="all"))) == 1
    # Request-boundary checkpoints (policy auto) are served by the export.
    assert gdc.refuse_reasons(_config(boundary=True)) == []
    assert len(gdc.refuse_reasons(_config(num_spec=0))) == 1
    assert len(gdc.refuse_reasons(_config(mode="none", num_spec=0))) == 2
    assert len(gdc.refuse_reasons(_config(), decode_kernel="triton")) == 1
    assert len(gdc.refuse_reasons(_config(), prefill_backend="flashinfer")) == 1


def test_resolve_is_off_by_default_and_fails_closed(monkeypatch):
    b12x = dict(decode_kernel="b12x", prefill_backend="b12x")
    monkeypatch.delenv("VLLM_GDN_DEFERRED_CHECKPOINTS", raising=False)
    assert gdc.resolve(_config(mode="none"), decode_kernel="cuda",
                       prefill_backend="triton") is False
    monkeypatch.setenv("VLLM_GDN_DEFERRED_CHECKPOINTS", "1")
    assert gdc.resolve(_config(), **b12x) is True
    with pytest.raises(ValueError, match="mamba-cache-mode align"):
        gdc.resolve(_config(mode="all"), **b12x)
    # A non-b12x layer must not silently ignore the flag.
    with pytest.raises(ValueError, match="b12x GDN decode kernel"):
        gdc.resolve(_config(), decode_kernel="cuda", prefill_backend="b12x")
    with pytest.raises(ValueError, match="b12x GDN prefill backend"):
        gdc.resolve(_config(), decode_kernel="b12x", prefill_backend="triton")


_INTERPRETED = textwrap.dedent(
    """
    import torch
    from vllm.v1.worker.gdn_deferred_commit import gather_gdn_commit_windows_kernel
    from vllm.v1.worker.mamba_utils import (
        postprocess_mamba_fused_kernel, precopy_mamba_align_fused_kernel)

    i32 = dict(dtype=torch.int32)
    z64 = torch.zeros(1, dtype=torch.int64)
    z32 = torch.zeros(1, **i32)
    meta = (z64, 0, z64, z64, z32, z64, z32, z32, z32, z64)

    # --- post-sampler decision: batch-row order, destination column ---------
    block_size, num_reqs = 8, 3
    idx_mapping = torch.tensor([2, 0, 1], **i32)        # batch row -> slot
    accepted = torch.tensor([3, 1, 5], **i32)           # slot order
    state_idx = torch.tensor([1, 1, 2], **i32)
    new_computed = torch.tensor([17, 9, 16], **i32)
    src = torch.full((4,), -1, **i32); acc = torch.ones(4, **i32)
    dst = torch.full((4,), -1, **i32)
    postprocess_mamba_fused_kernel[(num_reqs, 1, 1)](
        accepted, state_idx, None, new_computed, None, *meta,
        accepted.clone(), idx_mapping, num_reqs,
        block_size=block_size, COPY_BLOCK_SIZE=16, CONV_STATE_DIM_FIRST=False,
        HAS_IDX_MAPPING=True, PRECOMPUTED_NEW_COMPUTED=True, TEMPORAL_TILES=1,
        DECISION_ONLY=True, commit_src_col_ptr=src, commit_accepted_ptr=acc,
        commit_dst_col_ptr=dst)
    for b in range(num_reqs):
        r = int(idx_mapping[b])
        running = int(new_computed[r]) - int(accepted[r]) + 1
        aligned = int(new_computed[r]) // block_size * block_size
        dest = aligned // block_size - 1
        bias = aligned - running
        if aligned < running or (dest == int(state_idx[r]) and bias == 0):
            assert int(src[b]) == -1, (b, src)
            continue
        assert int(src[b]) == int(state_idx[r]), (b, src)
        assert int(acc[b]) == bias + 1, (b, acc)
        assert int(dst[b]) == dest and dest <= int(src[b]), (b, dst)
    assert int(src[3]) == -1 and (src[:3] >= 0).sum() == 2

    # --- pre-forward decision: batch-row order, in place, bias 0 skipped ----
    new_col = torch.tensor([3, 2, 5], **i32)            # slot order
    old_col = torch.tensor([2, 2, 4], **i32)            # slot 1: no migration
    bias = torch.tensor([3, 1, 0], **i32)               # slot 2: bias 0
    src = torch.full((4,), -1, **i32); acc = torch.ones(4, **i32)
    dst = torch.full((4,), -1, **i32)
    precopy_mamba_align_fused_kernel[(num_reqs, 1, 1)](
        new_col, old_col, bias, *meta, idx_mapping, num_reqs,
        COPY_BLOCK_SIZE=16, CONV_STATE_DIM_FIRST=False, HAS_IDX_MAPPING=True,
        TEMPORAL_TILES=1, DECISION_ONLY=True, commit_src_col_ptr=src,
        commit_accepted_ptr=acc, commit_dst_col_ptr=dst)
    # batch 0 -> slot 2 (bias 0: skip), batch 1 -> slot 0, batch 2 -> slot 1
    assert src.tolist() == [-1, 2, -1, -1], src
    assert dst.tolist() == [-1, 2, -1, -1] and int(acc[1]) == 4, (dst, acc)

    # --- gather, three groups with different block ids ---------------------
    src = torch.tensor([2, -1, 4, -1], **i32)
    dst = torch.tensor([1, -1, 4, -1], **i32)
    for group in range(3):
        table = torch.arange(4 * 16, **i32).view(4, 16) + 1000 * (group + 1)
        windows = torch.zeros((4, 5), **i32)
        destination = torch.zeros(4, **i32)
        num_seqs = torch.zeros(1, **i32)
        alias = torch.zeros(1, **i32)
        gather_gdn_commit_windows_kernel[(1,)](
            src, dst, table, table.stride(0), windows, destination, num_seqs,
            alias, MAX_REQS=4, BLOCK=4, COLUMNS=5)
        assert windows[0].tolist() == table[0, 2:7].tolist()
        assert windows[2].tolist() == table[2, 4:9].tolist()
        assert destination.tolist() == [int(table[0, 1]), -1, int(table[2, 4]), -1]
        assert int(num_seqs) == 4 and int(alias) == 0
    none = torch.full((4,), -1, **i32)
    gather_gdn_commit_windows_kernel[(1,)](
        none, none, table, table.stride(0), windows, destination, num_seqs,
        alias, MAX_REQS=4, BLOCK=4, COLUMNS=5)
    assert int(num_seqs) == 0 and destination.tolist() == [-1] * 4
    # A table that repeats a block id makes the destination name a record.
    table[0, 4] = table[0, 1]
    gather_gdn_commit_windows_kernel[(1,)](
        src, dst, table, table.stride(0), windows, destination, num_seqs,
        alias, MAX_REQS=4, BLOCK=4, COLUMNS=5)
    assert int(alias) == 1, alias
    print("ok")
    """
)


def test_decision_and_gather_kernels_under_the_triton_interpreter():
    pytest.importorskip("triton")
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    result = subprocess.run(
        [sys.executable, "-c", _INTERPRETED],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith("ok")


class _FakeGdnLayer:
    """Stands in for b12x's commit on a pool of 2-float blocks.

    Block value = number of tokens applied. The base holds checkpoint 0 and a
    record block holds garbage, so replaying ``accepted - 1`` records is
    ``pool[base] + accepted - 1``. The last pool block is a write sink for
    skipped rows, which keeps the stand-in free of host syncs (graph-safe).
    """

    b12x_gdn_deferred_checkpoints = True

    def __init__(self, pool):
        self.pool = pool

    def commit_b12x_gdn_deferred(
        self, *, state_indices, num_accepted_tokens, num_seqs, destination_indices
    ):
        rows = torch.arange(destination_indices.numel(), device=self.pool.device)
        live = (destination_indices >= 0) & (rows < num_seqs.long())
        sink = self.pool.shape[0] - 1
        target = torch.where(live, destination_indices.long(), sink)
        replayed = num_accepted_tokens[: rows.numel()].float() - 1
        value = self.pool[state_indices[:, 0].long()] + replayed[:, None]
        value = torch.where(live[:, None], value, self.pool[sink])
        self.pool.index_copy_(0, target, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_commit_then_copy_matches_the_shipped_copy_across_three_groups():
    from vllm.v1.worker.mamba_utils import postprocess_mamba_fused_kernel

    device = torch.device("cuda")
    i32 = dict(dtype=torch.int32, device=device)
    i64 = dict(dtype=torch.int64, device=device)
    block_size, max_reqs, groups, blocks = 8, 4, 3, 64
    idx_mapping = torch.tensor([2, 0, 1], **i32)       # batch row -> slot
    accepted = torch.tensor([3, 1, 5, 1], **i32)       # slot order
    state_idx = torch.tensor([1, 1, 2, 0], **i32)
    new_computed = torch.tensor([17, 9, 16, 0], **i32)
    base = [10.0, 20.0, 30.0]                          # batch-row order

    tables, shipped, initial = [], [], []
    for _ in range(groups):
        table = (
            torch.randperm(blocks - 1, device=device)[: max_reqs * 12]
            .view(max_reqs, 12)
            .to(torch.int32)
        )
        ship = torch.full((blocks, 2), -7.0, device=device)
        defer = ship.clone()
        for b in range(3):
            src = int(state_idx[int(idx_mapping[b])])
            for j in range(5):
                ship[int(table[b, src + j])] = base[b] + j  # checkpoints
                defer[int(table[b, src + j])] = base[b] if j == 0 else 1000.0 + j
        tables.append(table)
        shipped.append(ship)
        initial.append(defer)
    deferred = [p.clone() for p in initial]

    def launch(pool, table, **extra):
        postprocess_mamba_fused_kernel[(3, 1, 1)](
            accepted, state_idx, None, new_computed, None,
            torch.tensor([table.data_ptr()], **i64), table.stride(0),
            torch.tensor([pool.data_ptr()], **i64),
            torch.tensor([pool.stride(0) * pool.element_size()], **i64),
            torch.tensor([pool.element_size()], **i32),
            torch.tensor([pool.shape[1]], **i64),
            torch.zeros(1, **i32), torch.zeros(1, **i32), torch.zeros(1, **i32),
            torch.zeros(1, **i64),
            accepted.clone(), idx_mapping, 3,
            block_size=block_size, COPY_BLOCK_SIZE=16, CONV_STATE_DIM_FIRST=False,
            HAS_IDX_MAPPING=True, PRECOMPUTED_NEW_COMPUTED=True, TEMPORAL_TILES=1,
            state_temporal_deferred_ptr=torch.ones(1, **i32),
            **extra,
        )

    for group in range(groups):
        launch(shipped[group], tables[group])  # the shipped copy

    commit = gdc.GdnDeferredCommit(
        max_num_reqs=max_reqs,
        state_index_columns=5,
        device=device,
        groups=[(tables[g], [_FakeGdnLayer(deferred[g])]) for g in range(groups)],
    )
    for _step in range(2):  # eager first use, then the captured graph
        for group in range(groups):
            deferred[group].copy_(initial[group])
        launch(
            deferred[0], tables[0], DECISION_ONLY=True,
            commit_src_col_ptr=commit.src_col,
            commit_accepted_ptr=commit.accepted,
            commit_dst_col_ptr=commit.dst_col,
        )
        commit.commit()
        for group in range(groups):
            launch(deferred[group], tables[group], DEFERRED_TEMPORAL=True)
        torch.cuda.synchronize()
        assert int((commit.src_col >= 0).sum()) == 0  # reset after commit
        checked = 0
        for group in range(groups):
            table = tables[group]
            for b in range(3):
                r = int(idx_mapping[b])
                running = int(new_computed[r]) - int(accepted[r]) + 1
                aligned = int(new_computed[r]) // block_size * block_size
                if aligned < running:
                    continue
                dest = int(table[b, aligned // block_size - 1])
                assert torch.equal(deferred[group][dest], shipped[group][dest])
                src_block = int(table[b, int(state_idx[r])])
                if src_block != dest:
                    # The running window keeps its base for the next step.
                    assert float(deferred[group][src_block, 0]) == base[b]
                checked += 1
        assert checked == 2 * groups
    assert commit._graph is not None, "commit graph capture failed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_precopy_driver_commits_in_place_then_migrates_like_the_shipped_copy():
    from types import SimpleNamespace as NS

    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

    device = torch.device("cuda")
    i32 = dict(dtype=torch.int32, device=device)
    i64 = dict(dtype=torch.int64, device=device)
    max_reqs, groups, blocks = 4, 3, 64
    idx_mapping = torch.tensor([2, 0, 1], **i32)       # batch row -> slot
    src_col = torch.tensor([2, 3, 1, -1], **i32)       # slot order
    dst_col = torch.tensor([3, 3, 2, 0], **i32)        # slot 1: no migration
    token_bias = torch.tensor([2, 1, 0, 0], **i32)     # slot 2: bias 0
    base = [10.0, 20.0, 30.0]                          # batch-row order

    tables, shipped, initial = [], [], []
    for _ in range(groups):
        table = (
            torch.randperm(blocks - 1, device=device)[: max_reqs * 12]
            .view(max_reqs, 12)
            .to(torch.int32)
        )
        ship = torch.full((blocks, 2), -7.0, device=device)
        defer = ship.clone()
        for b in range(3):
            src = int(src_col[int(idx_mapping[b])])
            for j in range(5):
                ship[int(table[b, src + j])] = base[b] + j
                defer[int(table[b, src + j])] = base[b] if j == 0 else 1000.0 + j
        tables.append(table)
        shipped.append(ship)
        initial.append(defer)
    deferred = [p.clone() for p in initial]

    def ctx(pools, commit):
        return NS(
            is_initialized=True,
            total_states=groups,
            block_table_ptrs=torch.tensor([t.data_ptr() for t in tables], **i64),
            block_table_stride_req=tables[0].stride(0),
            state_base_addrs=torch.tensor([p.data_ptr() for p in pools], **i64),
            state_block_strides=torch.tensor(
                [p.stride(0) * p.element_size() for p in pools], **i64
            ),
            state_elem_sizes=torch.tensor([4] * groups, **i32),
            state_inner_sizes=torch.tensor([2] * groups, **i64),
            state_conv_widths=torch.zeros(groups, **i32),
            state_group_indices=torch.arange(groups, **i32),
            state_dim_row_count=torch.zeros(groups, **i32),
            state_dim_row_stride=torch.zeros(groups, **i64),
            state_temporal_deferred=torch.ones(groups, **i32),
            gdn_deferred_commit=commit,
        )

    run = MambaSpecDecodeGPUContext.run_fused_precopy
    run(ctx(shipped, None), 3, dst_col, src_col, token_bias, idx_mapping)
    commit = gdc.GdnDeferredCommit(
        max_num_reqs=max_reqs,
        state_index_columns=5,
        device=device,
        groups=[(tables[g], [_FakeGdnLayer(deferred[g])]) for g in range(groups)],
    )
    for _step in range(2):  # eager first use, then the captured graph
        for group in range(groups):
            deferred[group].copy_(initial[group])
        run(ctx(deferred, commit), 3, dst_col, src_col, token_bias, idx_mapping)
        torch.cuda.synchronize()
        for group in range(groups):
            for b in range(3):
                r = int(idx_mapping[b])
                if int(src_col[r]) == int(dst_col[r]):
                    continue
                dest = int(tables[group][b, int(dst_col[r])])
                assert torch.equal(deferred[group][dest], shipped[group][dest]), (
                    group, b, deferred[group][dest], shipped[group][dest])
    assert commit._graph is not None, "commit graph capture failed"
    assert int(commit.alias_errors) == 0


def _mixed_group_case(device):
    """One mamba group holding a deferred GDN layer (conv + temporal state), a
    PLE-like conv-only layer and a non-deferred temporal state, all sharing one
    block table. Only the GDN temporal state may take the deferred path; every
    other state must match the shipped copy bit for bit.
    """
    from types import SimpleNamespace as NS

    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

    i32 = dict(dtype=torch.int32, device=device)
    i64 = dict(dtype=torch.int64, device=device)
    max_reqs, blocks = 4, 64
    idx_mapping = torch.tensor([2, 0, 1], **i32)
    src_col = torch.tensor([2, 3, 1, -1], **i32)
    dst_col = torch.tensor([3, 3, 2, 0], **i32)
    token_bias = torch.tensor([2, 1, 0, 0], **i32)
    base = [10.0, 20.0, 30.0]
    generator = torch.Generator().manual_seed(7)
    table = (
        torch.randperm(blocks - 1, generator=generator)[: max_reqs * 12]
        .view(max_reqs, 12).to(torch.int32).to(device)
    )
    ds = is_conv_state_dim_first()
    conv_shape = (blocks, 6, 4) if ds else (blocks, 4, 6)  # width 4, dim 6

    def conv_state():
        return torch.randn(conv_shape, generator=generator).to(device)

    shipped = {
        "gdn_conv": conv_state(),
        "gdn_temporal": torch.full((blocks, 2), -7.0, device=device),
        "ple_conv": conv_state(),
        "other_temporal": torch.randn((blocks, 2), generator=generator).to(device),
    }
    for b in range(3):
        src = int(src_col[int(idx_mapping[b])])
        for j in range(5):
            shipped["gdn_temporal"][int(table[b, src + j])] = base[b] + j
    initial = {k: v.clone() for k, v in shipped.items()}
    for b in range(3):
        src = int(src_col[int(idx_mapping[b])])
        for j in range(1, 5):
            initial["gdn_temporal"][int(table[b, src + j])] = 1000.0 + j
    deferred = {k: v.clone() for k, v in initial.items()}
    order = ["gdn_conv", "gdn_temporal", "ple_conv", "other_temporal"]

    def ctx(pools, commit, flags):
        conv = [name.endswith("_conv") for name in order]
        t = [pools[name] for name in order]
        return NS(
            is_initialized=True,
            total_states=len(order),
            block_table_ptrs=torch.tensor([table.data_ptr()], **i64),
            block_table_stride_req=table.stride(0),
            state_base_addrs=torch.tensor([x.data_ptr() for x in t], **i64),
            state_block_strides=torch.tensor(
                [x.stride(0) * x.element_size() for x in t], **i64
            ),
            state_elem_sizes=torch.tensor([x.element_size() for x in t], **i32),
            state_inner_sizes=torch.tensor(
                [(1 if ds else x.stride(1)) if c else x[0].numel()
                 for x, c in zip(t, conv)], **i64
            ),
            state_conv_widths=torch.tensor(
                [(x.size(2) if ds else x.size(1)) if c else 0
                 for x, c in zip(t, conv)], **i32
            ),
            state_group_indices=torch.zeros(len(order), **i32),
            state_dim_row_count=torch.tensor(
                [x.size(1) if (c and ds) else 0 for x, c in zip(t, conv)], **i32
            ),
            state_dim_row_stride=torch.tensor(
                [x.stride(1) * x.element_size() if (c and ds) else 0
                 for x, c in zip(t, conv)], **i64
            ),
            state_temporal_deferred=torch.tensor(flags, **i32),
            gdn_deferred_commit=commit,
        )

    run = MambaSpecDecodeGPUContext.run_fused_precopy
    run(ctx(shipped, None, [0, 0, 0, 0]), 3, dst_col, src_col, token_bias,
        idx_mapping)
    gdn_layer = _FakeGdnLayer(deferred["gdn_temporal"])
    ple_layer = SimpleNamespace(b12x_gdn_deferred_checkpoints=False)
    commit = gdc.GdnDeferredCommit(
        max_num_reqs=max_reqs, state_index_columns=5, device=device,
        groups=[(table, [gdn_layer, ple_layer])],
    )
    run(ctx(deferred, commit, [0, 1, 0, 0]), 3, dst_col, src_col, token_bias,
        idx_mapping)
    if device.type == "cuda":
        torch.cuda.synchronize()
    for name in ("gdn_conv", "ple_conv", "other_temporal"):
        assert torch.equal(deferred[name], shipped[name]), name
    for b in range(3):
        r = int(idx_mapping[b])
        if int(src_col[r]) == int(dst_col[r]):
            continue
        dest = int(table[b, int(dst_col[r])])
        assert torch.equal(
            deferred["gdn_temporal"][dest], shipped["gdn_temporal"][dest]
        ), b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mixed_gdn_and_ple_group_keeps_the_shipped_copy_for_non_gdn_states():
    _mixed_group_case(torch.device("cuda"))


def test_mixed_gdn_and_ple_group_under_the_triton_interpreter():
    pytest.importorskip("triton")
    code = (
        "import torch, tests.v1.worker.test_gdn_deferred_commit as t;"
        "t._mixed_group_case(torch.device('cpu')); print('ok')"
    )
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith("ok")


def _boundary_export_case(device):
    """Request-boundary export with deferred checkpoints on, then the same
    step's block-boundary postprocess, against the shipped path.

    Rows (batch order): a response capture whose bias (2) differs from the
    postprocess bias (1) of an in-place boundary commit; a response whose
    accepted run ends exactly on the block boundary (capture bias 4 == the
    in-place commit's bias); prompt + response captures at bias 0; and a
    stop-truncated response (capture bias 1 < accepted - 1) next to a commit
    into an earlier block. The export must equal the committed state at the
    boundary, and must not disturb the window the postprocess commits from.
    """
    from types import SimpleNamespace as NS

    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.v1.core.boundary_checkpoint import (
        NUM_BOUNDARY_CHECKPOINT_SLOTS as K,
    )
    from vllm.v1.core.boundary_checkpoint import (
        PROMPT_CHECKPOINT_SLOT,
        RESPONSE_CHECKPOINT_SLOT,
    )
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

    i32 = dict(dtype=torch.int32, device=device)
    i64 = dict(dtype=torch.int64, device=device)
    max_reqs, blocks, groups, block_size = 4, 64, 2, 8
    idx_mapping = torch.tensor([1, 0, 3, 2], **i32)    # batch row -> slot
    state_idx = torch.tensor([1, 1, 2, 2], **i32)      # slot order
    accepted = torch.tensor([5, 3, 3, 1], **i32)
    new_computed = torch.tensor([16, 17, 16, 9], **i32)
    capture = {  # batch row -> {kind: (tokens, bias)}
        0: {RESPONSE_CHECKPOINT_SLOT: (17, 2)},
        1: {RESPONSE_CHECKPOINT_SLOT: (16, 4)},
        2: {PROMPT_CHECKPOINT_SLOT: (9, 0), RESPONSE_CHECKPOINT_SLOT: (9, 0)},
        3: {RESPONSE_CHECKPOINT_SLOT: (15, 1)},
    }
    capture_tokens = torch.zeros((4, K), **i32)
    capture_bias = torch.zeros((4, K), **i32)
    destination_blocks = torch.zeros((max_reqs, K, groups), **i32)
    spare = iter(range(48, blocks))  # capture blocks, outside every table
    for b, kinds in capture.items():
        for kind, (tokens, bias) in kinds.items():
            capture_tokens[b, kind], capture_bias[b, kind] = tokens, bias
            for g in range(groups):
                destination_blocks[int(idx_mapping[b]), kind, g] = next(spare)

    generator = torch.Generator().manual_seed(11)
    tables = [
        torch.randperm(48, generator=generator)[: max_reqs * 12]
        .view(max_reqs, 12).to(torch.int32).to(device)
        for _ in range(groups)
    ]
    ds = is_conv_state_dim_first()
    conv_shape = (blocks, 6, 4) if ds else (blocks, 4, 6)
    shipped = {
        "conv0": torch.randn(conv_shape, generator=generator).to(device),
        "temporal0": torch.full((blocks, 2), -7.0, device=device),
        "temporal1": torch.full((blocks, 2), -7.0, device=device),
    }
    deferred = {k: v.clone() for k, v in shipped.items()}
    base = [10.0, 20.0, 30.0, 40.0]
    for g in range(groups):
        for b in range(4):
            src = int(state_idx[int(idx_mapping[b])])
            for j in range(5):
                block = int(tables[g][b, src + j])
                shipped[f"temporal{g}"][block] = base[b] + j
                deferred[f"temporal{g}"][block] = base[b] if j == 0 else 1000 + j
    order = ["conv0", "temporal0", "temporal1"]

    def ctx(pools, commit, flags):
        conv = [name.startswith("conv") for name in order]
        t = [pools[name] for name in order]
        return NS(
            is_initialized=True,
            num_groups=groups,
            total_states=len(order),
            block_size=block_size,
            num_accepted_tokens_out=torch.zeros(max_reqs, **i32),
            block_table_ptrs=torch.tensor([x.data_ptr() for x in tables], **i64),
            block_table_stride_req=tables[0].stride(0),
            state_base_addrs=torch.tensor([x.data_ptr() for x in t], **i64),
            state_block_strides=torch.tensor(
                [x.stride(0) * x.element_size() for x in t], **i64
            ),
            state_elem_sizes=torch.tensor([x.element_size() for x in t], **i32),
            state_inner_sizes=torch.tensor(
                [(1 if ds else x.stride(1)) if c else x[0].numel()
                 for x, c in zip(t, conv)], **i64
            ),
            state_conv_widths=torch.tensor(
                [(x.size(2) if ds else x.size(1)) if c else 0
                 for x, c in zip(t, conv)], **i32
            ),
            state_group_indices=torch.tensor([0, 0, 1], **i32),
            state_dim_row_count=torch.tensor(
                [x.size(1) if (c and ds) else 0 for x, c in zip(t, conv)], **i32
            ),
            state_dim_row_stride=torch.tensor(
                [x.stride(1) * x.element_size() if (c and ds) else 0
                 for x, c in zip(t, conv)], **i64
            ),
            state_temporal_deferred=torch.tensor(flags, **i32),
            gdn_deferred_commit=commit,
        )

    def step(context):
        MambaSpecDecodeGPUContext.checkpoint_request_boundaries(
            context, idx_mapping, state_idx, capture_tokens, capture_bias,
            destination_blocks)
        MambaSpecDecodeGPUContext.run_fused_postprocess_align(
            context, 4, accepted.clone(), state_idx, new_computed, idx_mapping)

    step(ctx(shipped, None, [0, 0, 0]))
    commit = gdc.GdnDeferredCommit(
        max_num_reqs=max_reqs, state_index_columns=5, device=device,
        groups=[(tables[g], [_FakeGdnLayer(deferred[f"temporal{g}"])])
                for g in range(groups)],
    )
    step(ctx(deferred, commit, [0, 1, 1]))
    if device.type == "cuda":
        torch.cuda.synchronize()

    for name in order:
        g = 0 if name == "conv0" else int(name[-1])
        for b, kinds in capture.items():
            for kind in kinds:
                block = int(destination_blocks[int(idx_mapping[b]), kind, g])
                assert torch.equal(deferred[name][block], shipped[name][block]), (
                    name, b, kind, deferred[name][block], shipped[name][block])
    for g in range(groups):
        pool, ref = deferred[f"temporal{g}"], shipped[f"temporal{g}"]
        for b in range(4):
            r = int(idx_mapping[b])
            running = int(new_computed[r]) - int(accepted[r]) + 1
            aligned = int(new_computed[r]) // block_size * block_size
            if aligned < running:
                continue
            dest = int(tables[g][b, aligned // block_size - 1])
            assert torch.equal(pool[dest], ref[dest]), (g, b)
            src = int(tables[g][b, int(state_idx[r])])
            if src != dest:
                assert float(pool[src, 0]) == base[b], (g, b)
    assert int(commit.alias_errors) == 0
    assert int((commit.src_col >= 0).sum()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_boundary_export_equals_the_committed_state():
    _boundary_export_case(torch.device("cuda"))


def test_boundary_export_under_the_triton_interpreter():
    pytest.importorskip("triton")
    code = (
        "import torch, tests.v1.worker.test_gdn_deferred_commit as t;"
        "t._boundary_export_case(torch.device('cpu')); print('ok')"
    )
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith("ok")


# --------------------------------------------------------------------------
# Compact records (VLLM_GDN_COMPACT_RECORDS): records in per-layer side buffers
# indexed by request slot, no speculative mamba blocks.
# --------------------------------------------------------------------------


def test_compact_flag_is_declared_and_hashed_only_when_on(monkeypatch):
    assert "VLLM_GDN_COMPACT_RECORDS" in envs.environment_variables
    monkeypatch.delenv("VLLM_GDN_COMPACT_RECORDS", raising=False)
    unset = envs.compile_factors()
    monkeypatch.setenv("VLLM_GDN_COMPACT_RECORDS", "0")
    off = envs.compile_factors()
    monkeypatch.setenv("VLLM_GDN_COMPACT_RECORDS", "1")
    on = envs.compile_factors()
    # Off keeps the pre-knob compile key; on is a different key.
    assert "VLLM_GDN_COMPACT_RECORDS" not in unset and off == unset
    assert on.get("VLLM_GDN_COMPACT_RECORDS") is True
    assert {k: v for k, v in on.items() if k != "VLLM_GDN_COMPACT_RECORDS"} == off


def test_compact_refuses_without_deferred_or_the_v2_runner(monkeypatch):
    b12x = dict(decode_kernel="b12x", prefill_backend="b12x")
    v2 = SimpleNamespace(**vars(_config()), use_v2_model_runner=True)
    monkeypatch.delenv("VLLM_GDN_COMPACT_RECORDS", raising=False)
    assert gdc.resolve_compact(_config(mode="none"), decode_kernel="cuda",
                               prefill_backend="triton") is False
    monkeypatch.setenv("VLLM_GDN_COMPACT_RECORDS", "1")
    monkeypatch.delenv("VLLM_GDN_DEFERRED_CHECKPOINTS", raising=False)
    assert len(gdc.refuse_reasons(_config(), compact=True)) == 2
    with pytest.raises(ValueError, match="VLLM_GDN_DEFERRED_CHECKPOINTS=1"):
        gdc.resolve_compact(v2, **b12x)
    monkeypatch.setenv("VLLM_GDN_DEFERRED_CHECKPOINTS", "1")
    with pytest.raises(ValueError, match="V2 model runner"):
        gdc.resolve_compact(_config(), **b12x)
    with pytest.raises(ValueError, match="b12x GDN decode kernel"):
        gdc.resolve_compact(v2, decode_kernel="triton", prefill_backend="b12x")
    assert gdc.resolve_compact(v2, **b12x) is True


def test_compact_drops_speculative_blocks_from_every_mamba_spec(monkeypatch):
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as M

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_block_size=3024, mamba_page_size_padded=None,
            mamba_cache_mode="align", use_kda_recoverssm=False,
        ),
        num_speculative_tokens=4,
    )

    def spec(mamba_type):
        layer = SimpleNamespace(
            get_state_shape=lambda: ((4, 8), (2, 8, 8)),
            get_state_dtype=lambda: (torch.bfloat16, torch.float32),
            mamba_type=mamba_type,
        )
        return MambaBase.get_kv_cache_spec(layer, config)

    monkeypatch.delenv("VLLM_GDN_COMPACT_RECORDS", raising=False)
    assert spec(M.GDN_ATTN).num_speculative_blocks == 4
    monkeypatch.setenv("VLLM_GDN_COMPACT_RECORDS", "1")
    # GDN and the PLE short conv together: get_mamba_layer_groups rejects
    # mamba layers that disagree on num_speculative_blocks.
    assert spec(M.GDN_ATTN).num_speculative_blocks == 0
    assert spec(M.SHORT_CONV).num_speculative_blocks == 0
    with pytest.raises(ValueError, match="GDN and short-conv layers only"):
        spec(M.MAMBA2)


def test_a_fresh_request_slot_never_replays_stale_records():
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
        check_fresh_record_slots,
    )

    fresh = {3, 5}
    # A prefill row (not spec) leaves the slot fresh; its first spec step at
    # accepted == 1 clears it.
    check_fresh_record_slots(fresh, [3, 5], [False, True], [4, 1])
    assert fresh == {3}
    check_fresh_record_slots(fresh, [3], [True], [1])
    assert not fresh
    # The previous owner's acceptance leaking into a reused slot is caught.
    fresh = {7}
    with pytest.raises(AssertionError, match="3 stale records"):
        check_fresh_record_slots(fresh, [7], [True], [4])


_COMPACT_INTERPRETED = textwrap.dedent(
    """
    import torch
    from vllm.v1.worker.gdn_deferred_commit import (
        boundary_export_decision_kernel, gather_gdn_commit_windows_kernel)
    from vllm.v1.worker.mamba_utils import (
        get_aligned_state_indices_multi_group_kernel,
        postprocess_mamba_fused_kernel, precopy_mamba_align_fused_kernel)

    i32 = dict(dtype=torch.int32)
    i64 = dict(dtype=torch.int64)
    NS = 4  # num_spec: record row = 1 + req * NS + j - 1

    def rec(req, j):
        return 1 + req * NS + j - 1

    # --- aligned indices: column 0 from the table, 1.. from the request slot -
    block_size, groups, max_reqs = 8, 3, 6
    tables = [torch.arange(max_reqs * 16, **i32).view(max_reqs, 16) + 1000 * (g + 1)
              for g in range(groups)]
    ptrs = torch.tensor([t.data_ptr() for t in tables], **i64)
    seq_lens = torch.tensor([17, 9, 30, 0, 5, 7], **i32)
    # rows 0-2 real; row 3 has no tokens; rows 4-5 are graph padding.
    idx_mapping = torch.tensor([4, 0, 2, 1], **i64)
    num_real, num_padded = 4, 6

    def aligned(columns, idx=None, real=0):
        out = torch.full((groups, max_reqs, 1 + columns), -9, **i32)
        get_aligned_state_indices_multi_group_kernel[(1,)](
            ptrs, seq_lens, out, tables[0].stride(0), seq_lens.stride(0),
            out.stride(0), out.stride(1), out.stride(2), num_padded,
            CACHE_BLOCK_SIZE=block_size, NUM_GROUPS=groups, BLOCK_GROUPS=4,
            NUM_STATE_SLOTS=1 + columns, BLOCK_STATE_SLOTS=8, BLOCK_ROWS=8,
            **({} if idx is None else dict(
                NUM_RECORD_COLUMNS=columns, idx_mapping_ptr=idx,
                num_real_requests=real)))
        return out

    shipped = aligned(NS)  # the shipped window: 1 + num_speculative_blocks
    compact = aligned(NS, idx_mapping, num_real)
    for g in range(groups):
        for row in range(num_padded):
            first = max((int(seq_lens[row]) - 1) // block_size, 0)
            if int(seq_lens[row]) > 0:
                assert compact[g, row, 0] == tables[g][row, first], (g, row)
                window = tables[g][row, first:first + 5].tolist()
                assert shipped[g, row].tolist() == window
            else:
                assert compact[g, row, 0] == 0 and shipped[g, row, 0] == 0
            for j in range(1, NS + 1):
                live = row < num_real and int(seq_lens[row]) > 0
                want = rec(int(idx_mapping[row]), j) if live else 0  # 0 = sink
                assert compact[g, row, j] == want, (g, row, j, compact[g, row])
    # Every live record row is unique, and no padding row names a live one.
    live = [int(v) for v in compact[0, :, 1:].flatten() if int(v)]
    assert len(live) == len(set(live)) == 3 * NS
    # Slot reuse / row reorder: the ids follow the slot, not the batch row.
    swapped = aligned(NS, torch.tensor([2, 0, 4, 1], **i64), num_real)
    assert swapped[0, 0, 1:].tolist() == compact[0, 2, 1:].tolist()

    # --- decision kernels store the request slot under STORE_REQ_IDX ---------
    z64 = torch.zeros(1, **i64); z32 = torch.zeros(1, **i32)
    meta = (z64, 0, z64, z64, z32, z64, z32, z32, z32, z64)
    num_reqs = 3
    bmap = torch.tensor([2, 0, 1], **i32)
    accepted = torch.tensor([3, 1, 5], **i32)
    state_idx = torch.tensor([1, 1, 2], **i32)
    new_computed = torch.tensor([17, 9, 16], **i32)
    src = torch.full((4,), -1, **i32); acc = torch.ones(4, **i32)
    dst = torch.full((4,), -1, **i32); req = torch.full((4,), -5, **i32)
    postprocess_mamba_fused_kernel[(num_reqs, 1, 1)](
        accepted, state_idx, None, new_computed, None, *meta,
        accepted.clone(), bmap, num_reqs,
        block_size=8, COPY_BLOCK_SIZE=16, CONV_STATE_DIM_FIRST=False,
        HAS_IDX_MAPPING=True, PRECOMPUTED_NEW_COMPUTED=True, TEMPORAL_TILES=1,
        DECISION_ONLY=True, commit_src_col_ptr=src, commit_accepted_ptr=acc,
        commit_dst_col_ptr=dst, STORE_REQ_IDX=True, commit_req_idx_ptr=req)
    for b in range(num_reqs):
        if int(src[b]) >= 0:
            assert int(req[b]) == int(bmap[b]), (b, req)
    assert (src >= 0).sum() == 2

    new_col = torch.tensor([3, 2, 5], **i32); old_col = torch.tensor([2, 2, 4], **i32)
    bias = torch.tensor([3, 1, 0], **i32)
    src.fill_(-1); req.fill_(-5)
    precopy_mamba_align_fused_kernel[(num_reqs, 1, 1)](
        new_col, old_col, bias, *meta, bmap, num_reqs,
        COPY_BLOCK_SIZE=16, CONV_STATE_DIM_FIRST=False, HAS_IDX_MAPPING=True,
        TEMPORAL_TILES=1, DECISION_ONLY=True, commit_src_col_ptr=src,
        commit_accepted_ptr=acc, commit_dst_col_ptr=dst,
        STORE_REQ_IDX=True, commit_req_idx_ptr=req)
    assert src.tolist() == [-1, 2, -1, -1] and req.tolist() == [-5, 0, -5, -5]

    K = 3
    cap_tokens = torch.zeros((2, K), **i32); cap_bias = torch.zeros((2, K), **i32)
    cap_tokens[1, 2], cap_bias[1, 2] = 12, 2
    dest_blocks = torch.zeros((4, K, 2), **i32)
    dest_blocks[3, 2] = torch.tensor([50, 51])
    exp_dest = torch.full((2, 4), -1, **i32)
    src.fill_(-1); req.fill_(-5)
    boundary_export_decision_kernel[(2,)](
        torch.tensor([0, 3], **i32), torch.tensor([1, 1, 1, 6], **i32),
        cap_tokens, cap_bias, dest_blocks, src, acc, exp_dest,
        MAX_REQS=4, NUM_GROUPS=2, NUM_CAPTURES=K, KIND=2,
        STORE_REQ_IDX=True, commit_req_idx_ptr=req)
    assert src.tolist() == [-1, 6, -1, -1] and req.tolist() == [-5, 3, -5, -5]
    assert exp_dest[:, 1].tolist() == [50, 51] and int(acc[1]) == 3

    # --- compact gather: one-column window per group, ids from the slot ------
    src = torch.tensor([2, -1, 4, -1], **i32)
    dst = torch.tensor([1, -1, 4, -1], **i32)
    req = torch.tensor([5, 7, 0, 7], **i32)
    for group in range(3):
        # Only bt[src] is a live state; anything past it is poison now that
        # the window holds no speculative blocks.
        table = torch.full((4, 16), -77, **i32)
        table[0, :3] = torch.tensor([10, 11, 12]) + 100 * group
        table[2, :5] = torch.tensor([20, 21, 22, 23, 24]) + 100 * group
        windows = torch.zeros((4, NS + 1), **i32)
        destination = torch.zeros(4, **i32)
        num_seqs = torch.zeros(1, **i32)
        alias = torch.zeros(1, **i32)
        gather_gdn_commit_windows_kernel[(1,)](
            src, dst, table, table.stride(0), windows, destination, num_seqs,
            alias, MAX_REQS=4, BLOCK=4, COLUMNS=NS + 1,
            COMPACT=True, commit_req_idx_ptr=req)
        records = range(1, NS + 1)
        assert windows[0].tolist() == [int(table[0, 2])] + [rec(5, j) for j in records]
        assert windows[2].tolist() == [int(table[2, 4])] + [rec(0, j) for j in records]
        assert windows[1].tolist() == windows[3].tolist() == [0] * (NS + 1)
        assert destination.tolist() == [int(table[0, 1]), -1, int(table[2, 4]), -1]
        assert int(num_seqs) == 4 and int(alias) == 0
    # A destination equal to a record id is just a pool block: no alias count.
    table[0, 1] = rec(5, 1)
    gather_gdn_commit_windows_kernel[(1,)](
        src, dst, table, table.stride(0), windows, destination, num_seqs,
        alias, MAX_REQS=4, BLOCK=4, COLUMNS=NS + 1,
        COMPACT=True, commit_req_idx_ptr=req)
    assert int(destination[0]) == rec(5, 1) and int(alias) == 0
    print("ok")
    """
)


def test_compact_kernels_under_the_triton_interpreter():
    pytest.importorskip("triton")
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    result = subprocess.run(
        [sys.executable, "-c", _COMPACT_INTERPRETED],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith("ok")
