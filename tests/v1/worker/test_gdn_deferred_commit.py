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
    assert len(gdc.refuse_reasons(_config(boundary=True))) == 1
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
        gather_gdn_commit_windows_kernel[(1,)](
            src, dst, table, table.stride(0), windows, destination, num_seqs,
            MAX_REQS=4, BLOCK=4, COLUMNS=5)
        assert windows[0].tolist() == table[0, 2:7].tolist()
        assert windows[2].tolist() == table[2, 4:9].tolist()
        assert destination.tolist() == [int(table[0, 1]), -1, int(table[2, 4]), -1]
        assert int(num_seqs) == 4
    none = torch.full((4,), -1, **i32)
    gather_gdn_commit_windows_kernel[(1,)](
        none, none, table, table.stride(0), windows, destination, num_seqs,
        MAX_REQS=4, BLOCK=4, COLUMNS=5)
    assert int(num_seqs) == 0 and destination.tolist() == [-1] * 4
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
    assert commit._graph is not None or commit._graph_failed
