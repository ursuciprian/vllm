# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for deferred GDN checkpoints (VLLM_GDN_DEFERRED_CHECKPOINTS).

The Triton cases run the real kernels under ``TRITON_INTERPRET=1`` in a
subprocess, so they need Triton installed but no GPU.
"""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

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


def test_resolve_is_off_by_default_and_fails_closed(monkeypatch):
    monkeypatch.delenv("VLLM_GDN_DEFERRED_CHECKPOINTS", raising=False)
    assert gdc.resolve(_config(mode="none")) is False
    monkeypatch.setenv("VLLM_GDN_DEFERRED_CHECKPOINTS", "1")
    assert gdc.resolve(_config()) is True
    with pytest.raises(ValueError, match="mamba-cache-mode align"):
        gdc.resolve(_config(mode="all"))


_INTERPRETED = textwrap.dedent(
    """
    import torch
    from vllm.v1.worker.gdn_deferred_commit import gather_gdn_commit_windows_kernel
    from vllm.v1.worker.mamba_utils import postprocess_mamba_fused_kernel

    i32 = dict(dtype=torch.int32)
    block_size, num_reqs = 8, 3
    # Batch row b serves request-state slot idx_mapping[b]; decision inputs are
    # in slot order, the block table in batch order.
    idx_mapping = torch.tensor([2, 0, 1], **i32)
    accepted = torch.tensor([3, 1, 5], **i32)           # slot order
    state_idx = torch.tensor([1, 1, 2], **i32)          # running column
    new_computed = torch.tensor([17, 9, 16], **i32)
    out_accepted = accepted.clone()
    src = torch.full((num_reqs,), -1, **i32)
    acc = torch.ones(num_reqs, **i32)
    dst = torch.full((num_reqs,), -1, **i32)
    zeros64 = torch.zeros(1, dtype=torch.int64)
    zeros32 = torch.zeros(1, **i32)
    postprocess_mamba_fused_kernel[(num_reqs, 1, 1)](
        accepted, state_idx, None, new_computed, None,
        zeros64, 0, zeros64, zeros64, zeros32, zeros64, zeros32, zeros32,
        zeros32, zeros64, out_accepted, idx_mapping, num_reqs,
        block_size=block_size, COPY_BLOCK_SIZE=16, CONV_STATE_DIM_FIRST=False,
        HAS_IDX_MAPPING=True, PRECOMPUTED_NEW_COMPUTED=True, TEMPORAL_TILES=1,
        DECISION_ONLY=True, commit_src_col_ptr=src, commit_accepted_ptr=acc,
        commit_dst_col_ptr=dst,
    )
    # Reference, per batch row, from the kernel's own decision rule.
    for b in range(num_reqs):
        r = int(idx_mapping[b])
        running = int(new_computed[r]) - int(accepted[r]) + 1
        aligned = int(new_computed[r]) // block_size * block_size
        dest = aligned // block_size - 1
        bias = aligned - running
        noop = aligned < running or (dest == int(state_idx[r]) and bias == 0)
        if noop:
            assert int(src[b]) == -1, (b, src)
            continue
        assert int(src[b]) == int(state_idx[r]), (b, src, state_idx)
        assert int(acc[b]) == bias + 1, (b, acc)
        assert int(dst[b]) == dest and dest <= int(src[b]), (b, dst)
    assert (src >= 0).any(), "case degenerated to no boundary crossing"

    # Gather: window from the source column, destination from dst_col.
    table = torch.arange(3 * 16, **i32).view(3, 16) + 100
    windows = torch.zeros((3, 5), **i32)
    destination = torch.zeros(3, **i32)
    src2 = torch.tensor([2, -1, 4], **i32)
    dst2 = torch.tensor([1, -1, 4], **i32)
    gather_gdn_commit_windows_kernel[(3,)](
        src2, dst2, table, table.stride(0), windows, destination, 3, COLUMNS=5
    )
    assert windows[0].tolist() == table[0, 2:7].tolist()
    assert windows[2].tolist() == table[2, 4:9].tolist()
    assert destination.tolist() == [int(table[0, 1]), -1, int(table[2, 4])]
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
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith("ok")
