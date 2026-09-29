# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_VERIFY_TOPK_TRITON: Triton top-k/top-p dispatch below 8 rows (CPU).

Triton-vs-sort equivalence at batch size 1 is already covered on GPU by
tests/v1/sample/test_topk_topp_sampler.py.
"""

import pytest
import torch

import vllm.v1.sample.ops.topk_topp_sampler as tts


@pytest.fixture
def cuda_platform(monkeypatch):
    monkeypatch.setattr(tts.current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(tts, "HAS_TRITON", True)


@pytest.mark.parametrize(
    "env,dtype,rows,expect_triton",
    [
        ("1", torch.float32, 5, True),
        ("0", torch.float32, 5, False),
        ("1", torch.bfloat16, 5, False),  # Triton kernel is FP32 only
        ("0", torch.float32, 8, True),  # stock threshold unchanged
    ],
)
def test_dispatch(cuda_platform, monkeypatch, env, dtype, rows, expect_triton):
    monkeypatch.setenv("VLLM_VERIFY_TOPK_TRITON", env)
    calls = []
    monkeypatch.setattr(
        tts,
        "apply_top_k_top_p_triton",
        lambda x, k, p: calls.append("t") or x,
        raising=False,  # only imported when a Triton driver is active
    )
    monkeypatch.setattr(
        tts, "apply_top_k_top_p_pytorch", lambda x, k, p: calls.append("s") or x
    )
    logits = torch.randn(rows, 64, dtype=dtype)
    k = torch.full((rows,), 20, dtype=torch.int32)
    tts.apply_top_k_top_p(logits, k, None)
    assert calls == (["t"] if expect_triton else ["s"])
