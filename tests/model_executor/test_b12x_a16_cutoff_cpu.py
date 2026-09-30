# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_B12X_A16_MAX_TOKENS reaches b12x planning only when set (host-only)."""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear import b12x_blockscaled
from vllm.model_executor.layers.fused_moe import b12x as moe_b12x
from vllm.utils.b12x import B12xWorkload


def _linear(recipe):
    t = torch.empty(0)
    packed = SimpleNamespace(
        values=t, scale_mma=t, global_scale=t, global_scale_kind="tensor",
        in_features=256, out_features=128, padded_in_features=256,
        weight=SimpleNamespace(values=t, scale_mma=t),
    )
    return b12x_blockscaled.B12xBlockscaledLinear(
        packed, recipe=recipe, activation_mode="auto", layer_name="l"
    )


WORKLOAD = B12xWorkload(
    stage="weights", token_counts=(5, 10), fixed_token_counts=(5, 10),
    output_dtype=torch.bfloat16, max_tokens=8192, max_seqs=16, max_model_len=4096,
)


@pytest.mark.parametrize(
    "env,recipe,expected",
    (("0", "nvfp4", None), ("8", "nvfp4", 8), ("24", "nvfp4", 24), ("8", "mxfp8", None)),
)
def test_dense_cutoff_kwarg(monkeypatch, env, recipe, expected):
    monkeypatch.setenv("VLLM_B12X_A16_MAX_TOKENS", env)
    envs.disable_envs_cache()
    seen = {}
    api = SimpleNamespace(
        BlockscaledQuery=lambda **kw: kw,
        plan_regimes=lambda query, **kw: seen.update(kw) or "plan",
    )
    monkeypatch.setattr(b12x_blockscaled, "get_b12x_blockscaled", lambda: api)
    lin = _linear(recipe)
    assert lin.ensure_plan(WORKLOAD) == "plan"
    assert seen.get("a16_max_tokens") == expected
    assert seen["exact_m"] == (5, 10)
    # the preparation-unit key must change with the cutoff (plan-cache identity)
    assert lin.signature(WORKLOAD)[-1] == (expected or 0)


@pytest.mark.parametrize(
    "env,mode,fmt,dtype,expected",
    (
        ("0", "nvfp4", "modelopt_nvfp4", torch.bfloat16, {}),
        ("8", "nvfp4", "modelopt_nvfp4", torch.bfloat16, {"a16_max_tokens": 8}),
        ("24", "w4a8_nvfp4", "modelopt_nvfp4", torch.bfloat16, {"a16_max_tokens": 24}),
        ("8", "w4a16", "modelopt_nvfp4", torch.bfloat16, {}),
        ("8", "nvfp4", "fp4_e8m0_k32", torch.bfloat16, {}),
        ("8", "nvfp4", "modelopt_nvfp4", torch.float16, {}),
    ),
)
def test_moe_cutoff_kwargs(monkeypatch, env, mode, fmt, dtype, expected):
    monkeypatch.setenv("VLLM_B12X_A16_MAX_TOKENS", env)
    envs.disable_envs_cache()
    assert moe_b12x._a16_cutoff_kwargs(mode, fmt, dtype) == expected


def test_cutoff_is_a_compile_factor(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_B12X_A16_MAX_TOKENS", "0")
    off = envs.compile_factors()
    monkeypatch.setenv("VLLM_B12X_A16_MAX_TOKENS", "8")
    assert envs.compile_factors() != off
