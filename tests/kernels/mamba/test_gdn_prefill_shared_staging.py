# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_GDN_SHARED_PREFILL_STAGING: one GDN prefill staging per shape and device."""

import torch

import vllm.envs as envs
from vllm.model_executor.layers.mamba.ops import b12x_gdn_prefill as gp


def _alloc(max_tokens=64):
    return gp.GdnPrefillStaging.allocate(
        max_tokens=max_tokens, max_seqs=4, key_heads=2, value_heads=3,
        device=torch.device("cpu"),
    )


def test_off_allocates_one_staging_per_call(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_GDN_SHARED_PREFILL_STAGING", "0")
    gp._SHARED_STAGING.clear()
    a, b = _alloc(), _alloc()
    assert a is not b
    assert a.mixed_qkv.data_ptr() != b.mixed_qkv.data_ptr()
    assert not gp._SHARED_STAGING


def test_on_shares_one_staging_per_shape(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_GDN_SHARED_PREFILL_STAGING", "1")
    gp._SHARED_STAGING.clear()
    a, b = _alloc(), _alloc()
    assert a is b
    assert a.is_compatible(max_tokens=64, max_seqs=4, key_heads=2, value_heads=3,
                           device=torch.device("cpu"))
    assert a.mixed_qkv.shape == (64, (2 * 2 + 3) * 128)
    assert _alloc(max_tokens=128) is not a
    assert len(gp._SHARED_STAGING) == 2
    gp._SHARED_STAGING.clear()


def test_knob_is_not_a_compile_factor(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_GDN_SHARED_PREFILL_STAGING", "1")
    on = envs.compile_factors()
    monkeypatch.setenv("VLLM_GDN_SHARED_PREFILL_STAGING", "0")
    assert envs.compile_factors() == on
