# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for VLLM_NVFP4_MOE_MARLIN: ModelOpt MIXED_PRECISION NVFP4 routed experts
go weight-only to Marlin (BF16 activations); W4A16_NVFP4 experts and the layer's own
--moe-backend are untouched; off by default."""

import torch

import vllm.envs as envs
import vllm.model_executor.layers.quantization.modelopt as modelopt
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

LAYERS = {
    "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
    "mtp.layers.48.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
}


def _config():
    return modelopt.ModelOptMixedPrecisionConfig.from_config(
        {"quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": LAYERS}}
    )


def _experts():
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = FusedMoEConfig(
        num_experts=512,
        experts_per_token=10,
        hidden_dim=2560,
        intermediate_size=640,
        num_local_experts=512,
        num_logical_experts=512,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        moe_backend="b12x",
    )
    return layer


def _resolve(monkeypatch, knob, prefix):
    seen = []

    def fake_select(config, weight_key, activation_key):
        seen.append((config.moe_backend, activation_key))
        return NvFp4MoeBackend.MARLIN, None

    monkeypatch.setattr(modelopt, "select_nvfp4_moe_backend", fake_select)
    monkeypatch.setenv("VLLM_NVFP4_MOE_MARLIN", "1" if knob else "0")
    layer = _experts()
    method = _config().get_quant_method(layer, prefix)
    assert layer.moe_config.moe_backend == "b12x"  # the layer config is never changed
    return method, seen[0]


def test_main_experts_marlin_a16_when_on(monkeypatch):
    m, (backend, act) = _resolve(monkeypatch, True, "language_model.model.layers.0.mlp.experts")
    assert isinstance(m, modelopt.ModelOptNvFp4FusedMoE)
    assert backend == "marlin" and act is None and m.use_a16
    assert m.quant_config.group_size == 16


def test_main_experts_unchanged_when_off(monkeypatch):
    m, (backend, act) = _resolve(monkeypatch, False, "language_model.model.layers.0.mlp.experts")
    assert backend == "b12x" and act is not None and not m.use_a16


def test_mtp_experts_keep_backend(monkeypatch):
    m, (backend, act) = _resolve(monkeypatch, True, "mtp.layers.48.mlp.experts")
    assert backend == "b12x" and m.use_a16


def test_knob_declared():
    assert "VLLM_NVFP4_MOE_MARLIN" in envs.environment_variables
