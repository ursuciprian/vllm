# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in b12x SIMT GEMV for small-batch BF16 projections (``VLLM_QWEN38_B12X_GEMV``).

The MTP draft layer's attention projections and every MoE router gate are
unquantized BF16 in the checkpoint, so decode runs them through cuBLAS small-M
kernels at 150-170 GB/s on GB10 (r8 TP=1 profile: draft q|k|v [13312, 2560] 417 us
per M=1 draft pass). b12x's SIMT GEMV with ``rows_per_tile=8`` streams the same
weight once for up to 8 rows at ~240 GB/s (scripts/bench_draft_gemv.py: q|k|v
M=1 394 -> 278 us, M=5 292 -> 281 us).

Same math: BF16 operands, FP32 accumulation, BF16 output; only the summation
order differs from cuBLAS (relative error ~3e-5). Rows > 8, non-BF16 or
non-contiguous inputs take the stock ``F.linear`` inside the op.

Targets (comma list): ``mtp`` = MTP draft ``self_attn.qkv_proj`` / ``o_proj``;
``gate`` = MoE router gates (target and draft layers). The variable is declared
in ``vllm/envs.py``, so it is part of the torch AOT compile-cache key.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

logger = init_logger(__name__)

MAX_ROWS = 8  # rows sharing one weight read in the rows_per_tile=8 SIMT kernel
_LAUNCHERS: dict[str, object] = {}


def _targets() -> frozenset[str]:
    value = (envs.VLLM_QWEN38_B12X_GEMV or "off").lower()
    return frozenset() if value == "off" else frozenset(value.split(","))


class B12xGemvLinearMethod(UnquantizedLinearMethod):
    """Stock BF16 weights and loaders; decode rows <= 8 run b12x's SIMT GEMV."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.op_name = _encode_layer_name(name)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        w = layer.weight
        if not (w.is_cuda and w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_contiguous()):
            logger.warning("qwen38 b12x GEMV: %s weight %s %s not admitted, keeping cuBLAS",
                           self.name, tuple(w.shape), w.dtype)
            return
        from b12x.gemm.bf16_gemv._kernel import compile_projection

        n, k = (int(v) for v in w.shape)
        _LAUNCHERS[self.name] = compile_projection(
            w.device.index or 0, "simt", MAX_ROWS, n, k,
            "bfloat16", "bfloat16", "bfloat16", None,
        )
        logger.info_once("qwen38 b12x GEMV: %s [%d, %d] -> b12x simt rows<=%d", self.name,
                         n, k, MAX_ROWS)

    def apply(self, layer: nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        if bias is not None or self.name not in _LAUNCHERS:
            return super().apply(layer, x, bias)
        return torch.ops.vllm.qwen38_b12x_gemv(x, layer.weight, self.op_name)


def maybe_route_b12x_gemv(linear: nn.Module, target: str) -> bool:
    """Rebind an unquantized ``linear`` to the b12x GEMV when ``target`` is enabled."""
    if target not in _targets() or not current_platform.is_cuda():
        return False
    method = getattr(linear, "quant_method", None)
    if type(method) is not UnquantizedLinearMethod:
        logger.warning_once("qwen38 b12x GEMV: target %s has %s, not unquantized: skipped",
                            target, type(method).__name__)
        return False
    prefix = getattr(linear, "prefix", "")
    if not prefix:  # the op's layer name is baked into AOT graphs: it must be stable
        return False
    linear.quant_method = B12xGemvLinearMethod(prefix)
    return True


def _gemv(x: torch.Tensor, weight: torch.Tensor, layer_name: LayerNameType) -> torch.Tensor:
    launch = _LAUNCHERS.get(_resolve_layer_name(layer_name))
    if (launch is None or x.dim() != 2 or x.shape[0] > MAX_ROWS or x.dtype != torch.bfloat16
            or not x.is_contiguous()):
        return F.linear(x, weight)
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    if x.shape[0]:
        launch(x, weight, out, None)
    return out


def _gemv_fake(x: torch.Tensor, weight: torch.Tensor, layer_name: LayerNameType) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen38_b12x_gemv",
    op_func=_gemv,
    mutates_args=[],
    fake_impl=_gemv_fake,
)
