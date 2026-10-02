# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The b12x GEMV route must match F.linear at decode rows and fall back above them."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("b12x")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from vllm.models.qwen3_8_flash_next import b12x_gemv  # noqa: E402


@pytest.mark.parametrize("n,k", [(13312, 2560), (512, 2560)])
def test_gemv_matches_linear_and_falls_back(n, k):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16), requires_grad=False)
    method = b12x_gemv.B12xGemvLinearMethod(f"test.{n}x{k}")
    method.process_weights_after_loading(layer)
    assert method.name in b12x_gemv._LAUNCHERS
    for m in (1, 2, 5, 8, 9, 64):
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        exact = F.linear(x.float(), layer.weight.float())

        def err(y):
            return ((y.float() - exact).norm() / exact.norm()).item()

        out = method.apply(layer, x)
        if m > b12x_gemv.MAX_ROWS:  # fallback is the stock F.linear call
            assert torch.equal(out, F.linear(x, layer.weight)), m
            continue
        # FP32 accumulation, one BF16 rounding of the output: b12x sits on the
        # BF16 rounding floor. Do not compare against cuBLAS: its split-K router
        # GEMM (N=512, M>=2) adds BF16 partials and is itself ~2.4e-3 off.
        assert err(out) <= 1.05 * err(exact.to(torch.bfloat16)), (m, err(out))
