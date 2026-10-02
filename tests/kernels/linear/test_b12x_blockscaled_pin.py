# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.model_executor.kernels.linear.b12x_blockscaled import parse_blockscaled_pins


def test_parse_blockscaled_pins():
    assert parse_blockscaled_pins("") == []
    assert parse_blockscaled_pins("2560x6144@16=a16:64:128:8; 2560x3072@8=a16:64:64:8") == [
        (2560, 6144, 16, {"mode": "a16", "tile_n": 64, "tile_k": 128, "split_k": 8}),
        (2560, 3072, 8, {"mode": "a16", "tile_n": 64, "tile_k": 64, "split_k": 8}),
    ]
    for bad in ("2560x6144=a16:64:128:8", "2560x6144@16=quantized", "2560x6144@16=a16:64:8"):
        with pytest.raises(ValueError):
            parse_blockscaled_pins(bad)
