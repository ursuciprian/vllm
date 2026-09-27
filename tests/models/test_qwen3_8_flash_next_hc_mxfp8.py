# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HcMxfp8LinearMethod must report its b12x linear's scratch to side-stream callers."""

import types

import torch

from vllm.models.qwen3_8_flash_next.hyperconnection import HcMxfp8LinearMethod


def test_engaged_method_reports_kernel_workspace() -> None:
    calls = []
    inner = types.SimpleNamespace(
        get_workspace_size=lambda layer, rows: calls.append(rows) or 4096 * rows
    )
    method = HcMxfp8LinearMethod(inner, "mtp")
    layer = torch.nn.Module()

    # Not engaged (geometry kept BF16): the unquantized path needs no scratch.
    assert method.get_workspace_size(layer, 5) == 0
    assert calls == []

    method.engaged = True
    assert method.get_workspace_size(layer, 5) == 4096 * 5
    assert calls == [5]
