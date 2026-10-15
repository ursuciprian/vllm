# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba-aligned prefill splits as the 1x Qwen3.8 Flash Next serving recipe
schedules them (align mode, 3024-token state blocks, MTP reuse positions one
block below the prompt's last boundary, sparse retention), with
VLLM_DEBUG_PREFILL_SPLITS naming each stop.

The 8192 and 9072 schedules are the ones the k81 and k88 profiles recorded; the
16K/64K schedules at all three budgets match the k97 split log.
"""

import logging
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm import envs
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK = 3024


def _scheduler(budget, retention=0):
    manager = SimpleNamespace(
        kv_cache_spec=MagicMock(spec=MambaSpec),
        use_eagle=True,
        lookup_drops_eagle_block=False,
        hit_alignment_tokens=BLOCK,
        scheduler_block_size=BLOCK,
        block_size=BLOCK,
    )
    manager._expand_reachable_boundaries = types.MethodType(
        SingleTypeKVCacheManager._expand_reachable_boundaries, manager
    )
    sched = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=BLOCK, prefix_cache_retention_interval=retention
        ),
        drop_last_prefix_cache_block=False,
        mamba_has_prefill_checkpoint_blocks=False,
        use_eagle=True,
        max_num_scheduled_tokens=budget,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        hash_block_size=BLOCK,
        mamba_partial_cache_hit=False,
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=[manager])
        ),
    )
    return sched


def chunks(prompt, budget, retention=0):
    """Cold prompt alone in the batch: each step offers min(budget, rest)."""
    sched, out, start = _scheduler(budget, retention), [], 0
    while start < prompt:
        request = SimpleNamespace(
            request_id="r",
            num_computed_tokens=start,
            num_prompt_tokens=prompt,
            num_tokens=prompt,
            use_boundary_checkpoints=False,
            shared_prefix_boundary=0,
        )
        n = Scheduler._mamba_block_aligned_split(
            sched, request, min(budget, prompt - start)
        )
        assert n > 0
        out.append(n)
        start += n
    return out


def test_reproduces_the_profiled_16k_schedules():
    assert chunks(16384, 8192) == [6048, 6048, 3024, 1264]  # k81
    assert chunks(16384, 9072) == [9072, 3024, 3024, 1264]  # k88


@pytest.mark.parametrize(
    "prompt,budget,steps",
    [
        (16384, 12096, [12096, 3024, 1264]),
        (65536, 8192, [6048] * 10 + [3024, 2032]),
        (65536, 9072, [9072] * 6 + [6048, 3024, 2032]),
        (65536, 12096, [12096] * 5 + [3024, 2032]),
        (131072, 8192, [6048] * 21 + [3024, 1040]),
        (131072, 12096, [12096] * 10 + [6048, 3024, 1040]),
    ],
)
def test_predicted_schedules(prompt, budget, steps):
    assert chunks(prompt, budget) == steps


def test_debug_log_names_each_stop(monkeypatch, caplog):
    monkeypatch.setattr(envs, "VLLM_DEBUG_PREFILL_SPLITS", True, raising=False)
    with caplog.at_level(logging.INFO, logger="vllm.v1.core.sched.scheduler"):
        chunks(16384, 9072)
    reasons = [
        line.split("reason=")[1].split()[0]
        for line in caplog.messages
        if line.startswith("prefill split")
    ]
    # k97 labels: budget, reuse @12096, last_cache @15120, prompt end
    assert reasons == ["budget", "reachable", "last-cache-position", "prompt-end"]


def test_debug_log_is_off_by_default(monkeypatch, caplog):
    monkeypatch.setattr(envs, "VLLM_DEBUG_PREFILL_SPLITS", False, raising=False)
    with caplog.at_level(logging.INFO):
        chunks(16384, 9072)
    assert not [
        m for m in caplog.messages if m.startswith(("prefill split", "reachable"))
    ]
