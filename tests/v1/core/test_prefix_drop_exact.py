# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_PREFIX_DROP_EXACT: EAGLE/MTP keeps a hit's last full block when the
request's next token equals the token the cached drafter KV was written from.

Shape mirrors Qwen3.8-Flash-Next on the DGX pair: attention and GDN pages both
2864 tokens (align mode), MTP with one prefill lookahead token, prefill budget
8192, llama-benchy depth test = 16384 cached context + 2048 new tokens.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

from .test_prefix_caching import make_kv_cache_manager, make_request

pytestmark = pytest.mark.cpu_test

B = 2864
BUDGET = 8192
CONTEXT = 16384
NEW = 2048


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _manager(monkeypatch, exact: bool, use_eagle: bool = True, lookahead: int = 1):
    monkeypatch.setenv("VLLM_PREFIX_DROP_EXACT", "1" if exact else "0")
    config = KVCacheConfig(
        num_blocks=200,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attn"],
                FullAttentionSpec(
                    block_size=B, num_kv_heads=1, head_size=1, dtype=torch.float16
                ),
            ),
            KVCacheGroupSpec(
                ["gdn"],
                MambaSpec(
                    block_size=B,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=4 if use_eagle else 0,
                ),
            ),
        ],
    )
    return make_kv_cache_manager(
        config,
        max_model_len=262144,
        enable_caching=True,
        hash_block_size=B,
        use_eagle=use_eagle,
        num_prefill_lookahead=lookahead if use_eagle else 0,
    )


def _prefill(manager, request, use_eagle: bool = True) -> list[int]:
    """Prefill like the scheduler: 8192-token budget, align-mode split."""
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=B, prefix_cache_retention_interval=None
        ),
        # Scheduler.__init__ clears the back-off when the coordinator is exact.
        drop_last_prefix_cache_block=use_eagle
        and not manager.coordinator.prefix_drop_exact,
        use_eagle=use_eagle,
        max_num_scheduled_tokens=BUDGET,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        hash_block_size=B,
        mamba_has_prefill_checkpoint_blocks=False,
    )
    computed, num_computed, _ = manager.get_computed_blocks(request)
    request.num_computed_tokens = num_computed
    ends = []
    while request.num_computed_tokens < request.num_tokens:
        n = min(BUDGET, request.num_tokens - request.num_computed_tokens)
        n = Scheduler._mamba_block_aligned_split(stub, request, n)
        assert manager.allocate_slots(
            request,
            n,
            num_computed if not ends else 0,
            computed if not ends else None,
            num_lookahead_tokens=4 if use_eagle else 0,
        )
        request.num_computed_tokens += n
        ends.append(request.num_computed_tokens)
    return ends


def _decode_one(manager, request, token: int) -> None:
    request.append_output_token_ids(token)
    assert manager.allocate_slots(request, 1, num_lookahead_tokens=4)
    request.num_computed_tokens += 1


def _hit(manager, token_ids: list[int], rid: str) -> int:
    request = make_request(rid, token_ids, B, sha256)
    return manager.get_computed_blocks(request)[1]


def _context() -> list[int]:
    return [i % 5000 for i in range(CONTEXT)]


@pytest.mark.parametrize(
    ("exact", "expected_ends", "expected_hit"),
    [
        (False, [2 * B, 4 * B, CONTEXT], 4 * B),  # 11456: today's behavior
        (True, [2 * B, 4 * B, 5 * B, CONTEXT], 5 * B),  # 14320
    ],
)
def test_depth_16k_plus_2k(monkeypatch, exact, expected_ends, expected_hit):
    manager = _manager(monkeypatch, exact)
    assert manager.coordinator.prefix_drop_exact is exact
    ctx = _context()
    warm = make_request("warm", ctx, B, sha256)
    assert _prefill(manager, warm) == expected_ends
    manager.free(warm)
    new = [7000 + i for i in range(NEW)]
    assert _hit(manager, ctx + new, "depth") == expected_hit


def test_divergent_next_token_falls_back(monkeypatch):
    """Shared exactly up to 14320, different token 14320: drop as before."""
    manager = _manager(monkeypatch, exact=True)
    ctx = _context()
    warm = make_request("warm", ctx, B, sha256)
    _prefill(manager, warm)
    manager.free(warm)
    diverged = ctx[: 5 * B] + [9999] + ctx[5 * B + 1 :]
    assert _hit(manager, diverged, "div") == 4 * B
    # One token later is enough: the block's next token (14320) matches.
    diverged = ctx[: 5 * B + 1] + [9999] + ctx[5 * B + 2 :]
    assert _hit(manager, diverged, "div2") == 5 * B


@pytest.mark.parametrize("exact", [False, True])
def test_prompt_ends_on_boundary(monkeypatch, exact):
    """Next token unknown at cache time; recorded once the output is sampled."""
    manager = _manager(monkeypatch, exact)
    prompt = _context()[: 5 * B]
    req = make_request("warm", prompt, B, sha256)
    assert _prefill(manager, req)[-1] == 5 * B
    # Before sampling, the last block's drafter KV is not determined yet.
    assert _hit(manager, prompt + [42] * 100, "early") == 4 * B
    _decode_one(manager, req, 42)
    assert _hit(manager, prompt + [42] * 100, "same") == (5 * B if exact else 4 * B)
    assert _hit(manager, prompt + [43] * 100, "other") == 4 * B
    manager.free(req)


def test_exact_multiple_prompt_resend(monkeypatch):
    """Identical resend of a 3-block prompt: the last token is always recomputed."""
    manager = _manager(monkeypatch, exact=True)
    prompt = _context()[: 3 * B]
    req = make_request("warm", prompt, B, sha256)
    _prefill(manager, req)
    manager.free(req)
    # max hit = 3B - 1 -> two full blocks; block 1's next token is known.
    assert _hit(manager, prompt, "resend") == 2 * B


def test_no_mtp_unchanged(monkeypatch):
    """Without EAGLE/MTP there is no drop, and the flag resolves off."""
    for exact in (False, True):
        manager = _manager(monkeypatch, exact, use_eagle=False)
        assert manager.coordinator.prefix_drop_exact is False
        ctx = _context()
        warm = make_request("warm", ctx, B, sha256)
        _prefill(manager, warm, use_eagle=False)
        manager.free(warm)
        assert _hit(manager, ctx + [1] * NEW, "depth") == 5 * B


def test_multi_module_mtp_keeps_drop(monkeypatch):
    manager = _manager(monkeypatch, exact=True, lookahead=4)
    assert manager.coordinator.prefix_drop_exact is False


def test_eviction_clears_next_token(monkeypatch):
    manager = _manager(monkeypatch, exact=True)
    ctx = _context()
    warm = make_request("warm", ctx, B, sha256)
    _prefill(manager, warm)
    block = manager.coordinator.single_type_managers[0].req_to_blocks["warm"][4]
    assert block.eagle_next_token == ctx[5 * B]
    block.reset_hash()
    assert block.eagle_next_token is None
