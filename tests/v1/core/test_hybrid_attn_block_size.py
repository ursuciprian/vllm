# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_HYBRID_ATTN_BLOCK_SIZE: attention blocks smaller than the mamba page.

Shape mirrors Qwen3.8-Flash-Next at TP1 with compact GDN records: a legacy
attention block of 3024 tokens (one 3,290,112-byte page, the padded GDN page),
13 attention layers, MTP with one prefill lookahead token, prefill budget 8192.
With the knob at 232 the 13 attention layers share one group of 232-token
pages inside that same pool block and mamba checkpoints every 3016 tokens.
"""

import contextlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.platforms.interface import Platform
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from .test_prefix_caching import make_kv_cache_manager, make_request

pytestmark = pytest.mark.cpu_test

LEGACY = 3024
N = 232
M = 3016  # 13 * N
ATTN_LAYERS = 13
MAMBA_GROUPS = 4
BUDGET = 8192
CONTEXT = 16384
NEW = 2048
LOOKAHEAD = 4
POOL_BLOCK = LEGACY * 1088  # 3,290,112


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _config(split: bool, num_blocks: int) -> KVCacheConfig:
    mamba = MambaSpec(
        block_size=M if split else LEGACY,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=0,
    )
    attn = FullAttentionSpec(
        block_size=N if split else LEGACY,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
    )
    names = [f"attn.{i}" for i in range(ATTN_LAYERS)]
    attn_groups = (
        [KVCacheGroupSpec(names, attn)]
        if split
        else [KVCacheGroupSpec([name], attn) for name in names]
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            *(KVCacheGroupSpec([f"gdn.{i}"], mamba) for i in range(MAMBA_GROUPS)),
            *attn_groups,
        ],
    )


@pytest.fixture(params=[None, 0], ids=["dense", "retention0"])
def retention(request):
    """Dense mamba checkpoints, and the fork's default sparse retention (0)."""
    return request.param


def _manager(monkeypatch, split: bool, retention, num_blocks: int = 4000):
    monkeypatch.setenv("VLLM_HYBRID_ATTN_BLOCK_SIZE", str(N) if split else "0")
    monkeypatch.setenv("VLLM_PREFIX_DROP_EXACT", "1")
    return make_kv_cache_manager(
        _config(split, num_blocks),
        max_model_len=262144,
        enable_caching=True,
        hash_block_size=N if split else LEGACY,
        use_eagle=True,
        num_prefill_lookahead=1,
        retention_interval=retention,
    )


def _stub(manager, split: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=N if split else LEGACY,
            prefix_cache_retention_interval=getattr(
                manager.coordinator, "retention_interval", None
            ),
        ),
        block_size=M if split else LEGACY,  # scheduler block: LCM of group blocks
        kv_cache_manager=manager,
        drop_last_prefix_cache_block=not manager.coordinator.prefix_drop_exact,
        use_eagle=True,
        max_num_scheduled_tokens=BUDGET,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        hash_block_size=N if split else LEGACY,
        mamba_has_prefill_checkpoint_blocks=False,
    )


def _used(manager) -> int:
    return (
        manager.block_pool.num_gpu_blocks - 1 - manager.block_pool.get_num_free_blocks()
    )


def _prefill(manager, request, split: bool) -> tuple[list[int], int] | None:
    """Prefill like the scheduler; returns (chunk ends, peak pool use) or None."""
    stub = _stub(manager, split)
    computed, hit, _ = manager.get_computed_blocks(request)
    ends: list[int] = []
    peak = 0
    start = hit
    while start < request.num_tokens:
        manager.new_step_starts()
        local = hit if not ends else 0
        n = min(BUDGET, request.num_tokens - start)
        n = Scheduler._mamba_block_aligned_split(stub, request, n, local)
        if (
            manager.allocate_slots(
                request,
                n,
                local,
                computed if not ends else None,
                num_lookahead_tokens=LOOKAHEAD,
                has_scheduled_reqs=False,
            )
            is None
        ):
            return None
        start += n
        request.num_computed_tokens = start
        ends.append(start)
        peak = max(peak, _used(manager))
    return ends, peak


def _decode(manager, request, token: int) -> None:
    manager.new_step_starts()
    request.append_output_token_ids(token)
    assert manager.allocate_slots(request, 1, num_lookahead_tokens=LOOKAHEAD)
    request.num_computed_tokens += 1


def _held(manager, request_id: str) -> tuple[int, int]:
    """Non-null (mamba blocks per group, attention blocks per group)."""
    managers = manager.coordinator.single_type_managers
    counts = [
        sum(not b.is_null for b in m.req_to_blocks.get(request_id, []))
        for m in managers
    ]
    mamba = {
        c for m, c in zip(managers, counts) if isinstance(m.kv_cache_spec, MambaSpec)
    }
    attn = {
        c
        for m, c in zip(managers, counts)
        if not isinstance(m.kv_cache_spec, MambaSpec)
    }
    assert len(mamba) == 1 and len(attn) == 1
    return mamba.pop(), attn.pop()


def _context() -> list[int]:
    return [i % 5000 for i in range(CONTEXT)]


def _request(rid: str, tokens: list[int], split: bool):
    return make_request(rid, tokens, N if split else LEGACY, sha256)


def test_chunks_end_on_the_mamba_block(monkeypatch) -> None:
    """Prefill chunks align to 3016, not to the 232-token attention block."""
    monkeypatch.setenv("VLLM_HYBRID_ATTN_BLOCK_SIZE", str(N))
    manager = SimpleNamespace(coordinator=SimpleNamespace(prefix_drop_exact=True))
    request = _request("r", _context() + list(range(NEW)), split=True)
    stub = _stub(manager, split=True)
    ends, start = [], 0
    while start < request.num_tokens:
        n = min(BUDGET, request.num_tokens - start)
        start += Scheduler._mamba_block_aligned_split(stub, request, n)
        request.num_computed_tokens = start
        ends.append(start)
    assert ends == [2 * M, 4 * M, 6 * M, CONTEXT + NEW]


@pytest.mark.parametrize("split", [False, True])
def test_footprint_16k(monkeypatch, split, retention):
    """A 16K + 2K prefill holds cdiv(T, block) blocks per attention group."""
    manager = _manager(monkeypatch, split, retention)
    request = _request("r", _context() + list(range(NEW)), split)
    assert _prefill(manager, request, split) is not None
    mamba, attn = _held(manager, "r")
    assert attn == cdiv(CONTEXT + NEW + LOOKAHEAD, N if split else LEGACY)
    assert attn == (80 if split else 7)
    groups = 1 if split else ATTN_LAYERS
    assert _used(manager) == MAMBA_GROUPS * mamba + groups * attn
    manager.free(request)
    assert _used(manager) == 0


def _two_16k_requests(manager, split: bool) -> int | None:
    """Prefill two unrelated 16K + 2K requests; the peak pool use, or None."""
    peak = 0
    for rid, base in (("a", 0), ("b", 7)):
        tokens = [base + t for t in _context()] + list(range(NEW))
        result = _prefill(manager, _request(rid, tokens, split), split)
        if result is None:
            return None
        peak = max(peak, result[1])
    return peak


def test_admission_at_16k(monkeypatch, retention):
    """A pool sized for two 16K requests with 232-token blocks admits only
    one with 3024-token blocks."""
    peaks = {
        split: _two_16k_requests(_manager(monkeypatch, split, retention), split)
        for split in (False, True)
    }
    assert peaks[True] < peaks[False]
    pool = peaks[True] + 1  # + the null block
    assert (
        _two_16k_requests(_manager(monkeypatch, True, retention, pool), True)
        == peaks[True]
    )
    assert (
        _two_16k_requests(_manager(monkeypatch, False, retention, pool), False) is None
    )


def test_prefix_hit_lands_on_the_mamba_block(monkeypatch, retention):
    """Hits stay mamba-block aligned (partial hash hits off) and reuse the
    warm request's own attention blocks."""
    manager = _manager(monkeypatch, True, retention)
    assert manager.coordinator.enable_partial_hash_hits is False
    ctx = _context()
    warm = _request("warm", ctx, split=True)
    ends, _ = _prefill(manager, warm, split=True)
    assert ends == [2 * M, 4 * M, 5 * M, CONTEXT]
    attn_manager = manager.coordinator.single_type_managers[MAMBA_GROUPS]
    warm_attn = [b.block_id for b in attn_manager.req_to_blocks["warm"]]
    manager.free(warm)
    request = _request("depth", ctx + [7000 + i for i in range(NEW)], split=True)
    blocks, hit, _ = manager.get_computed_blocks(request)
    assert hit == 5 * M
    hit_attn = [b.block_id for b in blocks.blocks[MAMBA_GROUPS]]
    assert hit_attn == warm_attn[: 5 * M // N]
    # The token after the hit differs: the exact-drop check falls back a block.
    diverged = ctx[: 5 * M] + [9999] + ctx[5 * M + 1 :]
    assert manager.get_computed_blocks(_request("d", diverged, True))[1] == 4 * M


@pytest.mark.parametrize("split", [False, True])
def test_boundary_steps(monkeypatch, split, retention):
    """Decode across a mamba boundary: a group holds one running block, two on
    the crossing step; attention grows by one block every block of tokens."""
    block = M if split else LEGACY
    attn_block = N if split else LEGACY
    manager = _manager(monkeypatch, split, retention)
    request = _request("r", _context()[: block - 3], split)
    assert _prefill(manager, request, split) is not None
    seen = set()
    for token in range(2 * N):
        _decode(manager, request, 8000 + token)
        mamba, attn = _held(manager, "r")
        seen.add(mamba)
        assert attn == cdiv(request.num_computed_tokens + LOOKAHEAD, attn_block)
    assert seen == {1, 2}
    manager.free(request)
    assert _used(manager) == 0


def test_eviction(monkeypatch, retention):
    """In a 300-block pool, a 15 x 3016-token request leaves the cached 16K
    prefix whole (hit 5 x 3016); a 20 x 3016-token one evicts it (hit 0).
    Every block returns to the pool after each free."""
    for filler_blocks, expected in ((15, 5 * M), (20, 0)):
        manager = _manager(monkeypatch, True, retention, num_blocks=300)
        ctx = _context()
        warm = _request("warm", ctx, split=True)
        assert _prefill(manager, warm, split=True) is not None
        manager.free(warm)
        assert _used(manager) == 0
        filler = _request("fill", [9000 + i for i in range(filler_blocks * M)], True)
        assert _prefill(manager, filler, split=True) is not None
        manager.free(filler)
        assert _used(manager) == 0
        again = _request("again", ctx + [1] * 64, split=True)
        assert manager.get_computed_blocks(again)[1] == expected


@pytest.fixture
def _no_vllm_config_context(monkeypatch):
    import vllm.config.vllm as vllm_config_module

    monkeypatch.setattr(
        vllm_config_module,
        "set_current_vllm_config",
        lambda *a, **k: contextlib.nullcontext(),
    )


def _platform_config(mode: str = "align") -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            mamba_block_size=None,
            mamba_cache_mode=mode,
            user_specified_mamba_block_size=False,
            mamba_page_size_padded=None,
        ),
        model_config=SimpleNamespace(use_mla=False),
    )


class _Backend:
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(8)]


def _split(config) -> None:
    Platform._split_hybrid_attn_block(
        config,
        _Backend,
        attn_page_size_1_token=1088,
        mamba_page_size=3_289_088,
    )


def test_platform_sizes(monkeypatch, _no_vllm_config_context):
    monkeypatch.setenv("VLLM_HYBRID_ATTN_BLOCK_SIZE", str(N))
    config = _platform_config()
    _split(config)
    cache = config.cache_config
    assert (cache.block_size, cache.mamba_block_size) == (N, M)
    assert cache.mamba_page_size_padded == POOL_BLOCK
    _split(config)  # a second pass (block_size now 232) gives the same result
    assert (cache.block_size, cache.mamba_block_size) == (N, M)
    for value in ("236", str(LEGACY)):
        monkeypatch.setenv("VLLM_HYBRID_ATTN_BLOCK_SIZE", value)
        with pytest.raises(ValueError, match="multiple of 8"):
            _split(_platform_config())
    monkeypatch.setenv("VLLM_HYBRID_ATTN_BLOCK_SIZE", str(N))
    with pytest.raises(ValueError, match="align"):
        _split(_platform_config("all"))


def _qwen_specs(attn_block: int) -> dict:
    gdn = MambaSpec(
        block_size=M,
        shapes=((48, 128, 128), (10240, 7)),
        dtypes=(torch.float32, torch.bfloat16),
        mamba_cache_mode="align",
        num_speculative_blocks=0,
        page_size_padded=POOL_BLOCK,
    )
    ple = MambaSpec(
        block_size=M,
        shapes=((2560, 13),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
        num_speculative_blocks=0,
        page_size_padded=POOL_BLOCK,
    )
    from vllm.v1.attention.backends.b12x import B12xPagedAttentionBackend

    # The QSA spec: B12x K/V planes plus the selector tail as page padding.
    attn = replace(
        B12xPagedAttentionBackend.customize_spec(
            FullAttentionSpec(
                block_size=attn_block,
                num_kv_heads=2,
                head_size=256,
                head_size_v=256,
                dtype=torch.uint8,
            )
        ),
        page_size_padded=attn_block * 1088,
    )
    specs = {f"model.layers.{i}.gdn": gdn for i in range(3)}
    specs["model.layers.3.ple"] = ple
    specs.update({f"model.layers.{4 + i}.attn": attn for i in range(ATTN_LAYERS)})
    return specs


def _grouping_config() -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        model_config=SimpleNamespace(max_model_len=262144),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=0,
        ),
    )


def test_groups_pack_attention_layers_into_the_pool_block(monkeypatch):
    monkeypatch.setenv("VLLM_HYBRID_ATTN_BLOCK_SIZE", str(N))
    config = _grouping_config()
    groups = get_kv_cache_groups(config, _qwen_specs(N))
    assert [len(g.layer_names) for g in groups] == [1, 1, 1, 1, ATTN_LAYERS]
    kv_config = get_kv_cache_config_from_groups(config, groups, 6 << 30)
    assert kv_config.num_blocks == (6 << 30) // POOL_BLOCK
    (attn_tensor,) = [t for t in kv_config.kv_cache_tensors if len(t.layers) > 1]
    assert attn_tensor.layer_stride == N * 1088
    assert attn_tensor.block_stride == POOL_BLOCK
    assert ATTN_LAYERS * attn_tensor.layer_stride <= attn_tensor.block_stride
    assert all(t.block_stride == POOL_BLOCK for t in kv_config.kv_cache_tensors)


def test_packed_qsa_views_stay_inside_their_layer_page(monkeypatch):
    """Each attention layer's K/V planes and selector tail live in its own
    page of the shared pool block; the block stride spans all 13 pages."""
    from vllm.models.qwen3_8_flash_next.common.qsa_cache import (
        qsa_compressed_cache_view,
    )
    from vllm.v1.worker.utils import allocate_kv_cache

    monkeypatch.setenv("VLLM_HYBRID_ATTN_BLOCK_SIZE", str(N))
    config = _grouping_config()
    groups = get_kv_cache_groups(config, _qwen_specs(N))
    kv_config = get_kv_cache_config_from_groups(config, groups, 3 * POOL_BLOCK)
    assert kv_config.num_blocks == 3
    views = allocate_kv_cache(kv_config, torch.device("cpu"), KVCacheLayout.BLHNC)
    base = views["model.layers.4.attn"].untyped_storage().data_ptr()
    page = N * 1088
    for i in range(ATTN_LAYERS):
        kv = views[f"model.layers.{4 + i}.attn"]
        assert tuple(kv.shape[:3]) == (3, 2, N)
        assert kv.stride(0) * kv.element_size() == POOL_BLOCK
        start = kv.data_ptr() - base
        tail = qsa_compressed_cache_view(kv, compress_ratio=4, index_head_dim=128)
        tail_start = tail.data_ptr() - base
        tail_end = tail_start + tail[0].numel() * tail.element_size()
        assert start == i * page
        assert start + 2 * N * 512 == tail_start
        assert tail_end == (i + 1) * page <= POOL_BLOCK
        assert tail.stride(0) * tail.element_size() == POOL_BLOCK


def test_groups_reject_attention_that_overflows_the_block(monkeypatch):
    monkeypatch.setenv("VLLM_HYBRID_ATTN_BLOCK_SIZE", "240")
    with pytest.raises(ValueError, match="at most 232"):
        get_kv_cache_groups(_grouping_config(), _qwen_specs(240))


def test_knob_hashed_only_when_set(monkeypatch):
    monkeypatch.setenv("VLLM_HYBRID_ATTN_BLOCK_SIZE", "0")
    assert "VLLM_HYBRID_ATTN_BLOCK_SIZE" not in envs.compile_factors()
    monkeypatch.setenv("VLLM_HYBRID_ATTN_BLOCK_SIZE", str(N))
    assert "VLLM_HYBRID_ATTN_BLOCK_SIZE" in envs.compile_factors()
