# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A BF16 MTP drafter KV cache over an fp8 target must not shrink the block pool.

Shape of the single-Spark Qwen3.8-Flash-Next v3d recipe: 36 GDN layers, one PLE
short-conv state, 12 QSA layers and the MTP drafter's QSA layer, 3024-token
blocks, 14 GiB pool. The PLE singleton bucket makes every layer its own cache
group, so a pool block is one page of the largest spec. k57 measured
4568 -> 2353 blocks with the drafter page at 3024 BF16 tokens.
"""

import functools
import math
from types import SimpleNamespace

import pytest
import torch

import vllm.models.qwen3_8_flash_next.nvidia.qsa as qsa
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils as kvu
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

pytestmark = pytest.mark.cpu_test

BLOCK = 3024
POOL = 15032385536  # kv_cache_memory_bytes of the v3d recipe
FP8_PAGE = 3_290_112  # 3024 x (2 x 2 x 256 fp8 K|V + 64 B selector tail)


@pytest.fixture(autouse=True)
def _selector_tail(monkeypatch):
    # b12x sizes the selector tail on GPU hosts: one BF16 128-wide key per 4 tokens.
    monkeypatch.setattr(
        qsa,
        "qsa_padded_page_size_bytes",
        lambda spec, compress_ratio, index_head_dim: (
            spec.unpadded_page_size_bytes
            + spec.block_size // compress_ratio * index_head_dim * 2
        ),
    )


def _layer(cache_dtype: str):
    layer = SimpleNamespace(
        num_kv_heads=2,
        head_dim=256,
        kv_cache_dtype=cache_dtype,
        raw_ring_capacity=8,
        attn_backend=qsa.Qwen3_8FlashNextQSABackend,
        _qsa_model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    layer._cache_spec = functools.partial(
        qsa.Qwen3_8FlashNextQSAAttention._cache_spec, layer
    )
    return layer


def _spec(drafter_dtype: str):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK, cache_dtype="fp8")
    )
    return qsa.Qwen3_8FlashNextQSAAttention.get_kv_cache_spec(
        _layer(drafter_dtype), config
    )


def test_fit_block_to_page():
    bf16_per_token = 2 * 2 * 256 * 2 + 64
    assert qsa.fit_block_to_page(BLOCK, 8, lambda b: b * 1088, FP8_PAGE) == BLOCK
    assert (
        qsa.fit_block_to_page(BLOCK, 8, lambda b: b * bf16_per_token, FP8_PAGE) == 1512
    )
    with pytest.raises(ValueError):
        qsa.fit_block_to_page(BLOCK, 8, lambda b: b * 10**9, FP8_PAGE)


def test_target_dtype_layer_spec_unchanged():
    spec = _spec("fp8")
    assert spec.block_size == BLOCK and spec.page_size_bytes == FP8_PAGE


def test_bf16_drafter_spec_fits_target_page():
    spec = _spec("auto")
    assert spec.dtype == torch.bfloat16
    assert spec.block_size == 1512
    assert spec.page_size_bytes == FP8_PAGE
    assert spec.real_page_size_bytes <= FP8_PAGE


def _model_specs(drafter):
    mamba = functools.partial(
        MambaSpec, block_size=BLOCK, page_size_padded=FP8_PAGE, mamba_cache_mode="align"
    )
    specs = {}
    for i in range(48):
        if (i + 1) % 4:
            specs[f"model.layers.{i}.linear_attn"] = mamba(
                shapes=((3, 10240), (48, 128, 128)),
                dtypes=(torch.bfloat16, torch.float32),
            )
        else:
            specs[f"model.layers.{i}.self_attn.attn"] = _spec("fp8")
    specs["model.layers.3.ple"] = mamba(
        shapes=((7, 10240),), dtypes=(torch.bfloat16,), mamba_type="short_conv"
    )
    specs["mtp.layers.48.self_attn.attn"] = drafter
    return specs


def _config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=SimpleNamespace(method="mtp"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, decode_context_parallel_size=1
        ),
        model_config=SimpleNamespace(get_num_layers=lambda parallel_config: 48),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, prefix_match_unit=None
        ),
        kv_transfer_config=None,
    )


def _pool(drafter):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=SimpleNamespace(method="mtp"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, decode_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, prefix_match_unit=None
        ),
        kv_transfer_config=None,
    )
    groups = kvu.get_kv_cache_groups(config, _model_specs(drafter))
    blocks = POOL // kvu._get_kv_cache_bytes_per_block(groups)
    sizes = kvu.resolve_kv_cache_block_sizes(
        KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=groups),
        config,
    )
    return len(groups), blocks, sizes


def test_pool_blocks_match_k57_boot_logs():
    unfitted = FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
        page_size_padded=BLOCK * 2 * 2 * 256 * 2 + BLOCK // 4 * 256,
    )
    assert _pool(_spec("fp8"))[:2] == (50, 4568)  # shipped v3d
    assert _pool(unfitted)[:2] == (50, 2353)  # k57 arm
    groups, blocks, sizes = _pool(_spec("auto"))  # fitted BF16 drafter
    assert (groups, blocks) == (50, 4568)
    assert sizes == (BLOCK, 1512)  # scheduler block unchanged, hash block halves


def test_prefix_hit_with_half_block_drafter_group(monkeypatch):
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    from ..v1.core.test_prefix_caching import make_request

    monkeypatch.setenv("VLLM_PREFIX_DROP_EXACT", "1")
    init_none_hash(sha256)
    drafter = _spec("auto")
    groups = [
        KVCacheGroupSpec(["attn"], _spec("fp8")),
        KVCacheGroupSpec(
            ["gdn"],
            MambaSpec(
                block_size=BLOCK,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        ),
        KVCacheGroupSpec(["mtp"], drafter, is_draft_group=True),
    ]
    manager = KVCacheManager(
        KVCacheConfig(num_blocks=400, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=262144,
        enable_caching=True,
        hash_block_size=1512,
        scheduler_block_size=BLOCK,
        use_eagle=True,
        num_prefill_lookahead=1,
    )
    # The finer hash exists only for the drafter: hits stay 3024-aligned and the
    # exact EAGLE drop stays on, as on the shipped equal-block layout (k62).
    assert not manager.coordinator.enable_partial_hash_hits
    assert manager.coordinator.prefix_drop_exact
    prompt = list(range(5 * BLOCK + 100))
    request = make_request("a", prompt, 1512, sha256)
    _, hit, _ = manager.get_computed_blocks(request)
    assert hit == 0
    for end in range(BLOCK, len(prompt) + BLOCK, BLOCK):  # align-mode chunks
        n = min(end, len(prompt)) - request.num_computed_tokens
        assert manager.allocate_slots(request, n, num_lookahead_tokens=4)
        request.num_computed_tokens += n
    attn, _, mtp = manager.get_block_ids(request.request_id)
    assert len(mtp) == math.ceil((len(prompt) + 4) / 1512)
    assert len(attn) == math.ceil((len(prompt) + 4) / BLOCK)
    manager.free(request)

    again = make_request("b", prompt, 1512, sha256)
    _, hit, _ = manager.get_computed_blocks(again)
    assert hit > 0 and hit % BLOCK == 0 and hit < len(prompt)


@pytest.mark.parametrize("annotate", [False, True])
def test_model_layout_keeps_hits_scheduler_aligned(monkeypatch, annotate):
    """The real layout as get_kv_cache_configs builds it: nothing sets
    is_eagle_group for this model (only DeepSeek-V4 does), so the draft group
    must be found by layer index. k65 booted with fine-grained hits on because
    01e0702d keyed on is_eagle_group."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    monkeypatch.setenv("VLLM_PREFIX_DROP_EXACT", "1")
    init_none_hash(sha256)
    config = _config()
    groups = kvu.get_kv_cache_groups(config, _model_specs(_spec("auto")))
    assert not any(g.is_eagle_group for g in groups)
    if annotate:
        kvu.annotate_draft_groups(config, groups)
        assert [g.layer_names for g in groups if g.is_draft_group] == [
            ["mtp.layers.48.self_attn.attn"]
        ]
    kv_config = KVCacheConfig(
        num_blocks=4568, kv_cache_tensors=[], kv_cache_groups=groups
    )
    scheduler_block, hash_block = kvu.resolve_kv_cache_block_sizes(kv_config, config)
    assert (scheduler_block, hash_block) == (BLOCK, 1512)
    manager = KVCacheManager(
        kv_config,
        max_model_len=262144,
        enable_caching=True,
        hash_block_size=hash_block,
        scheduler_block_size=scheduler_block,
        use_eagle=True,
        num_prefill_lookahead=1,
    )
    assert manager.coordinator.enable_partial_hash_hits is not annotate
    assert manager.coordinator.prefix_drop_exact is annotate


@pytest.mark.parametrize("partial, first_chunk", [(True, 1512), (False, 2060)])
def test_short_prompt_prefill_is_one_step(partial, first_chunk):
    """k62 lost 11% on pp2048 c1: fine-grained hits stop a 2060-token prefill at
    the 1512 hash boundary, so it takes two steps instead of one."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from ..v1.core.test_prefix_caching import make_request

    init_none_hash(sha256)
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=BLOCK, prefix_cache_retention_interval=None
        ),
        drop_last_prefix_cache_block=False,
        use_eagle=True,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=partial,
        hash_block_size=1512,
        mamba_has_prefill_checkpoint_blocks=False,
    )
    request = make_request("p", list(range(2060)), 1512, sha256)
    assert Scheduler._mamba_block_aligned_split(stub, request, 2060) == first_chunk


@pytest.mark.parametrize("annotate", [False, True])
def test_engine_block_size_ignores_the_draft_group(annotate):
    """EngineCore sets cache_config.block_size from the groups, and the
    scheduler aligns Mamba prefill chunks to it. With the draft group's 1512
    it stopped a 2550-token prefill at 1512 (k68 trace: 1512 + 1038)."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from ..v1.core.test_prefix_caching import make_request

    init_none_hash(sha256)
    config = _config()
    groups = kvu.get_kv_cache_groups(config, _model_specs(_spec("auto")))
    if annotate:
        kvu.annotate_draft_groups(config, groups)
    block = kvu.target_block_size(groups)
    assert block == (BLOCK if annotate else 1512)
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block, prefix_cache_retention_interval=None
        ),
        drop_last_prefix_cache_block=False,
        use_eagle=True,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        hash_block_size=1512,
        mamba_has_prefill_checkpoint_blocks=False,
    )
    request = make_request("q", list(range(2550)), 1512, sha256)
    first = Scheduler._mamba_block_aligned_split(stub, request, 2550)
    assert first == (2550 if annotate else 1512)


def test_drafter_k_scale_env_keeps_compile_key(monkeypatch):
    from vllm import envs

    monkeypatch.setenv("VLLM_QWEN38_MTP_K_SCALE", "0.25")
    assert envs.VLLM_QWEN38_MTP_K_SCALE == 0.25
    assert "VLLM_QWEN38_MTP_K_SCALE" not in envs.compile_factors()
