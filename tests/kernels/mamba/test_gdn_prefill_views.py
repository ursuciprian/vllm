# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_GDN_PREFILL_VIEWS: the b12x mixed GDN path without token gathers.

CPU tests check the host-side condition (identity token order) against the
device worklist the general path gathers with. The CUDA test runs the general
path and the views path from the same states and asserts bit-identical output,
recurrent states, checkpoint blocks and convolution states.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.b12x_gdn_metadata import B12xGdnMixedMetadata


def _common(lengths, computed, device, width=3):
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    seq_lens = torch.tensor(lengths, dtype=torch.int32) + torch.tensor(
        computed, dtype=torch.int32
    )
    n = len(lengths)
    return SimpleNamespace(
        query_start_loc_cpu=starts,
        query_start_loc=starts.to(device),
        num_reqs=n,
        seq_lens=seq_lens.to(device),
        seq_lens_cpu_upper_bound=seq_lens,
        block_table_tensor=(
            torch.arange(width * n, dtype=torch.int32, device=device).reshape(n, width)
            + 10
        ),
    )


def _stage(work, lengths, computed, drafts, device="cpu"):
    n = len(lengths)
    work.stage(
        _common(lengths, computed, device, width=8),
        torch.arange(1, 1 + 3 * n, dtype=torch.int32, device=device).reshape(n, 3),
        torch.ones(n, dtype=torch.int32, device=device),
        torch.tensor(drafts),
        checkpoint_block_size=16,
    )


def test_identity_tokens_match_the_device_worklist():
    """Whenever the host claims identity, the gather indices are arange(live)."""
    rng = random.Random(101)
    work = B12xGdnMixedMetadata(
        max_tokens=256, max_seqs=4, state_columns=3, device="cpu"
    )
    claimed = refused = 0
    for _ in range(400):
        n = rng.randint(1, 4)
        drafts = [rng.choice((-1, -1, 1, 2)) for _ in range(n)]
        # a verification row holds at most state_columns tokens
        lengths = [
            rng.choice((0, 1, 2, 3))
            if d >= 0
            else rng.choice((0, 1, rng.randint(1, 60)))
            for d in drafts
        ]
        computed = [rng.choice((0, 16, 21)) for _ in range(n)]
        _stage(work, lengths, computed, drafts)
        live = work.identity_tokens
        spec = any(d >= 0 and length > 1 for d, length in zip(drafts, lengths))
        if spec:
            assert live is None
            refused += 1
            continue
        claimed += 1
        assert live == sum(lengths) == int(work.live_counts[1])
        assert torch.equal(work.token_indices[:live], torch.arange(live))
    assert claimed > 50 and refused > 50


def test_identity_tokens_follow_worklist_copies():
    source = B12xGdnMixedMetadata(
        max_tokens=64, max_seqs=3, state_columns=3, device="cpu"
    )
    other = B12xGdnMixedMetadata(
        max_tokens=64, max_seqs=3, state_columns=3, device="cpu"
    )
    _stage(source, [17, 1, 2], [0, 32, 16], [-1, -1, -1])
    other.copy_worklists_from(source)
    assert other.identity_tokens == source.identity_tokens == 20
    _stage(source, [17, 1, 3], [0, 32, 16], [-1, -1, 2])
    other.copy_worklists_from(source)
    assert other.identity_tokens is None


def _predicate_case(monkeypatch, **overrides):
    pytest.importorskip("triton")
    from vllm import envs
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    case = dict(
        on=True,
        num_spec_decodes=0,
        identity=40,
        rows=64,
        conv_dtype=torch.bfloat16,
        out=torch.zeros(64, 2, 4, dtype=torch.bfloat16),
        capturing=False,
    )
    case.update(overrides)
    monkeypatch.setattr(envs, "VLLM_GDN_PREFILL_VIEWS", case["on"], raising=False)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: case["capturing"]
    )
    attn = SimpleNamespace(
        num_spec_decodes=case["num_spec_decodes"],
        b12x_mixed=SimpleNamespace(identity_tokens=case["identity"]),
    )
    return QwenGatedDeltaNetAttention._b12x_prefill_views_rows(
        attn,
        torch.zeros(case["rows"], 8, dtype=torch.bfloat16),
        case["out"],
        torch.zeros(2, 8, 3, dtype=case["conv_dtype"]),
    )


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, 40),
        ({"identity": 64}, 64),
        ({"on": False}, None),
        ({"num_spec_decodes": 1}, None),
        ({"identity": None}, None),
        ({"identity": 0}, None),
        ({"identity": 65}, None),
        ({"conv_dtype": torch.float32}, None),
        ({"out": torch.zeros(64, 4, 2, dtype=torch.bfloat16).transpose(1, 2)}, None),
        ({"capturing": True}, None),
    ],
)
def test_views_condition_is_host_only(monkeypatch, overrides, expected):
    assert _predicate_case(monkeypatch, **overrides) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_views_path_is_bit_identical_to_the_general_path(
    default_vllm_config, monkeypatch
):
    from b12x.preparation import PreparationSession

    from vllm import envs
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
        RMSNormGated,
        is_conv_state_dim_first,
    )
    from vllm.model_executor.layers.mamba.ops.b12x_gdn_prefill import (
        B12xGdnPrefill,
        prefill_capacities,
    )
    from vllm.utils.b12x import B12xWorkload
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.worker.workspace import init_workspace_manager

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM12x is required")
    torch.manual_seed(101)
    device = torch.device("cuda")
    rows, seqs, key_heads, value_heads = 128, 3, 2, 6
    qkv = (2 * key_heads + value_heads) * 128
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.gqa_interleaved_layout = False
    layer.gdn_decode_kernel = "b12x"
    layer.gdn_prefill_backend = "b12x"
    layer.num_spec = 2
    layer.num_k_heads, layer.num_v_heads, layer.tp_size = key_heads, value_heads, 1
    layer.head_k_dim = layer.head_v_dim = 128
    layer.layer_norm_epsilon = 1e-6
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    layer.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
    layer.norm = RMSNormGated(
        128, eps=1e-6, norm_before_gate=True, activation="silu", device=device
    )
    layer.norm.weight.data.copy_(torch.rand(128, device=device) + 0.5)
    layer.A_log = torch.nn.Parameter(torch.full((value_heads,), -1.0, device=device))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(value_heads, device=device))
    layer.activation = "silu"
    layer.conv1d = torch.nn.Conv1d(
        qkv, qkv, 4, groups=qkv, device=device, dtype=torch.bfloat16
    )
    conv = (torch.randn(20, qkv, 5, device=device) * 0.1).bfloat16()
    pool = torch.randn(20, value_heads, 128, 128, device=device) * 0.1
    layer.kv_cache = (
        conv if is_conv_state_dim_first() else conv.transpose(-1, -2),
        pool,
    )
    layer._b12x_preparation_prefix = "test.gdn-prefill-views"
    layer._b12x_prefill_max_tokens = rows
    layer._b12x_prefill_max_seqs = seqs
    layer._b12x_prefill_staging = None
    layer._b12x_decode_plan = None
    layer._b12x_prefill_plans = {}
    layer._initialize_b12x_gdn_decode(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=seqs))
    )
    layer._initialize_b12x_gdn_prefill()
    init_workspace_manager(device)
    layer._b12x_decode_plan = layer._make_b12x_gdn_plan(pool.shape[0])
    layer._b12x_prefill_plans = {
        capacity: layer._b12x_gdn_prefill_declaration(capacity)
        for capacity in prefill_capacities(rows)
    }
    layer._b12x_prefill = B12xGdnPrefill(
        recurrent_state=pool,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        max_tokens=rows,
        max_seqs=seqs,
        key_heads=layer._b12x_local_key_heads,
        value_heads=layer._b12x_local_value_heads,
        checkpoint_export=True,
        plans=layer._b12x_prefill_plans,
        staging=layer._ensure_b12x_gdn_prefill_staging(),
    )
    workload = B12xWorkload(
        stage="state",
        token_counts=(1, 2, 4, 16, 32, 64, rows),
        fixed_token_counts=(),
        output_dtype=torch.bfloat16,
        max_tokens=rows,
        max_seqs=seqs,
        max_model_len=rows,
        speculative_tokens=2,
    )
    units = layer.get_b12x_preparation_units(layer, workload)
    session = PreparationSession(device=device, autotune=False)
    session.prepare(tuple(r for unit in units for r in unit.requests), autotune=False)
    metadata = B12xGdnMixedMetadata(
        max_tokens=rows, max_seqs=seqs, state_columns=3, device=device
    )
    # The projections are views of one wide row, as served: q|k|v|z and b|a.
    qkvz = (torch.randn(rows, qkv + value_heads * 128, device=device) * 0.25).bfloat16()
    ba = (torch.randn(rows, 2 * value_heads, device=device) * 0.25).bfloat16()
    mixed_qkv, gate = qkvz.split([qkv, value_heads * 128], dim=-1)
    b, a = ba.chunk(2, dim=-1)
    out = torch.empty(rows, value_heads, 128, dtype=torch.bfloat16, device=device)
    saved_conv, saved_pool = conv.clone(), pool.clone()
    views = []
    general_views = layer._forward_core_b12x_prefill_views
    layer._forward_core_b12x_prefill_views = lambda **kw: (
        views.append(kw["live"]),
        general_views(**kw),
    )

    def run(on, spec_decodes):
        monkeypatch.setattr(envs, "VLLM_GDN_PREFILL_VIEWS", on, raising=False)
        conv.copy_(saved_conv)
        pool.copy_(saved_pool)
        out.fill_(float("nan"))  # stale rows must not survive either path
        layer._forward_core_b12x_mixed(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            output_gate=gate.reshape(rows, value_heads, 128),
            core_attn_out=out,
            attn_metadata=GDNAttentionMetadata(
                0, 0, 0, 0, spec_decodes, 0, rows, b12x_mixed=metadata
            ),
        )
        torch.accelerator.synchronize()
        return out.clone(), conv.clone(), pool.clone()

    trials = [
        # lengths, computed (prefix-cache hit / earlier chunk), drafts, takes views
        ((40, 0, 0), (0, 0, 0), (-1, -1, -1), True),  # one fresh prompt, padded tail
        (
            (48, 33, 47),
            (0, 16, 5),
            (-1, -1, -1),
            True,
        ),  # resumed chunks, checkpoint inside
        ((64, 1, 63), (32, 9, 0), (-1, -1, -1), True),  # full rows, a one-token row
        ((1, 1, 1), (9, 9, 9), (-1, -1, -1), True),  # one token per request
        ((1, 33, 3), (9, 0, 9), (-1, -1, 2), False),  # a verification row: general path
    ]
    with torch.inference_mode():
        try:
            for lengths, computed, drafts, takes in trials:
                common = _common(lengths, computed, device)
                metadata.stage(
                    common,
                    torch.arange(1, 10, dtype=torch.int32, device=device).reshape(3, 3),
                    torch.ones(3, dtype=torch.int32, device=device),
                    torch.tensor(drafts),
                    checkpoint_block_size=16,
                )
                spec = int(not takes)
                views.clear()
                general = run(False, spec)
                again = run(False, spec)
                assert not views
                fast = run(True, spec)
                assert views == ([sum(lengths)] if takes else []), (lengths, views)
                for name, x, y, z in zip(
                    ("output", "conv", "pool"), general, again, fast
                ):
                    if (
                        name != "output"
                    ):  # slot 0 is the null block padded spec rows write
                        x, y, z = x[1:], y[1:], z[1:]
                    assert torch.equal(x, y), f"general path not deterministic: {name}"
                    assert torch.equal(x, z), (
                        f"views differ from general: {name} {lengths}"
                    )
                assert not general[0].isnan().any()
            # A captured graph replays for any batch of its size, so capture
            # keeps the gathers.
            metadata.stage(
                _common((48, 33, 47), (0, 16, 5), device),
                torch.arange(1, 10, dtype=torch.int32, device=device).reshape(3, 3),
                torch.ones(3, dtype=torch.int32, device=device),
                torch.tensor((-1, -1, -1)),
                checkpoint_block_size=16,
            )
            monkeypatch.setattr(envs, "VLLM_GDN_PREFILL_VIEWS", True, raising=False)
            views.clear()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                layer._forward_core_b12x_mixed(
                    mixed_qkv=mixed_qkv,
                    b=b,
                    a=a,
                    output_gate=gate.reshape(rows, value_heads, 128),
                    core_attn_out=out,
                    attn_metadata=GDNAttentionMetadata(
                        0, 0, 0, 0, 0, 0, rows, b12x_mixed=metadata
                    ),
                )
            assert not views
            del graph
        finally:
            torch.accelerator.synchronize()
            session.close()
