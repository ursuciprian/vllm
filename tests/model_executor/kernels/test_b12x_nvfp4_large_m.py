# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 NVFP4 dense layers with an MXFP8 copy for large row counts
(VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS / VLLM_B12X_NVFP4_MXFP8_CHECKPOINT)."""

from __future__ import annotations

import json
import types

import pytest
import torch

import vllm.model_executor.kernels.linear.nvfp4.b12x as nvfp4_mod
import vllm.utils.b12x as b12x_utils
from vllm.model_executor.kernels.linear import B12xNvFp4LinearKernel
from vllm.utils.b12x import B12xWorkload, b12x_linear_for, register_b12x_layer
from vllm.utils.torch_utils import _encode_layer_name

LUT = torch.tensor(nvfp4_mod._E2M1)
QKV = "model.language_model.layers.0.linear_attn.in_proj_qkv"
Z = "model.language_model.layers.0.linear_attn.in_proj_z"


def _holder(tag, workspace=0):
    calls = []
    return types.SimpleNamespace(
        tag=tag, calls=calls, layer_name=tag,
        run=lambda source, bias: calls.append(source.shape[0]) or source,
        get_workspace_size=lambda rows: workspace,
        unit=lambda workload, name: (tag, name, workload),
    )


def _dispatch_layer(min_rows=41):
    layer = torch.nn.Module()
    layer.b12x_linear = _holder("nvfp4", workspace=10)
    layer.b12x_large_m_linear = _holder("mxfp8", workspace=99)
    layer.b12x_large_m_min_rows = min_rows
    return layer


def test_holder_follows_row_cutoff():
    layer = _dispatch_layer()
    assert b12x_linear_for(layer, 1).tag == "nvfp4"
    assert b12x_linear_for(layer, 40).tag == "nvfp4"
    assert b12x_linear_for(layer, 41).tag == "mxfp8"
    assert b12x_linear_for(layer, 8192).tag == "mxfp8"
    plain = torch.nn.Module()
    plain.b12x_linear = _holder("only")
    assert b12x_linear_for(plain, 8192).tag == "only"
    assert b12x_linear_for(torch.nn.Module(), 4) is None


def test_op_body_runs_the_holder_for_its_rows():
    layer = _dispatch_layer()
    name = "large-m-op-body-probe"
    register_b12x_layer(name, layer)
    for rows in (8, 40, 48, 2048):
        b12x_utils._b12x_blockscaled_linear(
            torch.zeros(rows, 4), None, 4, _encode_layer_name(name))
    assert layer.b12x_linear.calls == [8, 40]
    assert layer.b12x_large_m_linear.calls == [48, 2048]


def test_projection_workspace_is_sized_for_the_serving_holder():
    # qwen_gdn_input_projections reserves qkvz scratch before the call; a
    # large-M call must get the MXFP8 holder's size, not the NVFP4 one.
    layer, other = _dispatch_layer(), torch.nn.Module()
    assert b12x_utils.get_b12x_projection_workspace_sizes(40, layer, other) == (10, 0)
    assert b12x_utils.get_b12x_projection_workspace_sizes(41, layer, other) == (99, 0)
    kernel = object.__new__(B12xNvFp4LinearKernel)
    assert kernel.get_workspace_size(layer, 40) == 10
    assert kernel.get_workspace_size(layer, 64) == 99


def test_preparation_declares_mxfp8_regimes_only_at_or_above_the_cutoff():
    layer = _dispatch_layer(min_rows=41)
    layer.b12x_nvfp4_packed_weight = types.SimpleNamespace(values=torch.empty(1))
    layer.weight, layer.weight_scale = torch.empty(1), torch.empty(1)
    layer.b12x_activation_mode = "a16"
    layer.b12x_bf16_input_supported = True
    counts = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 8192)
    workload = B12xWorkload(
        stage="weights", token_counts=counts, fixed_token_counts=counts[:-1],
        output_dtype=torch.bfloat16, max_tokens=8192, max_seqs=8, max_model_len=4096)
    kernel = object.__new__(B12xNvFp4LinearKernel)
    (nv_tag, nv_name, nv_wl), (mx_tag, mx_name, mx_wl) = (
        kernel.get_b12x_preparation_units(layer, workload))
    assert (nv_tag, mx_tag) == ("nvfp4", "mxfp8")
    assert nv_wl is workload  # the NVFP4 plan is unchanged from the plain arm
    assert mx_wl.fixed_token_counts == (48, 56, 64, 72, 80)
    assert mx_wl.max_tokens == 8192 and mx_name.startswith("linear.mxfp8.")


def test_checkpoint_prefixes_unfuse_packed_layers():
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig as Mixed,
    )

    config = types.SimpleNamespace(
        packed_modules_mapping={"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"]},
        _quantized_layer_prefix_candidates=Mixed._quantized_layer_prefix_candidates,
    )
    fused = Mixed._checkpoint_prefixes(
        config, "language_model.model.layers.0.linear_attn.in_proj_qkvz")
    assert (QKV, Z) in fused
    out = Mixed._checkpoint_prefixes(
        config, "language_model.model.layers.0.linear_attn.out_proj")
    assert ("model.language_model.layers.0.linear_attn.out_proj",) in out


def _nvfp4_layer(rows, k, seed=0):
    """A W4A16 layer in the unprocessed ModelOpt layout and its decoded weight.
    Values are products of E2M1 codes, power-of-two group scales and a 0.25
    global scale, so MXFP8 represents them exactly."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (rows, k), generator=g, dtype=torch.uint8)
    groups = torch.tensor([0.5, 1.0, 2.0])[torch.randint(0, 3, (rows, k // 16), generator=g)]
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(codes[:, ::2] | codes[:, 1::2] << 4, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(groups.to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
    layer.output_size_per_partition, layer.input_size_per_partition = rows, k
    return layer, LUT[codes.long()] * groups.repeat_interleave(16, 1) * 0.25


def _mxfp8(decoded):
    blocks = decoded.view(decoded.shape[0], -1, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(2**-60) / 448))
    values = (blocks / torch.exp2(exponent)[..., None]).view_as(decoded)
    return values.to(torch.float8_e4m3fn), (exponent + 127).to(torch.uint8)


def _checkpoint(tmp_path, tensors):
    from safetensors.torch import save_file

    save_file(tensors, str(tmp_path / "shard.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "shard.safetensors")}))
    return str(tmp_path)


@pytest.fixture
def fused_checkpoint(tmp_path, monkeypatch):
    nvfp4_mod._checkpoint_weight_map.cache_clear()
    layer, decoded = _nvfp4_layer(96, 64)
    qkv_w, qkv_s = _mxfp8(decoded[:64])
    z_w, z_s = _mxfp8(decoded[64:])
    path = _checkpoint(tmp_path, {
        f"{QKV}.weight": qkv_w, f"{QKV}.weight_scale": qkv_s,
        f"{Z}.weight": z_w, f"{Z}.weight_scale": z_s,
    })
    monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS", "41")
    monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_CHECKPOINT", path)
    yield layer, decoded
    nvfp4_mod._checkpoint_weight_map.cache_clear()


def test_copy_concatenates_shards_in_fused_order(fused_checkpoint):
    layer, decoded = fused_checkpoint
    weight, scale = nvfp4_mod.load_mxfp8_large_m_copy(layer, [("absent.name",), (QKV, Z)])
    assert weight.shape == (96, 64) and scale.shape == (96, 2)
    restored = weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, 1)
    torch.testing.assert_close(restored, decoded, rtol=0, atol=0)


def test_copy_rejects_wrong_shard_order_and_partition(fused_checkpoint):
    layer, _ = fused_checkpoint
    with pytest.raises(ValueError, match="differs from the NVFP4"):
        nvfp4_mod.load_mxfp8_large_m_copy(layer, [(Z, QKV)])
    layer.output_size_per_partition = 48  # a shard without the layer's TP rank
    with pytest.raises(ValueError, match="does not match the layer partition"):
        nvfp4_mod.load_mxfp8_large_m_copy(layer, [(QKV, Z)])


@pytest.mark.parametrize("shards,k", [
    (((QKV, 10240), (Z, 6144)), 2560),  # in_proj_qkvz at TP=1
    ((("model.language_model.layers.0.linear_attn.out_proj", 2560),), 6144),
], ids=["qkvz", "out"])
def test_check_at_gdn_shapes_under_the_loader_default_dtype(
        tmp_path, monkeypatch, shards, k):
    """The base loader runs process_weights_after_loading under
    set_default_torch_dtype(bfloat16), with the weights still in the ModelOpt
    layout. A sampled-row index built in BF16 rounds row 16383 up to 16384
    (2559 to 2560 for out_proj), past the end of the weight: a device-side
    assert at boot on GPU, an IndexError here."""
    from vllm.utils.torch_utils import set_default_torch_dtype

    nvfp4_mod._checkpoint_weight_map.cache_clear()
    layer, decoded = _nvfp4_layer(sum(rows for _, rows in shards), k, seed=63)
    tensors, start = {}, 0
    for name, rows in shards:
        tensors[f"{name}.weight"], tensors[f"{name}.weight_scale"] = _mxfp8(
            decoded[start:start + rows])
        start += rows
    path = _checkpoint(tmp_path, tensors)
    monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS", "41")
    monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_CHECKPOINT", path)
    names = tuple(name for name, _ in shards)
    try:
        with set_default_torch_dtype(torch.bfloat16):
            weight, scale = nvfp4_mod.load_mxfp8_large_m_copy(layer, [names])
            assert weight.shape == decoded.shape
            assert scale.shape == (decoded.shape[0], k // 32)
            if len(names) > 1:  # the check still fails loudly on a mismatched copy
                with pytest.raises(ValueError, match="differs from the NVFP4"):
                    nvfp4_mod.load_mxfp8_large_m_copy(layer, [names[::-1]])
    finally:
        nvfp4_mod._checkpoint_weight_map.cache_clear()


def test_copy_is_off_without_knob_or_names(fused_checkpoint, monkeypatch):
    layer, _ = fused_checkpoint
    with pytest.raises(ValueError, match="no MXFP8 weights"):
        nvfp4_mod.load_mxfp8_large_m_copy(layer, [("absent.name",)])
    monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS", "0")
    assert nvfp4_mod.load_mxfp8_large_m_copy(layer, [(QKV, Z)]) is None
    monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS", "41")
    monkeypatch.delenv("VLLM_B12X_NVFP4_MXFP8_CHECKPOINT")
    with pytest.raises(ValueError, match="needs VLLM_B12X_NVFP4_MXFP8_CHECKPOINT"):
        nvfp4_mod.load_mxfp8_large_m_copy(layer, [(QKV, Z)])


def test_attach_builds_an_unregistered_mxfp8_holder(monkeypatch):
    import vllm.model_executor.kernels.linear.mxfp8.b12x as mxfp8_mod

    monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS", "41")
    monkeypatch.setattr(mxfp8_mod.B12xMxfp8LinearKernel, "is_supported",
                        classmethod(lambda cls, compute_capability=None: (True, None)))
    packed = types.SimpleNamespace(out_features=96, weight=types.SimpleNamespace(values=None))
    monkeypatch.setattr(mxfp8_mod, "_import_b12x_blockscaled",
                        lambda: types.SimpleNamespace(pack_weight=lambda w, s: packed))
    layer = torch.nn.Module()
    name = "large-m-attach-probe"
    layer.b12x_layer_name = _encode_layer_name(name)
    weight = torch.zeros(96, 64, dtype=torch.float8_e4m3fn)
    nvfp4_mod.attach_mxfp8_large_m(layer, weight, torch.zeros(96, 2, dtype=torch.uint8))
    assert layer.b12x_large_m_linear.recipe == "mxfp8"
    assert layer.b12x_large_m_linear.packed is packed
    assert layer.b12x_large_m_min_rows == 41
    assert layer.b12x_mxfp8_copy.prefix == f"{name}.mxfp8"
    assert dict(layer.named_modules()) == {"": layer}
    assert b12x_utils.b12x_layer(f"{name}.mxfp8") is layer.b12x_mxfp8_copy


# TP>1: each rank's MXFP8 copy is its slice of the TP=1 copy, cut like vLLM cuts
# the NVFP4 weights. Toy GDN shapes: in_proj_qkvz (q, k, v, z) and out_proj.
OUT = "model.language_model.layers.0.linear_attn.out_proj"
TOY = {
    "qkvz": dict(sizes=[32, 32, 64, 64], k=64, names=(QKV, Z), split=128),
    "out": dict(sizes=[48], k=128, names=(OUT,), split=None),
}


@pytest.fixture
def fake_tp(monkeypatch):
    """Real vLLM parallel linears on CPU for (rank, world size) without a process group."""
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    state = {"rank": 0, "tp": 1}
    for mod in (linear, parameter):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: state["rank"], raising=False)
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: state["tp"], raising=False)

    def make(kind, rank, tp, in_features=None, out_features=None):
        state.update(rank=rank, tp=tp)
        toy = TOY[kind]
        k = toy["k"] if in_features is None else in_features
        if kind == "qkvz":
            return linear.MergedColumnParallelLinear(
                k, toy["sizes"], bias=False, params_dtype=torch.float32)
        return linear.RowParallelLinear(
            k, out_features or toy["sizes"][0], bias=False, params_dtype=torch.float32)
    return make


def _vllm_cut(make, kind, full, rank, tp=2):
    """``full`` (TP=1 layout, values exact in FP32) as vLLM's weight loader
    places it on ``rank``: the checkpoint tensors go through the layer's own
    weight_loader with the shard ids of the model's stacked mapping
    (in_proj_qkv -> (0, 1, 2), in_proj_z -> 3)."""
    layer = make(kind, rank, tp, in_features=full.shape[1], out_features=full.shape[0])
    param = layer.weight
    split = TOY[kind]["split"]
    if split is None:
        param.weight_loader(param, full.float())
    else:
        param.weight_loader(param, full[:split].float(), (0, 1, 2))
        param.weight_loader(param, full[split:].float(), 3)
    return param.data.clone()


def _rank_layer(make, kind, rank, codes, groups, tp=2):
    """A rank's W4A16 layer: a real parallel linear holding the rank's cut of
    the TP=1 NVFP4 codes and group scales (unprocessed ModelOpt layout)."""
    layer = make(kind, rank, tp)
    cut_codes = _vllm_cut(make, kind, codes, rank, tp).to(torch.uint8)
    cut_groups = _vllm_cut(make, kind, groups, rank, tp)
    del layer.weight
    layer.weight = torch.nn.Parameter(cut_codes, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(cut_groups.to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
    nibbles = torch.stack((cut_codes & 15, cut_codes >> 4), -1).flatten(1).long()
    decoded = LUT[nibbles] * cut_groups.repeat_interleave(16, 1) * 0.25
    return layer, decoded


def _toy_checkpoint(tmp_path, monkeypatch, kind, seed=5):
    nvfp4_mod._checkpoint_weight_map.cache_clear()
    toy = TOY[kind]
    g = torch.Generator().manual_seed(seed)
    rows, k = sum(toy["sizes"]), toy["k"]
    nib = torch.randint(0, 16, (rows, k), generator=g, dtype=torch.uint8)
    codes = nib[:, ::2] | nib[:, 1::2] << 4
    groups = torch.tensor([0.5, 1.0, 2.0])[torch.randint(0, 3, (rows, k // 16), generator=g)]
    decoded = LUT[nib.long()] * groups.repeat_interleave(16, 1) * 0.25
    tensors, start = {}, 0
    split = toy["split"] or rows
    for name, (a, b) in zip(toy["names"], ((0, split), (split, rows))):
        tensors[f"{name}.weight"], tensors[f"{name}.weight_scale"] = _mxfp8(decoded[a:b])
    monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS", "41")
    monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_CHECKPOINT", _checkpoint(tmp_path, tensors))
    return codes, groups, decoded


def _dequant(weight, scale):
    return weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, 1)


@pytest.mark.parametrize("kind", ["qkvz", "out"])
def test_tp2_rank_copy_is_the_tp1_copy_cut_like_the_nvfp4_weights(
        tmp_path, monkeypatch, fake_tp, kind):
    codes, groups, decoded = _toy_checkpoint(tmp_path, monkeypatch, kind)
    names = [TOY[kind]["names"]]
    try:
        tp1, _ = _rank_layer(fake_tp, kind, 0, codes, groups, tp=1)
        w1, s1 = nvfp4_mod.load_mxfp8_large_m_copy(tp1, names)
        torch.testing.assert_close(_dequant(w1, s1), decoded, rtol=0, atol=0)
        rows = []
        for rank in (0, 1):
            layer, rank_decoded = _rank_layer(fake_tp, kind, rank, codes, groups)
            w, s = nvfp4_mod.load_mxfp8_large_m_copy(layer, names)
            n, k = layer.output_size_per_partition, layer.input_size_per_partition
            assert w.shape == (n, k) and s.shape == (n, k // 32), (w.shape, s.shape)
            assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.uint8
            # the same slice vLLM's loader takes of the TP=1 copy (weights and scales)
            assert torch.equal(w.float(), _vllm_cut(fake_tp, kind, w1, rank))
            assert torch.equal(s.float(), _vllm_cut(fake_tp, kind, s1, rank))
            # and it decodes to exactly the rank's NVFP4 weights
            torch.testing.assert_close(_dequant(w, s), rank_decoded, rtol=0, atol=0)
            rows.append(w.shape[0])
        if kind == "qkvz":
            assert rows == [96, 96]  # q16 k16 v32 z32 per rank, not the first/second half
        # a rank copy with another rank's cut fails the NVFP4 check loudly
        layer, _ = _rank_layer(fake_tp, kind, 1, codes, groups)
        layer.tp_rank = 0
        with pytest.raises(ValueError, match="differs from the NVFP4"):
            nvfp4_mod.load_mxfp8_large_m_copy(layer, names)
    finally:
        nvfp4_mod._checkpoint_weight_map.cache_clear()


@pytest.mark.parametrize("kind", ["qkvz", "out"])
def test_tp2_dispatch_output_matches_tp1(tmp_path, monkeypatch, fake_tp, kind):
    """Fake TP=2 through the real op body: rows below the cutoff run each rank's
    NVFP4 holder, rows at or above it its MXFP8 holder; gathering the column
    outputs / summing the row partials gives the TP=1 layer's output."""
    codes, groups, decoded = _toy_checkpoint(tmp_path, monkeypatch, kind, seed=11)
    names = [TOY[kind]["names"]]

    def holder(tag, w):
        calls = []
        return types.SimpleNamespace(
            tag=tag, calls=calls, layer_name=tag,
            run=lambda x, bias: calls.append(x.shape[0]) or x @ w.T,
            get_workspace_size=lambda rows: 0)

    try:
        layers = []
        for rank in (0, 1):
            layer, rank_decoded = _rank_layer(fake_tp, kind, rank, codes, groups)
            w, s = nvfp4_mod.load_mxfp8_large_m_copy(layer, names)
            layer.b12x_linear = holder("nvfp4", rank_decoded)
            layer.b12x_large_m_linear = holder("mxfp8", _dequant(w, s))
            layer.b12x_large_m_min_rows = 41
            name = f"large-m-tp2-{kind}-{rank}"
            register_b12x_layer(name, layer)
            layers.append((layer, _encode_layer_name(name)))
        g = torch.Generator().manual_seed(3)
        for rows in (8, 40, 41, 64):
            x = torch.randn(rows, TOY[kind]["k"], generator=g, dtype=torch.float64).float()
            ref = x @ decoded.T
            outs = []
            for rank, (layer, name) in enumerate(layers):
                xr = x if kind == "qkvz" else _vllm_cut(fake_tp, kind, x, rank)
                outs.append(b12x_utils._b12x_blockscaled_linear(
                    xr, None, layer.output_size_per_partition, name))
                assert (layer.b12x_large_m_linear if rows >= 41 else layer.b12x_linear).calls[-1] == rows
            if kind == "qkvz":
                for rank, out in enumerate(outs):
                    torch.testing.assert_close(out, _vllm_cut(fake_tp, kind, ref.T, rank).T)
            else:
                torch.testing.assert_close(outs[0] + outs[1], ref)
        for layer, _ in layers:
            assert layer.b12x_linear.calls == [8, 40]
            assert layer.b12x_large_m_linear.calls == [41, 64]
    finally:
        nvfp4_mod._checkpoint_weight_map.cache_clear()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="SM120/SM121 required")
def test_each_path_matches_its_standalone_layer(tmp_path, monkeypatch):
    """GPU: rows below the cutoff are bit-exact with a plain W4A16 layer, rows at
    or above it with a plain MXFP8 layer of the same weights, eager and in a graph."""
    import vllm.model_executor.parameter as parameter
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMxFp8Config,
        ModelOptMxFp8LinearMethod,
        ModelOptNvFp4Config,
        ModelOptNvFp4W4A16LinearMethod,
    )
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        reset_workspace_manager,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    from test_b12x_linear import _prepare  # same directory, no package

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    nvfp4_mod._checkpoint_weight_map.cache_clear()
    device = torch.device("cuda", torch.accelerator.current_device_index())
    n, k, cutoff = 2560, 6144, 41  # the GDN out_proj shape at TP=1
    host, decoded = _nvfp4_layer(n, k, seed=3817)
    mx_w, mx_s = _mxfp8(decoded)
    name = "model.language_model.layers.0.linear_attn.out_proj"
    path = _checkpoint(tmp_path, {f"{name}.weight": mx_w, f"{name}.weight_scale": mx_s})
    counts, fixed = (1, 8, 40, 41, 48, 64, 300), (1, 8, 40, 48, 64)
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="b12x"))

    def w4a16(prefixes):
        method = ModelOptNvFp4W4A16LinearMethod(
            ModelOptNvFp4Config(quant_method="W4A16_NVFP4",
                                is_checkpoint_nvfp4_serialized=True), prefixes)
        layer = torch.nn.Module()
        with torch.device(device):
            method.create_weights(layer, k, [n], k, n, torch.bfloat16)
        layer.weight.copy_(host.weight)
        layer.weight_scale.copy_(host.weight_scale)
        layer.weight_scale_2.fill_(0.25)
        with set_default_torch_dtype(torch.bfloat16):  # as the base loader runs it
            method.process_weights_after_loading(layer)
        return method, layer

    reset_workspace_manager()
    init_workspace_manager(device)
    current_workspace_manager().reserve_all(((1,), torch.uint8))
    sessions = []
    try:
        with set_current_vllm_config(config), torch.no_grad():
            monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS", str(cutoff))
            monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_CHECKPOINT", path)
            both, layer = w4a16(((name,),))
            assert layer.b12x_large_m_min_rows == cutoff
            monkeypatch.setenv("VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS", "0")
            plain_nv, nv_layer = w4a16(((name,),))
            assert getattr(nv_layer, "b12x_large_m_linear", None) is None
            plain_mx = ModelOptMxFp8LinearMethod(
                ModelOptMxFp8Config(is_checkpoint_mxfp8_serialized=True,
                                    kv_cache_quant_algo=None, exclude_modules=[]))
            mx_layer = torch.nn.Module()
            with torch.device(device):
                plain_mx.create_weights(mx_layer, k, [n], k, n, torch.bfloat16)
            mx_layer.weight.copy_(mx_w)
            mx_layer.weight_scale.copy_(mx_s)
            with set_default_torch_dtype(torch.bfloat16):
                plain_mx.process_weights_after_loading(mx_layer)
            for target, exact in ((layer, fixed), (nv_layer, fixed),
                                  (mx_layer, tuple(m for m in fixed if m >= cutoff))):
                sessions.append(_prepare(target, device=device, counts=counts,
                                         fixed=exact, max_tokens=512)[0])
            for rows in counts:
                source = torch.randn(rows, k, dtype=torch.bfloat16, device=device) * 0.125
                ref_method, ref_layer = ((plain_mx, mx_layer) if rows >= cutoff
                                         else (plain_nv, nv_layer))
                expected = ref_method.apply(ref_layer, source)
                assert torch.equal(both.apply(layer, source), expected), rows
                wanted = layer.b12x_large_m_linear if rows >= cutoff else layer.b12x_linear
                assert b12x_linear_for(layer, rows) is wanted
                if rows >= cutoff:  # the paths differ, so equality above proves the switch
                    assert not torch.equal(plain_nv.apply(nv_layer, source), expected), rows
                exact = (source.float() @ decoded.to(device).T)
                assert ((expected.float() - exact).norm() / exact.norm()) < 0.05
                if rows in fixed:
                    graph = torch.cuda.CUDAGraph()
                    with sessions[0].capture():
                        with torch.cuda.graph(graph):
                            output = both.apply(layer, source)
                    source.mul_(-0.5)
                    graph.replay()
                    torch.accelerator.synchronize(device)
                    replayed = output.clone()
                    graph.reset()
                    assert torch.equal(replayed, ref_method.apply(ref_layer, source)), rows
    finally:
        for session in sessions:
            session.close()
        reset_workspace_manager()
        nvfp4_mod._checkpoint_weight_map.cache_clear()


@pytest.mark.parametrize(
    "name, value",
    (("VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS", "41"),
     ("VLLM_B12X_NVFP4_MXFP8_CHECKPOINT", "/snapshots/mxfp8")),
)
def test_dispatch_knobs_are_compile_factors(monkeypatch, name, value):
    # Both change the declared b12x plans, so a warm AOT cache must not load across them.
    import vllm.envs as envs

    envs.disable_envs_cache()
    monkeypatch.delenv(name, raising=False)
    off = envs.compile_factors()
    monkeypatch.setenv(name, value)
    assert envs.compile_factors() != off
