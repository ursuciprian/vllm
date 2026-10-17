# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""k114: FlashInfer SM12x CuTe DSL W4A4 MoE for the NVFP4 routed experts.

FlashInfer's ``SM12xNvfp4Config`` runner (flashinfer-ai/flashinfer#5863),
imported from ``flashinfer_k114``: FlashInfer 38e6b894 installed beside the
image's FlashInfer under another top-level name, so the sampler and every other
FlashInfer user keep the image's build and FlashInfer's ``b12x`` compatibility
package never shadows the real b12x. Selected by VLLM_FI_SM12X_W4A4_MOE=1 with
``--moe-backend b12x`` for trunk layers only (oracle/nvfp4.py).

Weights: the checkpoint's [gate; up] w13 is handed to ``prepare_weights`` as
[up; gate] (it flips it back), and the prepared block scales are written back
into the checkpoint storage, so no second copy of the experts stays alive.
Activation global scales are per layer: 1 / max over experts of the checkpoint
input scales (the k109 contract). Token-count buckets share one compile cache
and, unlike FlashInfer's default, one workspace per bucket across all layers;
every bucket up to max_num_batched_tokens is compiled and allocated on the
first layer, before the KV pool is sized and before CUDA graph capture, and the
first layer is checked against an FP32 dequantized reference.
"""

import time

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

FI_PACKAGE = "flashinfer_k114"
# The fail-fast numeric check: k109 measured 0.10-0.16 rel L2 for this backend
# against an FP32 reference; a w13 order or scale bug lands near or above 1.
CHECK_MAX_REL_L2 = 0.35

# ponytail: one workspace per token bucket shared by every layer. Safe while
# routed-expert layers run one at a time on one stream (no DBO); per-stream
# workspaces if that changes.
_SHARED_WORKSPACES: dict[int, list[torch.Tensor]] = {}
_STATE = {"layers": 0, "warmed": False, "announced": False}


def _fi():
    try:
        import flashinfer_k114.fused_moe as fm
    except ImportError as e:
        raise ImportError(
            "VLLM_FI_SM12X_W4A4_MOE=1 needs the flashinfer_k114 package "
            "(FlashInfer 38e6b894, installed by the k114 image overlay)"
        ) from e
    return fm


def _dequant(packed: torch.Tensor, block: torch.Tensor, gs: torch.Tensor):
    """[R, K/2] E2M1 (low nibble first) x [R, K/16] E4M3 x scalar -> FP32."""
    lut = torch.tensor(
        [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
        dtype=torch.float32,
    )
    b = packed.view(torch.uint8)
    v = lut[torch.stack(((b & 0xF).long(), (b >> 4).long()), -1).reshape(b.shape[0], -1)]
    return v * block.float().repeat_interleave(16, dim=-1) * gs.float()


class FlashInferSm12xW4A4Experts(mk.FusedMoEExpertsModular):
    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.max_num_tokens = moe_config.max_num_tokens
        self.topk = moe_config.experts_per_token
        self._runner = None
        self._weights = None
        self._fi_layer = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        fm = _fi()
        w13, w2 = layer.w13_weight, layer.w2_weight
        s13, s2 = layer.w13_weight_scale, layer.w2_weight_scale
        g1 = layer.w13_weight_scale_2.float().reshape(-1).contiguous()
        g2 = layer.w2_weight_scale_2.float().reshape(-1).contiguous()
        num_experts, two_i, half_h = w13.shape
        inter, hidden, dev = two_i // 2, half_h * 2, w13.device
        a1_scale, a2_scale = layer.w13_input_scale.float(), layer.w2_input_scale.float()

        check = None
        if not _STATE["warmed"]:  # reference from the logical tensors, before the swizzle
            check = self._reference(w13, s13, g1, w2, s2, g2, a1_scale, hidden, num_experts)

        gate, up = w13.chunk(2, dim=1)
        sg, su = s13.chunk(2, dim=1)
        act = fm.SwiGLU()
        view = fm.SM12xNvfp4Config.prepare_weights(
            torch.cat((up, gate), dim=1), torch.cat((su, sg), dim=1), g1,
            w2, s2, g2, activation=act, num_local_experts=num_experts,
            hidden_size=hidden, intermediate_size=inter, device=dev,
        )
        view = dict(view.items())
        # Back into the checkpoint storage (b12x's weight pool when loaded with
        # --load-format b12x): same byte counts at this geometry, no copy kept.
        for key, param in (("w1", w13), ("w1_scale", s13), ("w2", w2), ("w2_scale", s2)):
            t = view[key]
            if t.data_ptr() == param.data_ptr():
                continue
            dst = param.data
            if not dst.is_contiguous() or t.numel() * t.element_size() != dst.numel() * dst.element_size():
                raise ValueError(f"k114: prepared {key} {tuple(t.shape)} does not fit {tuple(dst.shape)}")
            flat = dst.view(torch.uint8).view(-1)
            flat.copy_(t.contiguous().view(torch.uint8).view(-1))
            view[key] = flat.view(t.dtype).view(t.shape)
            del t
        view["w1_input_global_scale"] = (1.0 / a1_scale.max()).reshape(())
        view["w2_input_global_scale"] = (1.0 / a2_scale.max()).reshape(())

        config = fm.MoEConfig(
            routing=fm.RoutingConfig(num_experts=num_experts, top_k=self.topk),
            quant=fm.QuantConfig(weight=fm.QuantFormat.NVFP4, activation=fm.QuantFormat.NVFP4),
            experts=fm.ExpertConfig(intermediate_size=inter),
            activation=act,
            backend=fm.BackendOptions((fm.SM12xNvfp4Config(),)),
            finalize=fm.MoEFinalizeConfig(do_finalize=True, use_fused_finalize=True),
            execution=fm.ExecutionConfig(enable_pdl=False, tune_max_num_tokens=self.max_num_tokens),
        )
        self._fi_layer = fm.MoELayer(config, device=dev)
        runner = self._fi_layer.runners[0]
        if runner.backend_key != "sm12x_nvfp4":
            raise RuntimeError(f"k114: FlashInfer picked {runner.backend_key}, not sm12x_nvfp4")
        runner._workspaces = _SHARED_WORKSPACES
        self._weights = fm.MoEWeightPack()
        self._weights.prepare_for(runner.backend_key, view)
        self._runner = runner
        _STATE["layers"] += 1
        if check is not None:
            self._warm_and_check(fm, hidden, num_experts, check)

    def _reference(self, w13, s13, g1, w2, s2, g2, a1_scale, hidden, num_experts):
        gen = torch.Generator(device=w13.device).manual_seed(114)
        rows, inter = 16, w13.shape[1] // 2
        sigma = float(a1_scale.median()) * 6 * 448 / 5
        x = (torch.randn(rows, hidden, device=w13.device, generator=gen, dtype=torch.float32) * sigma).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(num_experts, device=w13.device, generator=gen)[: self.topk]
                           for _ in range(rows)]).to(torch.int32)
        wts = torch.softmax(torch.randn(rows, self.topk, device=w13.device, generator=gen, dtype=torch.float32), -1)
        # explicit fp32: vLLM sets the default dtype to the model dtype (bf16) during load
        ref = torch.zeros(rows, hidden, device=w13.device, dtype=torch.float32)
        xf = x.float()
        for e in torch.unique(ids).tolist():
            r, k = (ids == e).nonzero(as_tuple=True)
            w = _dequant(w13[e], s13[e], g1[e])
            h = torch.nn.functional.silu(xf[r] @ w[:inter].T) * (xf[r] @ w[inter:].T)
            y = h.to(torch.bfloat16).float() @ _dequant(w2[e], s2[e], g2[e]).T
            ref.index_add_(0, r, y * wts[r, k, None])
        return x, ids, wts.float().contiguous(), ref

    def _run(self, fm, x, ids, wts):
        acts = fm.MoEActivationPack(hidden_states_q=x, hidden_states_scale=None,
                                    topk_ids=ids, topk_weights=wts)
        return self._runner.forward(self._runner.pack_inputs(acts, self._weights), tactic=-1)

    def _warm_and_check(self, fm, hidden, num_experts, check):
        from flashinfer_k114.fused_moe.utils import get_hybrid_num_tokens_buckets

        t0 = time.monotonic()
        buckets = get_hybrid_num_tokens_buckets(self.max_num_tokens)
        dev = self._runner.device
        for t in buckets:  # compile + workspace for every bucket, before KV sizing and capture
            x = torch.zeros(t, hidden, dtype=torch.bfloat16, device=dev)
            ids = (torch.arange(t * self.topk, device=dev) % num_experts).to(torch.int32).view(t, self.topk)
            self._run(fm, x, ids, torch.full((t, self.topk), 1.0 / self.topk, device=dev, dtype=torch.float32))
        x, ids, wts, ref = check
        err = float((self._run(fm, x, ids, wts).float() - ref).norm() / ref.norm().clamp_min(1e-30))
        torch.cuda.synchronize(dev)
        ws = sum(t.numel() * t.element_size() for v in _SHARED_WORKSPACES.values() for t in v)
        logger.info(
            "k114: FlashInfer SM12x W4A4 MoE (%s) warmed %d token buckets up to %d in %.1f s, "
            "shared workspace %.1f MiB; first-layer check rel L2 %.4f vs FP32 reference (limit %.2f)",
            FI_PACKAGE, len(buckets), self.max_num_tokens, time.monotonic() - t0, ws / 2**20,
            err, CHECK_MAX_REL_L2,
        )
        if not err <= CHECK_MAX_REL_L2:
            raise RuntimeError(f"k114: FlashInfer SM12x W4A4 MoE failed the load check (rel L2 {err:.4f})")
        _STATE["warmed"] = True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda() and current_platform.is_device_capability_family(120)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key: QuantKey | None, activation_key: QuantKey | None) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()  # the kernel applies the router weights

    def workspace_shapes(self, M, N, K, topk, global_num_experts, local_num_experts,
                         expert_tokens_meta, activation):
        return (1,), (0,), (M, K)  # scratch lives in _SHARED_WORKSPACES

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True  # BF16 in, NVFP4 quantization inside the kernel

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
              expert_tokens_meta, apply_router_weight_on_input):
        if expert_map is not None or apply_router_weight_on_input:
            raise ValueError("k114: FlashInfer SM12x W4A4 MoE takes no expert map or input-side router weights")
        if not _STATE["announced"]:
            _STATE["announced"] = True
            logger.info("k114: routed experts on FlashInfer SM12x W4A4 MoE: %d layers on this rank",
                        _STATE["layers"])
        out = self._run(_fi(), hidden_states.contiguous(), topk_ids.to(torch.int32).contiguous(),
                        topk_weights.to(torch.float32).contiguous())
        output.copy_(out)
