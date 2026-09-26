# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X-backed HyperConnection modules for Qwen3.8-Flash-Next."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.weight_transfer import allocate_weights
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    set_b12x_preparation_provider,
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_hyperconnection,
)

logger = init_logger(__name__)


def _hyperconnection_api() -> Any:
    api = get_b12x_hyperconnection()
    if api is None:
        raise ImportError(
            "Qwen3.8-Flash-Next requires b12x.norm.hyperconnection; "
            "install the b12x serving extra"
        )
    return api



# ---------------------------------------------------------------------------
# Online MXFP8 weights for the hyper-connection mixers and the MoE router gate
# (lever B2, results/kernel-pass/lmhead-hc-design.md section 5).
#
# The HC mixer streams the largest dense weight bucket in the model: two
# replicated BF16 projections per GatedResidual,
#
#   input_mix_weight_down_block_inject  [336, 10240]  6.88 MB  (merged, 12 pad)
#   input_mix_weight_up                 [10240, 320]  6.55 MB
#
# x 97 instances/step = 1303 MB/node/step, measured at 8.22 ms, i.e. 158 GB/s
# against a 273 GB/s LPDDR5x roofline. The MoE router gate adds [512, 2560]
# x 48 = 126 MB at 0.85 ms. Quantizing those weights to MXFP8 g32 once at load
# halves the stream with zero extra collectives and zero extra launches.
#
# Weight loading is untouched. ``create_weights`` still comes from
# ``UnquantizedLinearMethod``, so the ``MergedColumnParallelLinear`` shard
# loaders, the 12 alignment pad rows and ``_HC_WEIGHTS_MAPPER`` see exactly the
# parameter they saw before. The in-tree ``Mxfp8OnlineLinearMethod`` quantizes
# that loaded BF16 weight once, in ``process_weights_after_loading``, and hands
# the packed weight to b12x's block-scaled linear -- the same kernel that
# already serves this model's MXFP8 linear-attention projections.
#
# Activation precision is left to b12x (``VLLM_B12X_MXFP8_ACTIVATION_MODE`` /
# ``VLLM_B12X_DENSE_ACTIVATION_MODE``, default "auto"), which resolves it per
# exact-M regime: BF16 activations (W8A16) for the decode regimes at M <= 8
# where ``in_features == padded_in_features``, and the quantized-activation
# path at prefill capacity. Pinning "a16" here would pin the 16-row A16 tile
# for prefill too, so it is an operator choice, not a patch decision.
#
# NOT bit-identical: MXFP8 rounding is ~2^-8 relative per weight, so logits
# move by ~1e-2 and ``scripts/logits_equiv.py`` fails by construction. Judge on
# behaviour. Gated by ``VLLM_QWEN38_HC_MXFP8`` (default "off"; "hc",
# "gate" or "off"). Mutually exclusive with mods/vllm-qwen38-bf16-gemv, which
# rebinds ``quant_method`` on the same modules.
#
# The gate is declared in ``vllm/envs.py`` and read through ``envs``, never
# through a bare ``os.getenv``. Only ``environment_variables`` entries reach
# ``envs.compile_factors()`` and therefore the AOT compile-cache key, and this
# gate changes how many b12x plans a boot creates. b12x plan handles are a
# process-local monotonic counter (``b12x/preparation/types.py``) that torch
# bakes into the compiled graph as an integer constant, so two target sets
# sharing one cache entry would dereference each other's plans -- which is
# exactly the `plan belongs to gemm.blockscaled_precision, not
# norm.hyperconnection` crash the "hc"-only arm hit on 2026-09-22.
# ---------------------------------------------------------------------------

_HC_MXFP8_TARGETS = frozenset(
    part
    for part in (envs.VLLM_QWEN38_HC_MXFP8 or "hc,gate").lower().replace(" ", "").split(",")
    if part and part != "off"
)

# MXFP8 microscaling group, and b12x's A16 / workspace-quantization geometry
# gates (b12x/gemm/blockscaled/_tuning.py::_validate_config).
_MXFP8_GROUP = 32
_MXFP8_N_ALIGN = 8

_HC_MXFP8_LOGGED: set = set()


def _hc_mxfp8_method() -> Any:
    """An online MXFP8 linear method bound to b12x, or None when unavailable."""
    try:
        from vllm.model_executor.kernels.linear.mxfp8.b12x import (
            B12xMxfp8LinearKernel,
        )
        from vllm.model_executor.layers.quantization.online.mxfp8 import (
            Mxfp8OnlineLinearMethod,
        )

        method = Mxfp8OnlineLinearMethod()
    except Exception:  # noqa: BLE001 - absence is a routing decision
        return None
    if not isinstance(method.kernel, B12xMxfp8LinearKernel):
        return None
    return method


def _hc_mxfp8_admits(weight: torch.Tensor) -> bool:
    """Whether b12x's block-scaled contract can serve this weight geometry."""
    return (
        weight.dim() == 2
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and int(weight.shape[1]) > 0
        and int(weight.shape[1]) % _MXFP8_GROUP == 0
        and int(weight.shape[0]) > 0
        and int(weight.shape[0]) % _MXFP8_N_ALIGN == 0
    )


class HcMxfp8LinearMethod(UnquantizedLinearMethod):
    """BF16 weight loading, MXFP8 weights, b12x block-scaled GEMM at run.

    Rebound onto a constructed ``LinearBase`` *after* ``create_weights`` has
    run, so the layer keeps its stock BF16 parameter and its stock loaders.
    ``process_weights_after_loading`` then does the one-shot quantization; a
    geometry b12x cannot serve keeps the stock cuBLAS path and says so.
    """

    def __init__(self, method: Any, target: str) -> None:
        super().__init__()
        self.mxfp8_method = method
        self.target = target
        self.engaged = False

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight = getattr(layer, "weight", None)
        prefix = str(getattr(layer, "prefix", "") or "?")
        if weight is None or not _hc_mxfp8_admits(weight.data):
            shape = None if weight is None else tuple(weight.shape)
            logger.warning_once(
                "qwen38 HC MXFP8: target=%s %s shape=%s is outside the b12x "
                "block-scaled contract (K%%%d, N%%%d, contiguous BF16 on CUDA) "
                "-- keeping BF16",
                self.target, prefix, shape, _MXFP8_GROUP, _MXFP8_N_ALIGN,
            )
            return super().process_weights_after_loading(layer)
        out_features, in_features = (int(value) for value in weight.shape)
        self.mxfp8_method.process_weights_after_loading(layer)
        self.engaged = True
        key = (self.target, out_features, in_features)
        if key not in _HC_MXFP8_LOGGED:
            _HC_MXFP8_LOGGED.add(key)
            logger.info(
                "qwen38 HC MXFP8: target=%s shape=[%d,%d] -> mxfp8 g32, "
                "activation_mode=%s, %.2f MB -> %.2f MB per instance",
                self.target, out_features, in_features,
                getattr(layer, "b12x_activation_mode", "?"),
                out_features * in_features * 2 / 1e6,
                _mxfp8_bytes(out_features, in_features) / 1e6,
            )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.engaged:
            return super().apply(layer, x, bias)
        return self.mxfp8_method.apply(layer, x, bias)


def _mxfp8_bytes(out_features: int, in_features: int) -> int:
    """Packed MXFP8 weight + swizzled UE8M0 scale bytes b12x actually streams.

    ``blockscaled.pack_weight`` pads K up to 128 and stores the scales in the
    F8_128x4 layout, so a K that is not a multiple of 128 (the HC
    up-projection's 320) costs more than K/32 scale bytes.
    """
    padded_k = -(-in_features // 128) * 128
    n_tiles = -(-out_features // 128)
    k_tiles = -(-(padded_k // _MXFP8_GROUP) // 4)
    return out_features * padded_k + n_tiles * k_tiles * 512


def maybe_route_hc_mxfp8(linear: nn.Module, target: str) -> bool:
    """Rebind ``linear`` to online MXFP8 weights when its target is enabled.

    Called at construction time, after ``create_weights``. Only the method is
    replaced, so the parameter, its loaders and its shard layout are the stock
    unquantized ones; the weight is quantized once, at the end of loading.
    """
    if target not in _HC_MXFP8_TARGETS or not current_platform.is_cuda():
        return False
    if type(getattr(linear, "quant_method", None)) is not UnquantizedLinearMethod:
        return False
    method = _hc_mxfp8_method()
    if method is None:
        return False
    linear.quant_method = HcMxfp8LinearMethod(method, target)
    return True


@dataclass(frozen=True)
class HyperConnectionConfig:
    hc_count: int
    hidden_size: int
    params_dtype: torch.dtype
    hc_lowrank: int
    rms_norm_eps: float
    hc_per_branch_norm: bool = True


class GroupedGemmaRMSNorm(nn.Module):
    """Checkpoint-compatible zero-centered grouped RMSNorm weight."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by group_size={group_size}"
            )
        self.eps = float(eps)
        self.group_size = group_size
        self.weight = nn.Parameter(
            allocate_weights(torch.zeros, hidden_size, dtype=dtype)
        )


class HyperConnectionWorkspace(nn.Module):
    """Fixed-capacity storage shared by all HC modules in one model."""

    def __init__(self, config: HyperConnectionConfig, max_tokens: int) -> None:
        super().__init__()
        if not config.hc_per_branch_norm:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next requires one RMSNorm group per HC stream"
            )
        self.config = config
        self.max_tokens = int(max_tokens)
        self.device = torch.device(current_platform.current_device())
        width = config.hc_count * config.hidden_size
        factory = dict(device=self.device, dtype=config.params_dtype)
        self.register_buffer(
            "normalized", torch.empty(max_tokens, width, **factory), persistent=False
        )
        self.register_buffer(
            "bottleneck",
            torch.empty(max_tokens, config.hc_lowrank, **factory),
            persistent=False,
        )
        self.register_buffer(
            "block_input",
            torch.empty(max_tokens, config.hidden_size, **factory),
            persistent=False,
        )

    def caps(self, max_tokens: int):
        api = _hyperconnection_api()
        return api.Caps(
            device=self.device,
            max_tokens=max_tokens,
            hidden_size=self.config.hidden_size,
            streams=self.config.hc_count,
            lowrank=self.config.hc_lowrank,
            dtype=self.config.params_dtype,
        )

    def bind(self, plan, tokens: int):
        return _hyperconnection_api().bind(
            plan,
            normalized=self.normalized,
            bottleneck=self.bottleneck,
            block_input=self.block_input,
            tokens=tokens,
        )


class GatedResidual(nn.Module):
    """Learned HC mixer with B12X pointwise and residual kernels."""

    def __init__(
        self,
        config: HyperConnectionConfig,
        workspace: HyperConnectionWorkspace,
        *,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.params_dtype != torch.bfloat16:
            raise TypeError("Qwen3.8-Flash-Next HC requires BF16 parameters")
        self.config = config
        # The workspace is owned once by the enclosing model.  Keeping a plain
        # reference here avoids registering the same large buffer module under
        # every decoder block.
        object.__setattr__(self, "_workspace", workspace)
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = self.hyper_hidden_size
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=config.hidden_size,
            dtype=config.params_dtype,
        )

        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            sizes = [self.lora_rank, self.hc_count]
            if self.pad_size:
                sizes.append(self.pad_size)
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                sizes,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
            maybe_route_hc_mxfp8(
                self.input_mix_weight_down_block_inject, "hc"
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
            maybe_route_hc_mxfp8(self.input_mix_weight_down, "hc")
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )
        maybe_route_hc_mxfp8(self.input_mix_weight_up, "hc")
        self._preparation_prefix = prefix or "qwen3_8_flash_next.hyperconnection"
        self._plans: dict[str, object] = {}
        if not getattr(self, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(self, self)
    def _request_name(self, operation: str, tokens: int) -> str:
        return f"{self._preparation_prefix}.hc.{operation}.m{tokens}"

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> Sequence[B12xPreparationUnit]:
        if layer is not self:
            raise ValueError("HC preparation owner mismatch")
        if workload.stage != "weights":
            return ()
        if self.hc_norm.weight.is_meta:
            return ()
        if workload.max_tokens > self.workspace.max_tokens:
            raise PreparationResourceUnavailableError(
                f"{self._preparation_prefix} HC workspace capacity "
                f"{self.workspace.max_tokens} cannot serve {workload.max_tokens}"
            )
        api = _hyperconnection_api()
        operations = (
            "grouped_rmsnorm",
            "scaled_silu",
            "gate_mean",
            "combine",
            "combine_norm",
        )
        requests = []
        tokens = workload.max_tokens
        plans = {}
        for operation in operations:
            plan = api.plan(
                self.workspace.caps(tokens),
                invocation={"operation": operation, "eps": self.config.rms_norm_eps},
            )
            plans[operation] = plan
            requests.append(plan.request(
                name=self._request_name(operation, tokens),
                prepare_call=self._prepare_call(operation, tokens),
                benchmark_call=self._benchmark_call(operation, tokens),
            ))
        self._plans = plans
        return (
            B12xPreparationUnit(
                name="HYPERCONNECTION",
                key=(self._preparation_prefix, tokens),
                requests=tuple(requests),
                stage="weights",
                autotune=not workload.eager_only,
            ),
        )

    def _prepare_call(self, operation: str, tokens: int):
        """Prime the installed operation against its durable workspace."""
        return self._call_factory(operation, tokens, benchmark=False)

    def _benchmark_call(self, operation: str, tokens: int):
        """Measure an isolated binding; it is never retained for serving."""
        return self._call_factory(operation, tokens, benchmark=True)

    def _call_factory(self, operation: str, tokens: int, *, benchmark: bool):
        def prepare(state):
            from b12x.preparation import PreparedCall
            from b12x.norm.hyperconnection import _impl

            factory = dict(device=self.workspace.device, dtype=self.config.params_dtype)
            width = self.hyper_hidden_size
            activation_inputs = []
            activation_owners = []

            def activation(shape):
                value = torch.empty(shape, **factory)
                template = torch.arange(value.numel(), **factory).reshape(shape)
                template.div_(max(value.numel(), 1))
                activation_inputs.append((value, template))
                activation_owners.extend((value, template))
                return value

            def produce():
                for value, template in activation_inputs:
                    value.copy_(template)

            # Serving priming writes its owned workspace. Trials instead own
            # independent output storage so no measured binding can escape.
            def output(shape, serving):
                return torch.empty(shape, **factory) if benchmark else serving

            if operation == "grouped_rmsnorm":
                source = activation((tokens, width))
                out = output((tokens, width), self.workspace.normalized)
                run = lambda: _impl.run_grouped_rmsnorm_impl(
                    source, self.hc_norm.weight, eps=self.config.rms_norm_eps,
                    plan=state, out=out,
                )
                owners = (out,)
            elif operation == "scaled_silu":
                source = activation((tokens, self.lora_rank))
                out = output((tokens, self.lora_rank), self.workspace.bottleneck)
                run = lambda: _impl.run_scaled_silu_impl(source, plan=state, out=out)
                owners = (out,)
            elif operation == "gate_mean":
                source = activation((tokens, width))
                gates = activation((tokens, width))
                out = output((tokens, self.hidden_size), self.workspace.block_input)
                run = lambda: _impl.run_gate_mean_impl(source, gates, plan=state, out=out)
                owners = (out,)
            else:
                hidden = activation((tokens, width))
                block = activation((tokens, self.hidden_size))
                injection = activation((tokens, self.hc_count))
                if operation == "combine":
                    run = lambda: _impl.run_combine_impl(
                        hidden, block, injection, plan=state,
                    )
                else:
                    run = lambda: _impl.run_combine_norm_impl(
                        hidden, block, injection, self.hc_norm.weight,
                        eps=self.config.rms_norm_eps, plan=state,
                    )
                owners = ()
            return PreparedCall(
                run=run,
                produce=produce,
                owners=(*activation_owners, *owners),
            )
        return prepare

    def _plan_for(self, operation: str):
        try:
            return self._plans[operation]
        except KeyError:
            raise PreparationResourceUnavailableError(
                f"{self._preparation_prefix} lacks a declared {operation} plan"
            ) from None

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def workspace(self) -> HyperConnectionWorkspace:
        return self._workspace

    def _binding(self, hidden_states: torch.Tensor, operation: str):
        return self.workspace.bind(self._plan_for(operation), hidden_states.shape[0])

    def _mix_normalized(self, normalized: torch.Tensor):
        api = _hyperconnection_api()
        if self.use_combine:
            down_and_injection = self.input_mix_weight_down_block_inject(normalized)
            projected_down = down_and_injection[:, : self.lora_rank]
            injection_start = self.lora_rank
            # The projection owner stays live through the downstream residual
            # combine; readers consume row-strided slices without staging.
            injection = down_and_injection[
                :, injection_start : injection_start + self.hc_count
            ]
        else:
            projected_down = self.input_mix_weight_down(normalized)
            injection = None

        bottleneck = api.run_scaled_silu(
            projected_down, binding=self._binding(normalized, "scaled_silu")
        )
        gate_logits = self.input_mix_weight_up(bottleneck)
        block_input = api.run_gate_mean(
            normalized,
            gate_logits,
            binding=self._binding(normalized, "gate_mean"),
        )
        return block_input, injection

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        api = _hyperconnection_api()
        binding = self._binding(hidden_states, "grouped_rmsnorm")
        normalized = api.run_grouped_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            eps=self.config.rms_norm_eps,
            binding=binding,
        )
        block_input, injection = self._mix_normalized(normalized)
        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        api = _hyperconnection_api()
        combined, normalized = api.run_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            eps=self.config.rms_norm_eps,
            plan=self._plan_for("combine_norm"),
        )
        block_input, injection = self._mix_normalized(normalized)
        return combined, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        api = _hyperconnection_api()
        return api.run_combine(
            hidden_states,
            block_output,
            injection,
            plan=self._plan_for("combine"),
        )


__all__ = [
    "GatedResidual",
    "HcMxfp8LinearMethod",
    "maybe_route_hc_mxfp8",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
    "HyperConnectionWorkspace",
]
