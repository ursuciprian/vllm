# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-prefill tail routing for the MTP draft layer MoE.

The first draft pass runs the draft layer over every verified position of a
request, but only the request's tail row (its last accepted token) is sampled
and fed back. The other rows only need their KV entries, which attention writes
before the MoE. Giving every row of a request its tail row's experts keeps the
tail rows' routing unchanged and cuts the experts the MoE streams from those of
num_tokens rows to those of num_reqs rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )

logger = init_logger(__name__)


class TailRouting:
    """Row -> source-row map applied to the router output while active."""

    def __init__(self, capacity: int, device: torch.device | str) -> None:
        self.rows = torch.arange(capacity, dtype=torch.int64, device=device)
        self.sources = self.rows.clone()
        self.active = False

    def begin(
        self,
        query_start_loc: torch.Tensor,
        last_token_indices: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Map each row of requests [0, num_reqs) to its request's tail row.

        Rows past the last request keep their own routing. Runs eagerly before
        the draft prefill (graph replays read ``sources``), with no host sync.
        A tail lies inside its request, so every source is below the row count;
        at capture ``last_token_indices`` is zeroed, mapping rows to row 0.
        """
        if num_reqs <= 0:
            self.sources.copy_(self.rows)
        else:
            ends = query_start_loc[1 : num_reqs + 1].to(torch.int64)
            req = torch.searchsorted(ends, self.rows, right=True)
            tails = last_token_indices[:num_reqs][req.clamp(max=num_reqs - 1)]
            torch.where(req < num_reqs, tails, self.rows, out=self.sources)
        self.active = True

    def end(self) -> None:
        self.active = False

    def select(self, num_rows: int) -> torch.Tensor | None:
        if not self.active or num_rows > self.sources.numel():
            return None
        return self.sources[:num_rows]


def install_tail_routing(router: FusedMoERouter, routing: TailRouting) -> None:
    """Wrap ``router.select_experts`` so active rows take their source's routes."""
    select_experts = router.select_experts

    def tail_routed_select_experts(*args, **kwargs):
        topk_weights, topk_ids = select_experts(*args, **kwargs)
        sources = routing.select(topk_ids.shape[0])
        if sources is None:
            return topk_weights, topk_ids
        logger.info_once("MTP prefill tail routing engaged in the draft MoE")
        return topk_weights.index_select(0, sources), topk_ids.index_select(0, sources)

    router.select_experts = tail_routed_select_experts  # type: ignore[method-assign]
