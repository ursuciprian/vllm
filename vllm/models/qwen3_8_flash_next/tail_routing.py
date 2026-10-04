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

from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class TailRouting:
    """Row -> source-row map applied to the router output while active."""

    def __init__(self, capacity: int, device: torch.device | str) -> None:
        self.rows = torch.arange(capacity, dtype=torch.int64, device=device)
        self.sources = self.rows.clone()
        self.active = False
        self.engaged = False

    def begin(
        self,
        query_start_loc: torch.Tensor,
        last_token_indices: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Map each row of requests [0, num_reqs) to its request's tail row.

        Rows past the last request keep their own routing. Runs eagerly before
        the draft prefill (graph replays read ``sources``), with no host sync.
        """
        if num_reqs <= 0:
            self.sources.copy_(self.rows)
        else:
            bounds = query_start_loc[: num_reqs + 1].to(torch.int64)
            starts, ends = bounds[:-1], bounds[1:]
            req = torch.searchsorted(ends, self.rows, right=True)
            inside = req < num_reqs
            req.clamp_(max=num_reqs - 1)
            # Clamp into the request's own rows so a stale index cannot leave it.
            tail = last_token_indices[:num_reqs].to(torch.int64)[req]
            tail = torch.minimum(torch.maximum(tail, starts[req]), ends[req] - 1)
            torch.where(inside, tail, self.rows, out=self.sources)
        self.active = True

    def end(self) -> None:
        self.active = False

    def select(self, num_rows: int) -> torch.Tensor | None:
        if not self.active or num_rows > self.sources.numel():
            return None
        return self.sources[:num_rows]


def install_tail_routing(router: Any, routing: TailRouting) -> None:
    """Wrap ``router.select_experts`` so active rows take their source's routes."""
    select_experts = router.select_experts

    def tail_routed_select_experts(*args, **kwargs):
        topk_weights, topk_ids = select_experts(*args, **kwargs)
        sources = routing.select(topk_ids.shape[0])
        if sources is None:
            return topk_weights, topk_ids
        if not routing.engaged:
            routing.engaged = True
            logger.info("MTP prefill tail routing engaged (%d rows)", sources.numel())
        return topk_weights.index_select(0, sources), topk_ids.index_select(0, sources)

    router.select_experts = tail_routed_select_experts
