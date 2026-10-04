# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the MTP draft-prefill tail routing (VLLM_MTP_PREFILL_TAIL_ROUTING)."""

import torch

from vllm.models.qwen3_8_flash_next.tail_routing import (
    TailRouting,
    install_tail_routing,
)

# 4 requests: two decode rows windows (5 rows), a 2-row window and a 5-row
# window, then padding up to 20 rows. Tails sit before rejected rows.
QSL = torch.tensor([0, 5, 10, 12, 17, 17, 17], dtype=torch.int32)
TAILS = torch.tensor([2, 9, 10, 15, 0, 0], dtype=torch.int64)
EXPECTED = [2] * 5 + [9] * 5 + [10] * 2 + [15] * 5 + [17, 18, 19]


class _TopKRouter:
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None):
        weights, ids = torch.topk(router_logits.softmax(-1), self.top_k, dim=-1)
        return weights, ids.to(torch.int32)


def test_sources_map_each_row_to_its_request_tail() -> None:
    routing = TailRouting(20, "cpu")
    routing.begin(QSL, TAILS, 4)
    assert routing.select(20).tolist() == EXPECTED
    assert routing.select(12).tolist() == EXPECTED[:12]


def test_inactive_oversized_and_empty_batches_keep_own_rows() -> None:
    routing = TailRouting(20, "cpu")
    assert routing.select(4) is None
    routing.begin(QSL, TAILS, 4)
    assert routing.select(21) is None
    routing.end()
    assert routing.select(4) is None
    routing.begin(QSL, TAILS, 0)
    assert routing.select(20).tolist() == list(range(20))


def test_stale_tail_index_stays_inside_its_request() -> None:
    routing = TailRouting(20, "cpu")
    routing.begin(QSL, torch.tensor([7, 3, 99, 15]), 4)
    assert routing.select(17).tolist() == [4] * 5 + [5] * 5 + [11] * 2 + [15] * 5


def test_router_wrapper_keeps_tail_routes_and_shrinks_expert_set() -> None:
    torch.manual_seed(0)
    router = _TopKRouter(top_k=10)
    hidden = torch.randn(20, 8)
    logits = torch.randn(20, 512)
    plain_w, plain_ids = router.select_experts(hidden, logits)

    routing = TailRouting(64, "cpu")
    install_tail_routing(router, routing)
    routing.begin(QSL, TAILS, 4)
    w, ids = router.select_experts(hidden, logits)
    assert ids.dtype == plain_ids.dtype and ids.is_contiguous()
    tails = TAILS[:4]
    torch.testing.assert_close(w[tails], plain_w[tails], rtol=0, atol=0)
    assert torch.equal(ids[tails], plain_ids[tails])
    src = torch.tensor(EXPECTED)
    assert torch.equal(ids, plain_ids[src])
    used = torch.unique(plain_ids[tails]).numel() + torch.unique(plain_ids[17:]).numel()
    assert torch.unique(ids).numel() <= used < torch.unique(plain_ids).numel()

    routing.end()
    w, ids = router.select_experts(hidden, logits)
    assert torch.equal(ids, plain_ids) and torch.equal(w, plain_w)
