# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduced draft vocabulary for the Qwen3.8-Flash-Next MTP head.

Idea and the K=65536 id list ported from dime-online/qwen3.8-Flash-DGX-UltraFast
(Apache-2.0, recipe/build/image/src/vllm_mtp_draft_vocab.py). This version is
TP-sharded and works on the b12x loader's routed (meta) checkpoint views.

Torch-only on purpose so tests/models/test_qwen38_draft_vocab_cpu.py can load it
without importing vLLM.
"""

from __future__ import annotations

import gzip
from collections.abc import Callable

import torch


def load_draft_vocab_ids(path: str, vocab_size: int) -> torch.Tensor:
    """Sorted unique int64 target ids from an id-list file (one per line,
    '#' comments, optionally gzipped). Refuses empty lists and bad ids."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        ids = {
            int(line.split("#", 1)[0]) for line in fh if line.split("#", 1)[0].strip()
        }
    if not ids:
        raise ValueError(f"draft vocab {path}: empty id list")
    if min(ids) < 0 or max(ids) >= vocab_size:
        raise ValueError(f"draft vocab {path}: ids must lie in [0, {vocab_size})")
    return torch.tensor(sorted(ids), dtype=torch.long, device="cpu")


def scatter_draft_logits(
    logits: torch.Tensor, ids: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """[N, K] subset logits -> [N, vocab_size], -inf outside the subset.

    softmax(-inf) is exactly 0, so the draft proposal q has support only on
    ``ids``. Rejection sampling accepts x with min(1, p(x)/q(x)) and resamples
    from max(p - q, 0) / Z, which equals p off the subset; the emitted
    distribution is p for any q, so only acceptance can move."""
    full = logits.new_full((logits.shape[0], vocab_size), float("-inf"))
    return full.index_copy_(1, ids, logits)


def gather_shard_rows(
    weight: torch.Tensor,
    local_ids: torch.Tensor,
    materialize: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """Rows ``local_ids`` (sorted, CPU) of a full [V, H] head weight.

    Narrows to the contiguous span covering this rank's ids first: under the
    b12x loader ``weight`` is a meta view routed to the checkpoint, and a
    narrow is a view it can read, while index_select is not. ``materialize``
    turns a meta span into real data; None keeps it meta (counting pass)."""
    lo = int(local_ids[0])
    span = weight.narrow(0, lo, int(local_ids[-1]) + 1 - lo)
    if span.is_meta and materialize is not None:
        span = materialize(span)
    return span.index_select(0, (local_ids - lo).to(span.device))
