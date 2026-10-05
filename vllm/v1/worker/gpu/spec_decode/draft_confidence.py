# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafter confidence for the MTP confidence gate (VLLM_MTP_CONFIDENCE_THRESHOLD).

For each request and draft step this computes q(x), the probability the drafter
gave the token it proposed, from the cached pre-temperature draft logits:
q = softmax(logits / T)[x], with T = 1 for greedy requests (their proposal is
the argmax, so T = 1 only measures how peaked the drafter was). The confident
run of a request is the number of leading drafts with q >= threshold.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _draft_token_prob_kernel(
    # [num_reqs, num_steps] fp32 out
    q_ptr,
    q_stride,
    # [max_num_reqs, max_steps, V] pre-temperature draft logits
    logits_ptr,
    logits_stride_0,
    logits_stride_1,
    # [num_reqs, max_steps] proposed tokens, batch order
    tokens_ptr,
    tokens_stride,
    # [num_reqs] request state slots
    idx_mapping_ptr,
    # [max_num_reqs] temperatures by request state slot
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    step = tl.program_id(1)
    slot = tl.load(idx_mapping_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + slot).to(tl.float32)
    temp = tl.where(temp > 0.0, temp, 1.0)
    base = logits_ptr + slot * logits_stride_0 + step * logits_stride_1

    # Online max / sum-exp over the vocab. Columns outside a reduced draft
    # vocabulary hold -inf and contribute nothing.
    m = tl.full((), float("-inf"), tl.float32)
    s = tl.zeros((), tl.float32)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(base + offs, mask=offs < vocab_size, other=float("-inf"))
        x = x.to(tl.float32) / temp
        new_m = tl.maximum(m, tl.max(x, axis=0))
        alpha = tl.where(new_m == float("-inf"), 1.0, tl.exp(m - new_m))
        e = tl.where(x == float("-inf"), 0.0, tl.exp(x - new_m))
        s = s * alpha + tl.sum(e, axis=0)
        m = new_m

    token = tl.load(tokens_ptr + row * tokens_stride + step).to(tl.int64)
    x_tok = tl.load(base + token).to(tl.float32) / temp
    q = tl.where(s > 0.0, tl.exp(x_tok - m) / s, 0.0)
    tl.store(q_ptr + row * q_stride + step, q)


def draft_token_probs(
    draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """q(x) of each proposed token, [num_reqs, num_steps] fp32."""
    num_reqs = idx_mapping.shape[0]
    q = torch.empty(
        num_reqs, num_steps, dtype=torch.float32, device=draft_logits.device
    )
    if num_reqs == 0 or num_steps == 0:
        return q
    assert draft_logits.stride(-1) == 1 and draft_tokens.stride(-1) == 1
    _draft_token_prob_kernel[(num_reqs, num_steps)](
        q,
        q.stride(0),
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_tokens,
        draft_tokens.stride(0),
        idx_mapping,
        temperature,
        draft_logits.shape[-1],
        BLOCK_SIZE=4096,
    )
    return q


def confident_run_length(q: torch.Tensor, threshold: float) -> torch.Tensor:
    """Leading drafts with q >= threshold per request, int32 [num_reqs]."""
    return (q >= threshold).to(torch.int32).cumprod(dim=1).sum(dim=1, dtype=torch.int32)
