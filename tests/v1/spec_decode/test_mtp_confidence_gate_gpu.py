# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests for the MTP confidence gate: the drafter-confidence kernel, and the
rejection sampler's output distribution when drafting stops early or deepens.

The end-to-end greedy check (depth up to 6 vs fixed depth 4 on the real model)
is the Thunderdome logits capture of the arm against its v3d control.
"""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.draft_confidence import (
    confident_run_length,
    draft_token_probs,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEVICE = "cuda"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab_size", [1000, 151_936])
def test_draft_token_probs_match_reference(dtype, vocab_size):
    torch.manual_seed(0)
    max_reqs, steps = 5, 6
    logits = torch.randn(max_reqs, steps, vocab_size, device=DEVICE).mul(3).to(dtype)
    # A reduced draft vocabulary leaves -inf outside the subset.
    logits[:, :, vocab_size // 2 :] = float("-inf")
    idx_mapping = torch.tensor([3, 0, 4], dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([0.6, 0.0, 1.0, 1.0, 1.3], device=DEVICE)
    tokens = torch.randint(0, vocab_size // 2, (3, steps), device=DEVICE)
    tokens[0, 0] = logits[3, 0].float().argmax()

    q = draft_token_probs(logits, tokens, idx_mapping, temperature, steps)

    ref = torch.empty_like(q)
    for row, slot in enumerate(idx_mapping.tolist()):
        t = temperature[slot].item() or 1.0  # greedy: T=1 confidence
        probs = torch.softmax(logits[slot].float() / t, dim=-1)
        ref[row] = probs.gather(-1, tokens[row][:, None]).squeeze(-1)
    torch.testing.assert_close(q, ref, rtol=1e-4, atol=1e-6)

    # Only the first `num_steps` columns are written.
    assert draft_token_probs(logits, tokens, idx_mapping, temperature, 4).shape == (
        3,
        4,
    )


def _stop_after_first_unconfident(q: torch.Tensor, threshold: float) -> torch.Tensor:
    """Drafts kept per trial: up to and including the first q < threshold."""
    run = confident_run_length(q, threshold)
    return torch.clamp(run + 1, max=q.shape[1])


@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("max_depth", [4, 6])
def test_rejection_sample_exact_with_confidence_stop(temperature, max_depth):
    """Chain stops early at a confidence threshold; verify must stay exact.

    Every position has its own target p_i and drafter q_i (independent of the
    drafted prefix), so given that position i is emitted, its token must follow
    p_i whether it came from an accepted draft, a residual resample or, after a
    stop, a bonus sample (the -1 placeholder path). Chi-square per position.
    """
    torch.manual_seed(1)
    vocab, trials = 64, 40_000
    K = max_depth
    target_logits_pos = torch.randn(K + 1, vocab, device=DEVICE) * 2
    draft_logits_pos = target_logits_pos[:K] + torch.randn(K, vocab, device=DEVICE)

    # Drafts and their proposal probabilities.
    draft_probs = torch.softmax(draft_logits_pos / temperature, dim=-1)  # [K, V]
    drafts = torch.stack(
        [torch.multinomial(draft_probs[i], trials, replacement=True) for i in range(K)],
        dim=1,
    )  # [trials, K]
    q = draft_probs.gather(1, drafts.T).T  # [trials, K]
    keep = _stop_after_first_unconfident(q, 0.3)
    steps = torch.arange(K, device=DEVICE)
    drafts_masked = torch.where(steps[None] < keep[:, None], drafts, -1)

    num_logits = trials * (K + 1)
    target_logits = (target_logits_pos / temperature).repeat(trials, 1)
    draft_logits = draft_logits_pos[None].expand(trials, K, vocab).contiguous()
    draft_sampled = torch.zeros(trials, K + 1, dtype=torch.int64, device=DEVICE)
    draft_sampled[:, 1:] = drafts_masked
    out, num_sampled = rejection_sample(
        target_logits=target_logits,
        draft_logits=draft_logits,
        draft_sampled=draft_sampled.reshape(-1),
        cu_num_logits=torch.arange(trials + 1, dtype=torch.int32, device=DEVICE)
        * (K + 1),
        pos=torch.arange(num_logits, dtype=torch.int32, device=DEVICE),
        idx_mapping=torch.arange(trials, dtype=torch.int32, device=DEVICE),
        expanded_idx_mapping=torch.arange(
            trials, dtype=torch.int32, device=DEVICE
        ).repeat_interleave(K + 1),
        expanded_local_pos=torch.arange(K + 1, dtype=torch.int32, device=DEVICE).repeat(
            trials
        ),
        temperature=torch.full((trials,), temperature, device=DEVICE),
        seed=torch.arange(trials, dtype=torch.int64, device=DEVICE),
        num_speculative_steps=K,
    )
    # Never emit past a stop: at most kept drafts + 1 tokens.
    assert bool((num_sampled.long() <= keep + 1).all())

    target_probs = torch.softmax(target_logits_pos / temperature, dim=-1)
    for i in range(K + 1):
        emitted = num_sampled > i
        n = int(emitted.sum())
        if n < 2000:
            continue
        observed = torch.bincount(out[emitted, i], minlength=vocab).float()
        expected = target_probs[i] * n
        ok = expected >= 5
        obs = torch.cat([observed[ok], observed[~ok].sum()[None]])
        exp = torch.cat([expected[ok], expected[~ok].sum()[None]])
        if exp[-1] < 5:
            obs, exp = obs[:-1], exp[:-1]
        chi2 = ((obs - exp) ** 2 / exp).sum().item()
        df = obs.numel() - 1
        assert chi2 < df + 10 * (2 * df) ** 0.5, (
            f"position {i}: chi2={chi2:.1f} df={df} n={n}"
        )


def test_greedy_depth6_emits_depth4_tokens():
    """Greedy: a deeper chain emits the same tokens, just more per round.

    The target is a deterministic chain (argmax of row i is tokens[i]); the
    drafter matches it for a while and then diverges. Rounds at depth 6 and at
    depth 4 must produce the same token stream.
    """
    vocab, length = 32, 40
    torch.manual_seed(2)
    chain = torch.randint(0, vocab, (length + 8,), device=DEVICE)
    drafter = chain.clone()
    drafter[torch.arange(5, length + 8, 7, device=DEVICE)] += 1
    drafter %= vocab

    def run(depth: int) -> list[int]:
        out: list[int] = []
        while len(out) < length:
            start = len(out)
            rows = torch.full((depth + 1, vocab), -10.0, device=DEVICE)
            rows[torch.arange(depth + 1), chain[start : start + depth + 1]] = 10.0
            drafted = torch.zeros(depth + 1, dtype=torch.int64, device=DEVICE)
            drafted[1:] = drafter[start : start + depth]
            sampled, num = rejection_sample(
                target_logits=rows,
                draft_logits=None,
                draft_sampled=drafted,
                cu_num_logits=torch.tensor(
                    [0, depth + 1], dtype=torch.int32, device=DEVICE
                ),
                pos=torch.arange(
                    start, start + depth + 1, dtype=torch.int32, device=DEVICE
                ),
                idx_mapping=torch.zeros(1, dtype=torch.int32, device=DEVICE),
                expanded_idx_mapping=torch.zeros(
                    depth + 1, dtype=torch.int32, device=DEVICE
                ),
                expanded_local_pos=torch.arange(
                    depth + 1, dtype=torch.int32, device=DEVICE
                ),
                temperature=torch.zeros(1, device=DEVICE),
                seed=torch.zeros(1, dtype=torch.int64, device=DEVICE),
                num_speculative_steps=depth,
            )
            out.extend(sampled[0, : int(num[0])].tolist())
        return out[:length]

    assert run(6) == run(4) == chain[:length].tolist()
