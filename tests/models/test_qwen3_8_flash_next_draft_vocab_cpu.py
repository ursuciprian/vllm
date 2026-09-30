# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the Qwen3.8-Flash-Next reduced MTP draft vocabulary.

Loads vllm/models/qwen3_8_flash_next/draft_vocab.py by path (torch only), so it
runs without a vLLM build: python -m pytest <this file>.
"""

import gzip
import importlib.util
from pathlib import Path

import pytest
import torch

_SRC = (
    Path(__file__).resolve().parents[2]
    / "vllm/models/qwen3_8_flash_next/draft_vocab.py"
)
_spec = importlib.util.spec_from_file_location("draft_vocab", _SRC)
dv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dv)

V = 1000


def test_load_ids_text_gz_comments(tmp_path):
    text = "# header\n5\n3  # trailing\n\n5\n999\n0\n"
    plain = tmp_path / "ids.txt"
    plain.write_text(text)
    gz = tmp_path / "ids.txt.gz"
    with gzip.open(gz, "wt") as fh:
        fh.write(text)
    for p in (plain, gz):
        ids = dv.load_draft_vocab_ids(str(p), V)
        assert ids.tolist() == [0, 3, 5, 999] and ids.dtype == torch.long


@pytest.mark.parametrize("text", ["# only comments\n", "1\n1000\n", "-1\n"])
def test_load_ids_rejects(tmp_path, text):
    p = tmp_path / "bad.txt"
    p.write_text(text)
    with pytest.raises(ValueError):
        dv.load_draft_vocab_ids(str(p), V)


def test_scatter_places_logits_and_masks_rest():
    ids = torch.tensor([2, 7, 9])
    sub = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    full = dv.scatter_draft_logits(sub, ids, 12)
    assert full.shape == (2, 12)
    assert torch.equal(full[:, ids], sub)
    mask = torch.ones(12, dtype=torch.bool)
    mask[ids] = False
    assert torch.isneginf(full[:, mask]).all()


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_shard_rows_match_full_gather(tp):
    torch.manual_seed(0)
    weight = torch.randn(V, 8)
    ids = torch.randperm(V)[:256].sort().values
    k = ids.numel() // tp
    parts = []
    for r in range(tp):
        local = ids[r * k : (r + 1) * k]
        parts.append(dv.gather_shard_rows(weight, local, materialize=None))
    # Concatenated TP shards == the full-vocab head restricted to ids, in order.
    assert torch.equal(torch.cat(parts), weight[ids])


def test_shard_rows_meta_path_materializes_only_the_span():
    weight = torch.empty(V, 8, device="meta")
    local = torch.tensor([100, 101, 150, 400])
    seen = []

    def materialize(span):
        seen.append(span.shape)
        return torch.arange(span.numel(), dtype=torch.float32).view(span.shape)

    counted = dv.gather_shard_rows(weight, local, materialize=None)
    assert counted.is_meta and counted.shape == (4, 8)
    assert seen == []
    rows = dv.gather_shard_rows(weight, local, materialize=materialize)
    assert seen == [torch.Size([301, 8])]  # rows 100..400 only
    assert torch.equal(rows[:, 0], (local - 100).float() * 8)


def _rejection_output(p, q):
    """Exact one-step distribution of speculative sampling with proposal q."""
    accept = torch.minimum(q, p)  # q(x) * min(1, p(x)/q(x))
    residual = torch.clamp(p - q, min=0)
    return accept + (1 - accept.sum()) * residual / residual.sum()


def _kernel_residual(p_logits, q_logits):
    """Residual as vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
    _resample_kernel builds it in log space (-inf draft entries included)."""
    lp = p_logits - torch.logsumexp(p_logits, -1)
    lq = q_logits - torch.logsumexp(q_logits, -1)
    ratio = torch.exp(lq - lp)
    res = torch.where(ratio < 1.0, lp + torch.log1p(-ratio), float("-inf"))
    return torch.softmax(res, -1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_target_distribution_unchanged(seed):
    g = torch.Generator().manual_seed(seed)
    vocab = 64
    p_logits = torch.randn(vocab, generator=g, dtype=torch.float64) * 3
    ids = torch.randperm(vocab, generator=g)[:16].sort().values
    sub = torch.randn(16, generator=g, dtype=torch.float64) * 3
    q_logits = dv.scatter_draft_logits(sub[None], ids, vocab)[0]
    q = torch.softmax(q_logits, -1)
    p = torch.softmax(p_logits, -1)

    off = torch.ones(vocab, dtype=torch.bool)
    off[ids] = False
    assert (q[off] == 0).all()  # proposal never reaches out-of-set ids
    torch.testing.assert_close(_rejection_output(p, q), p)
    # The sampler's log-space residual equals max(p - q, 0) / Z.
    residual = torch.clamp(p - q, min=0)
    torch.testing.assert_close(
        _kernel_residual(p_logits, q_logits), residual / residual.sum()
    )
    # Target already truncated (top-k) where the draft is also -inf: no NaN.
    p_masked = p_logits.clone()
    p_masked[off.nonzero()[:4, 0]] = float("-inf")
    assert not torch.isnan(_kernel_residual(p_masked, q_logits)).any()


def test_gumbel_draft_sampling_stays_in_set():
    torch.manual_seed(0)
    ids = torch.tensor([3, 17, 40])
    full = dv.scatter_draft_logits(torch.randn(4096, 3), ids, 64)
    gumbel = -torch.log(-torch.log(torch.rand(4096, 64)))
    picked = (full / 0.7 + gumbel).argmax(-1)
    assert torch.isin(picked, ids).all()
