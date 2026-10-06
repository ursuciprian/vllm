# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the MTP refit capture hook on a mock input batch."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open

from vllm.v1.worker.gpu import mtp_capture
from vllm.v1.worker.gpu.mtp_capture import MtpCapture, select_rows

H, V, K = 16, 50, 5


def _batch(chunks, tokens):
    """chunks: (req_id, req_idx, num_computed, num_scheduled, prefill_len)."""
    qsl = np.cumsum([0] + [c[3] for c in chunks])
    pos = np.concatenate([np.arange(c[2], c[2] + c[3]) for c in chunks])
    ids = np.concatenate([tokens[c[1]][c[2] : c[2] + c[3]] for c in chunks])
    return SimpleNamespace(
        num_reqs=len(chunks),
        req_ids=[c[0] for c in chunks],
        idx_mapping_np=np.array([c[1] for c in chunks]),
        query_start_loc_np=qsl,
        num_computed_tokens_np=np.array([c[2] for c in chunks]),
        prefill_len_np=np.array([c[4] for c in chunks]),
        is_prefilling_np=np.array([c[2] < c[4] for c in chunks]),
        positions=torch.from_numpy(pos),
        input_ids=torch.from_numpy(ids).to(torch.int32),
    )


def _read(path):
    with safe_open(path, "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, f.metadata()  # noqa: SIM118


def test_select_rows_tail_and_decode():
    b = _batch(
        [("a", 0, 0, 8, 10), ("b", 1, 10, 1, 10)], {0: np.arange(10), 1: np.arange(11)}
    )
    b.is_prefilling_np[1] = False  # decode row
    assert select_rows(b, tail=5) == [(0, 5, 8, False)]
    assert select_rows(b, tail=100) == [(0, 0, 8, False)]
    assert select_rows(b, tail=2) == []


def test_two_chunks_stitch_shapes_topk_and_sha1(tmp_path, monkeypatch):
    torch.manual_seed(0)
    plen = {0: 13, 1: 4}
    tokens = {r: np.random.randint(0, V, size=n) for r, n in plen.items()}
    all_ids = torch.zeros(2, 32, dtype=torch.int32)
    for r, t in tokens.items():
        all_ids[r, : len(t)] = torch.from_numpy(t)
    w = torch.randn(V, H)
    head = lambda h: h.float() @ w.T  # noqa: E731
    cap = MtpCapture(str(tmp_path), topk=K, tail=10)
    monkeypatch.setattr(mtp_capture, "LOGITS_CHUNK_ROWS", 3)  # several head chunks
    seen = {}
    for chunks in (
        [("a", 0, 0, 8, 13), ("b", 1, 0, 4, 4)],  # a: positions 3..7 stored
        [("a", 0, 8, 5, 13)],  # a: last chunk 8..12
    ):
        b = _batch(chunks, tokens)
        n = int(b.query_start_loc_np[-1])
        multi = torch.randn(n + 2, 4 * H, dtype=torch.bfloat16)  # +2 padded rows
        sample = torch.randn(n + 2, H, dtype=torch.bfloat16)
        cap.on_step(b, multi, sample, head, all_ids)
        for i, c in enumerate(chunks):
            r0 = int(b.query_start_loc_np[i])
            for p in range(c[2], c[2] + c[3]):
                seen[(c[0], p)] = (multi[r0 + p - c[2]], sample[r0 + p - c[2]])
    cap.close()
    (shard,) = sorted(tmp_path.glob("shard-*.safetensors"))
    t, meta = _read(shard)
    reqs = json.loads(meta["requests"])
    assert [r["id"] for r in reqs] == ["a", "b"]
    for r, rid in ((0, "a"), (1, "b")):
        want = hashlib.sha1(tokens[r].astype("<i4").tobytes()).hexdigest()
        assert reqs[r]["sha1"] == want and reqs[r]["prefill_len"] == plen[r]
    n = t["tokens"].shape[0]
    assert n == 10 + 4
    assert t["hidden"].shape == (n, 4 * H) and t["hidden"].dtype == torch.bfloat16
    assert t["topk_ids"].shape == (n, K) and t["topk_ids"].dtype == torch.int32
    assert t["topk_logprobs"].dtype == torch.float16
    for rid, r in (("a", 0), ("b", 1)):
        rows = (t["req"] == [x["id"] for x in reqs].index(rid)).nonzero().flatten()
        pos = t["positions"][rows]
        assert pos.tolist() == list(range(plen[r] - min(10, plen[r]), plen[r]))
        assert t["tokens"][rows].tolist() == tokens[r][pos.numpy()].tolist()
        for row, p in zip(rows.tolist(), pos.tolist()):
            m, s = seen[(rid, p)]
            assert torch.equal(t["hidden"][row], m)
            ref_lp, ref_ids = torch.log_softmax(head(s[None]), -1).topk(K)
            assert t["topk_ids"][row].tolist() == ref_ids[0].tolist()
            torch.testing.assert_close(
                t["topk_logprobs"][row].float(), ref_lp[0], atol=2e-3, rtol=1e-3
            )


def test_shards_roll_and_split_request(tmp_path, monkeypatch):
    monkeypatch.setattr(mtp_capture, "SHARD_ROWS", 4)
    tokens = {0: np.arange(6)}
    ids = torch.from_numpy(np.arange(6)).to(torch.int32)[None]
    head = lambda h: h.float()  # noqa: E731
    cap = MtpCapture(str(tmp_path), topk=2, tail=100)
    for chunk in ((0, 5), (5, 1)):
        b = _batch([("x", 0, chunk[0], chunk[1], 6)], tokens)
        n = chunk[1]
        cap.on_step(
            b, torch.zeros(n, 8, dtype=torch.bfloat16), torch.randn(n, 3), head, ids
        )
    cap.close()
    shards = sorted(tmp_path.glob("shard-*.safetensors"))
    assert [p.name for p in shards] == [
        "shard-00000.safetensors",
        "shard-00001.safetensors",
    ]
    (t0, m0), (t1, m1) = _read(shards[0]), _read(shards[1])
    assert t0["positions"].tolist() == [0, 1, 2, 3, 4] and t1["positions"].tolist() == [
        5
    ]
    assert json.loads(m0["requests"])[0]["sha1"] is None
    assert json.loads(m1["requests"])[0]["sha1"] is not None
    # A restart in the same directory appends instead of overwriting.
    assert MtpCapture(str(tmp_path))._shard == 2


def test_knobs_declared_and_outside_compile_key(monkeypatch):
    pytest.importorskip("zmq")  # compile_factors imports vllm.config
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_MTP_CAPTURE_DIR", "/tmp/x")
    names = ("VLLM_MTP_CAPTURE_DIR", "VLLM_MTP_CAPTURE_TOPK", "VLLM_MTP_CAPTURE_TAIL")
    assert all(n in envs.environment_variables for n in names)
    assert not set(names) & set(envs.compile_factors())
