# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP refit capture: store what the MTP layer reads, plus the target's top-k.

Off unless ``VLLM_MTP_CAPTURE_DIR`` is set. For every prefill row whose position
is in the stored tail (``position >= prefill_len - VLLM_MTP_CAPTURE_TAIL``) it
keeps the target's multi-stream state (the MTP ``hidden_states`` input), the
token id, the position and the top-k log-probs of the target's full-vocab
softmax at T=1, computed with the target's own head. Rows are written as
``shard-%05d.safetensors`` every ``SHARD_ROWS`` rows (and on idle / shutdown):

    hidden         [N, hc*hidden] bf16  target multi_hidden at position t
    tokens         [N]            int32 x_t
    positions      [N]            int32 t
    topk_ids       [N, K]         int32 ids of the distribution of x_{t+1}
    topk_logprobs  [N, K]         fp16  their log-probs
    req            [N]            int32 index into metadata ``requests``

``__metadata__["requests"]`` is a JSON list of ``{"id", "prefill_len",
"sha1", "drafts"?}``; sha1 (of the request's full token list as little-endian
int32) and drafts (``on_drafts``) are set in the shard holding the request's
last prefill chunk. Chunked prefill
splits a request across steps and shards; readers stitch by id and position.

Rows still buffered are written 30 s after the last step and at shutdown:
wait more than 35 s after the last request before stopping the server.

Host-side only: an eager head call, a top-k and a device-to-host copy after
the target forward. It declares no b12x plan and changes no graph, so its
knobs stay out of ``envs.compile_factors()``.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

SHARD_ROWS = 65536
LOGITS_CHUNK_ROWS = 512
IDLE_FLUSH_S = 30.0


def select_rows(batch: Any, tail: int) -> list[tuple[int, int, int, bool]]:
    """Per batch row of a prefilling request: (batch_idx, first stored
    position, end position, last prefill chunk?). Positions only; the token
    row of position p is ``query_start_loc[i] + p - num_computed_tokens[i]``.
    """
    out = []
    for i in range(batch.num_reqs):
        if not batch.is_prefilling_np[i]:
            continue
        p0 = int(batch.num_computed_tokens_np[i])
        n = int(batch.query_start_loc_np[i + 1] - batch.query_start_loc_np[i])
        plen = int(batch.prefill_len_np[i])
        end = min(p0 + n, plen)
        lo = max(p0, plen - tail)
        if lo < end:
            out.append((i, lo, end, end == plen))
    return out


DRAFT_TOPK = 20


class MtpCapture:
    def __init__(self, out_dir: str, topk: int = 20, tail: int = 6144) -> None:
        if topk < 1 or tail < 1:
            raise ValueError("VLLM_MTP_CAPTURE_TOPK and _TAIL must be >= 1")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.topk = topk
        self.tail = tail
        self._lock = threading.Lock()
        self._rows: dict[str, list[torch.Tensor]] = {}
        self._num_rows = 0
        self._reqs: dict[str, dict[str, Any]] = {}
        self._shard = 1 + max(
            (
                int(f[6:11])
                for f in os.listdir(out_dir)
                if f.startswith("shard-") and f.endswith(".safetensors")
            ),
            default=-1,
        )
        self._last_step = time.monotonic()
        self._finals: list[tuple[int, str]] = []
        self._closed = False
        threading.Thread(target=self._idle_flush, daemon=True).start()
        atexit.register(self.close)

    @classmethod
    def from_config(cls, vllm_config: Any) -> MtpCapture:
        import vllm.envs as envs

        parallel = vllm_config.parallel_config
        spec = vllm_config.speculative_config
        if (
            parallel.tensor_parallel_size
            * parallel.pipeline_parallel_size
            * parallel.data_parallel_size
            != 1
        ):
            raise ValueError("VLLM_MTP_CAPTURE_DIR supports TP=1, PP=1, DP=1 only")
        if spec is None or spec.method != "mtp":
            raise ValueError("VLLM_MTP_CAPTURE_DIR needs MTP speculative decoding")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError(
                "VLLM_MTP_CAPTURE_DIR needs --no-enable-prefix-caching: "
                "a cached prefix is never computed, so it is never captured"
            )
        return cls(
            envs.VLLM_MTP_CAPTURE_DIR,
            topk=envs.VLLM_MTP_CAPTURE_TOPK,
            tail=envs.VLLM_MTP_CAPTURE_TAIL,
        )

    @torch.inference_mode()
    def on_step(
        self,
        batch: Any,
        multi_hidden: torch.Tensor,
        sample_hidden: torch.Tensor,
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
        all_token_ids: torch.Tensor,
    ) -> None:
        self._last_step = time.monotonic()
        self._finals = []
        if self._num_rows >= SHARD_ROWS:
            self.flush()
        sel = select_rows(batch, self.tail)
        if not sel:
            return
        qsl = batch.query_start_loc_np
        rows = np.concatenate(
            [
                np.arange(lo, end) + (qsl[i] - batch.num_computed_tokens_np[i])
                for i, lo, end, _ in sel
            ]
        )
        expected_pos = np.concatenate([np.arange(lo, end) for _, lo, end, _ in sel])
        idx = torch.from_numpy(rows.astype(np.int64)).to(multi_hidden.device)
        positions = batch.positions.index_select(0, idx).to(torch.int32).cpu()
        if not np.array_equal(positions.numpy(), expected_pos):
            raise RuntimeError("MTP capture: batch positions disagree with row map")
        sample = sample_hidden.index_select(0, idx)
        ids, lps = [], []
        for c in range(0, sample.shape[0], LOGITS_CHUNK_ROWS):
            logits = compute_logits(sample[c : c + LOGITS_CHUNK_ROWS])
            lp, top = torch.log_softmax(logits.float(), dim=-1).topk(self.topk, dim=-1)
            ids.append(top.to(torch.int32))
            lps.append(lp.to(torch.float16))
        part = {
            "hidden": multi_hidden.index_select(0, idx).cpu(),
            "tokens": batch.input_ids.index_select(0, idx).to(torch.int32).cpu(),
            "positions": positions,
            "topk_ids": torch.cat(ids).cpu(),
            "topk_logprobs": torch.cat(lps).cpu(),
        }
        finals = {}
        for i, _, _, last in sel:
            if last:
                plen = int(batch.prefill_len_np[i])
                req_idx = int(batch.idx_mapping_np[i])
                toks = all_token_ids[req_idx, :plen].to(torch.int32).cpu().numpy()
                finals[batch.req_ids[i]] = hashlib.sha1(
                    toks.astype("<i4").tobytes()
                ).hexdigest()
        with self._lock:
            req_col = []
            for i, lo, end, _ in sel:
                rid = batch.req_ids[i]
                meta = self._reqs.setdefault(
                    rid,
                    {
                        "id": rid,
                        "prefill_len": int(batch.prefill_len_np[i]),
                        "sha1": None,
                        "_k": len(self._reqs),
                    },
                )
                meta["sha1"] = finals.get(rid, meta["sha1"])
                req_col.append(torch.full((end - lo,), meta["_k"], dtype=torch.int32))
            part["req"] = torch.cat(req_col)
            for k, v in part.items():
                self._rows.setdefault(k, []).append(v)
            self._num_rows += int(rows.shape[0])
        self._finals = [(i, batch.req_ids[i]) for i, _, _, last in sel if last]

    @torch.inference_mode()
    def on_drafts(
        self,
        batch: Any,
        sampled: torch.Tensor,
        drafts: torch.Tensor,
        draft_logits: torch.Tensor | None = None,
    ) -> None:
        """For requests whose prefill ended this step, keep [sampled token, draft
        tokens...] as ``drafts`` in their metadata: the vLLM drafter's own chain
        from the last prefill row (greedy at T=0), the reference for the refit's
        GPU parity check. With ``draft_logits`` (the probabilistic drafter's
        pre-temperature cache, [max_num_reqs, steps, vocab], row = the request's
        state index) also keep ``draft_topk``: per draft step the top
        ``DRAFT_TOPK`` [ids, logits] the served drafter produced."""
        if self._finals:
            first = sampled[: batch.num_reqs, 0].cpu().tolist()
            chain = drafts[: batch.num_reqs].cpu().tolist()
            topk = None
            if draft_logits is not None:
                n = drafts.shape[1]
                rows = batch.idx_mapping[[i for i, _ in self._finals]].long()
                v, ids = draft_logits[rows, :n].float().topk(DRAFT_TOPK, dim=-1)
                topk = {i: [ids[j].tolist(), v[j].tolist()] for j, (i, _) in enumerate(self._finals)}
            with self._lock:
                for i, rid in self._finals:
                    if rid in self._reqs:
                        self._reqs[rid]["drafts"] = [first[i], *chain[i]]
                        if topk is not None:
                            self._reqs[rid]["draft_topk"] = topk[i]
            self._finals = []
        if self._num_rows >= SHARD_ROWS:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._num_rows:
                return
            rows, reqs, shard = self._rows, self._reqs, self._shard
            self._rows, self._reqs, self._num_rows = {}, {}, 0
            self._shard += 1
        from safetensors.torch import save_file

        tensors = {k: torch.cat(v).contiguous() for k, v in rows.items()}
        requests = sorted(reqs.values(), key=lambda m: m["_k"])
        meta = {
            "format": "mtp-capture-v1",
            "topk": str(self.topk),
            "tail": str(self.tail),
            "requests": json.dumps(
                [{k: v for k, v in m.items() if k != "_k"} for m in requests]
            ),
        }
        path = os.path.join(self.out_dir, f"shard-{shard:05d}.safetensors")
        try:
            save_file(tensors, path + ".tmp", metadata=meta)
            os.replace(path + ".tmp", path)
        except Exception:
            logger.exception(
                "MTP capture: shard %s lost (%d rows, %d requests)",
                path,
                tensors["tokens"].shape[0],
                len(requests),
            )
            raise

    def _idle_flush(self) -> None:
        while not self._closed:
            time.sleep(5.0)
            if time.monotonic() - self._last_step > IDLE_FLUSH_S:
                with contextlib.suppress(Exception):  # logged in flush
                    self.flush()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.flush()
