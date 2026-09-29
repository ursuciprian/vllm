# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the page-cache PLE table (VLLM_PLE_MMAP).

Loads vllm/models/qwen3_8_flash_next/ple_mmap.py by path (numpy only), so it
runs without a vLLM build: python -m pytest <this file>.
"""

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

_SRC = (
    Path(__file__).resolve().parents[2] / "vllm/models/qwen3_8_flash_next/ple_mmap.py"
)
_spec = importlib.util.spec_from_file_location("ple_mmap", _SRC)
ple_mmap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ple_mmap)

TABLE_ROWS, SHARD_ROWS, ROW_BYTES = 1000, 128, (80, 10)
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def _write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        data = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": "U8",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def _file_sources(path):
    """(name -> absolute payload offset), as the checkpoint loader reports it."""
    with open(path, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
    return {k: 8 + size + v["data_offsets"][0] for k, v in header.items()}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """An in-memory table split into shard planes over three files, with
    unrelated tensors between them so row offsets are unaligned."""
    rng = np.random.default_rng(0)
    planes = [rng.integers(0, 256, (TABLE_ROWS, b), dtype=np.uint8) for b in ROW_BYTES]
    root = tmp_path_factory.mktemp("ckpt")
    shard_count = -(-TABLE_ROWS // SHARD_ROWS)
    sources = []
    for file in range(3):
        tensors = {f"filler_{file}": rng.integers(0, 256, 37 + file, dtype=np.uint8)}
        for shard in range(file, shard_count, 3):
            rows = slice(shard * SHARD_ROWS, (shard + 1) * SHARD_ROWS)
            tensors[f"{PREFIX}.shard_{shard}.weight"] = planes[0][rows]
            tensors[f"{PREFIX}.shard_{shard}.weight_scale"] = planes[1][rows]
            tensors[f"filler_{file}_{shard}"] = rng.integers(0, 256, 3, np.uint8)
        path = root / f"model-{file:05d}.safetensors"
        _write_safetensors(path, tensors)
        for name, offset in _file_sources(path).items():
            if name.startswith(PREFIX):
                shard, suffix = name[len(PREFIX) + 7 :].split(".")
                sources.append((suffix == "weight_scale", int(shard), path, offset))
    return planes, sources


def _rows(checkpoint, shard_start=0, shard_end=TABLE_ROWS, workers=4):
    planes, sources = checkpoint
    rows = ple_mmap.PageCacheRows(
        table_rows=TABLE_ROWS,
        shard_rows=SHARD_ROWS,
        shard_start=shard_start,
        shard_end=shard_end,
        row_bytes=ROW_BYTES,
        workers=workers,
    )
    registered = {
        (plane, shard)
        for plane, shard, path, offset in sources
        if rows.add_source(plane, shard, str(path), offset)
    }
    rows.freeze()
    return rows, registered


def _expected(planes, ids, lo, hi):
    local = (ids >= lo) & (ids < hi)
    return [
        np.where(local[:, None], p[np.clip(ids, 0, TABLE_ROWS - 1)], 0) for p in planes
    ]


@pytest.mark.parametrize("count", [1, 80, 5000])  # decode, c1 MTP x4, prefill
def test_gather_matches_in_memory_table(checkpoint, count):
    planes, _ = checkpoint
    rows, _ = _rows(checkpoint)
    rng = np.random.default_rng(count)
    ids = rng.integers(-3, TABLE_ROWS + 3, count, dtype=np.int64)
    ids[:3] = [0, TABLE_ROWS - 1, SHARD_ROWS][:count]
    outs = tuple(np.full((count + 7, b), 0xAB, np.uint8) for b in ROW_BYTES)
    rows.gather(ids, outs)
    for out, want in zip(outs, _expected(planes, ids, 0, TABLE_ROWS)):
        np.testing.assert_array_equal(out[:count], want)
        assert (out[count:] == 0xAB).all()  # positional: nothing past count


def test_tp_rank_reads_only_its_rows(checkpoint):
    planes, _ = checkpoint
    lo, hi = 256, 640  # rank 1 of a TP split
    rows, registered = _rows(checkpoint, lo, hi)
    assert {shard for _, shard in registered} == {2, 3, 4}
    ids = np.arange(-2, TABLE_ROWS + 2, dtype=np.int64)
    outs = tuple(np.empty((ids.size, b), np.uint8) for b in ROW_BYTES)
    rows.gather(ids, outs)
    for out, want in zip(outs, _expected(planes, ids, lo, hi)):
        np.testing.assert_array_equal(out, want)


def test_missing_local_shard_fails_at_freeze(checkpoint):
    _, sources = checkpoint
    rows = ple_mmap.PageCacheRows(
        table_rows=TABLE_ROWS,
        shard_rows=SHARD_ROWS,
        shard_start=0,
        shard_end=TABLE_ROWS,
        row_bytes=ROW_BYTES,
    )
    for plane, shard, path, offset in sources:
        if (plane, shard) != (1, 5):
            rows.add_source(plane, shard, str(path), offset)
    with pytest.raises(ValueError, match="plane 1 shard 5"):
        rows.freeze()


def test_source_past_end_of_file_is_rejected(checkpoint):
    _, sources = checkpoint
    _, _, path, _ = sources[0]
    rows = ple_mmap.PageCacheRows(
        table_rows=TABLE_ROWS,
        shard_rows=SHARD_ROWS,
        shard_start=0,
        shard_end=TABLE_ROWS,
        row_bytes=ROW_BYTES,
    )
    with pytest.raises(ValueError, match="exceeds"):
        rows.add_source(0, 0, str(path), path.stat().st_size - 10)
