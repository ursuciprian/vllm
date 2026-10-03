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


def _rows(checkpoint, shard_start=0, shard_end=TABLE_ROWS, workers=4, **knobs):
    planes, sources = checkpoint
    rows = ple_mmap.PageCacheRows(
        table_rows=TABLE_ROWS,
        shard_rows=SHARD_ROWS,
        shard_start=shard_start,
        shard_end=shard_end,
        row_bytes=ROW_BYTES,
        workers=workers,
        **knobs,
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


def test_keepalive_reads_only_while_gathers_are_recent(checkpoint, monkeypatch):
    import time

    monkeypatch.setattr(ple_mmap, "_KEEPALIVE_IDLE_S", 0.3)
    rows, _ = _rows(checkpoint, keepalive_ms=10)
    time.sleep(0.15)
    assert rows.keepalive_reads == 0  # no gather yet: the drive may sleep
    outs = tuple(np.empty((8, b), np.uint8) for b in ROW_BYTES)
    rows.gather(np.arange(8, dtype=np.int64), outs)
    time.sleep(0.15)
    assert rows.keepalive_reads > 0
    time.sleep(0.4)  # past the idle window: the reads stop
    idle = rows.keepalive_reads
    time.sleep(0.15)
    assert rows.keepalive_reads == idle


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


def test_row_cache_host_buffers_ignore_default_device(monkeypatch):
    """Model init runs under a CUDA default device; every host staging buffer
    must name its device, or pinning fails at boot (593c8b42f). A meta default
    device stands in for CUDA: an allocation that relies on it lands on meta."""
    import sys
    import types

    torch = pytest.importorskip("torch")

    def host_tensor(shape, dtype, device):
        return torch.zeros(shape, dtype=dtype, device="cpu")

    class MappedHostAllocation:
        def __init__(self, shape, dtype, device):
            self.device_view = host_tensor(shape, dtype, device)
            self.host_view = host_tensor(shape, dtype, device)

    class DiskRowCache:
        pass

    class DiskTable:
        pass

    fakes = {
        "b12x": types.ModuleType("b12x"),
        "b12x.sequence": types.ModuleType("b12x.sequence"),
        "b12x.sequence._shared": types.ModuleType("b12x.sequence._shared"),
        "b12x.sequence._shared.disk_table": types.SimpleNamespace(
            DiskRowCache=DiskRowCache, MappedHostAllocation=MappedHostAllocation
        ),
        "b12x.sequence.ple_embedding": types.ModuleType("b12x.sequence.ple_embedding"),
        "b12x.sequence.ple_embedding._disk": types.SimpleNamespace(DiskTable=DiskTable),
    }
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)

    class NoPinning(torch.overrides.TorchFunctionMode):
        # Without CUDA there is nothing to pin for; the device is what counts.
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            kwargs.pop("pin_memory", None)
            return func(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    layout = types.SimpleNamespace(
        caps=types.SimpleNamespace(
            device="cuda:0", max_tokens=4, quant_mode="nvfp4_group16"
        ),
        head_count=16,
        padded_vocab_size=TABLE_ROWS,
        shard_start=0,
        shard_end=TABLE_ROWS,
        weight_shape=(TABLE_ROWS, 80),
        weight_dtype=torch.uint8,
        head_dim=160,
    )
    ple_mmap._page_cache_disk_table_cls.cache_clear()
    try:
        with torch.device("meta"), NoPinning():
            table = ple_mmap.make_page_cache_disk_table(layout, SHARD_ROWS)
    finally:
        ple_mmap._page_cache_disk_table_cls.cache_clear()
    cache = table._cache
    for tensor in (cache.ids_host, cache.weight_host, cache.scale_host):
        assert tensor.device.type == "cpu"
    assert [p.shape for p in cache._planes] == [(64, 80), (64, 10)]


def test_reader_stats_log_every_n_reads(caplog):
    stats = ple_mmap.ReaderStats(every=3)
    with caplog.at_level("INFO", logger="vllm.ple_mmap"):
        for _ in range(7):
            stats.add(np.array([1, 1, 2, -1], np.int64), 0.001, 0.002)
    lines = [r.getMessage() for r in caplog.records if "PLE reader" in r.getMessage()]
    assert len(lines) == 2 and stats.reads == 1
    assert "3 reads, 4.0 lookups/read (3.0 unique)" in lines[0]
    assert "gather 2.000 ms/read" in lines[0]


@pytest.mark.parametrize(
    "knobs",
    [
        dict(parallel_lookups=32, chunk=8),  # r8 tp1-ple-par: decode on the pool
        dict(parallel_lookups=0, chunk=1, workers=8),
        dict(madvise="normal"),
    ],
)
@pytest.mark.parametrize("count", [80, 5000])
def test_gather_knobs_keep_rows_identical(checkpoint, knobs, count):
    planes, _ = checkpoint
    rows, _ = _rows(checkpoint, **knobs)
    ids = np.random.default_rng(7).integers(-3, TABLE_ROWS + 3, count, dtype=np.int64)
    outs = tuple(np.full((count, b), 0xAB, np.uint8) for b in ROW_BYTES)
    rows.gather(ids, outs)
    for out, want in zip(outs, _expected(planes, ids, 0, TABLE_ROWS)):
        np.testing.assert_array_equal(out, want)


def test_willneed_pass_names_every_row(checkpoint, monkeypatch):
    """The WILLNEED pass must advise exactly the bytes the gather then copies."""
    import os

    calls = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *a: calls.append(a), raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_WILLNEED", 3, raising=False)
    planes, _ = checkpoint
    rows, _ = _rows(checkpoint, willneed_max=128)
    ids = np.array([5, -1, 999, 300, 5, TABLE_ROWS], np.int64)
    outs = tuple(np.zeros((ids.size, b), np.uint8) for b in ROW_BYTES)
    rows.gather(ids, outs)
    assert len(calls) == 2 * 4  # 4 local lookups x 2 planes, none for -1 / out of range
    got = [os.pread(fd, length, offset) for fd, offset, length, advice in calls]
    want = [p[i].tobytes() for p in planes for i in (5, 999, 300, 5)]
    assert got == want and all(c[3] == 3 for c in calls)
    for out, exp in zip(outs, _expected(planes, ids, 0, TABLE_ROWS)):
        np.testing.assert_array_equal(out, exp)
    calls.clear()
    big = np.arange(129, dtype=np.int64)  # above willneed_max: no pass
    rows.gather(big, tuple(np.zeros((129, b), np.uint8) for b in ROW_BYTES))
    assert not calls


def test_prewarm_reads_each_local_plane_once(checkpoint, caplog):
    import threading

    with caplog.at_level("INFO", logger="vllm.ple_mmap"):
        _rows(checkpoint, 256, 640, prewarm="all")
        for t in threading.enumerate():
            if t.name == "ple-prewarm":
                t.join(10)
    msg = next(
        r.getMessage() for r in caplog.records if "PLE prewarm" in r.getMessage()
    )
    local_bytes = 3 * SHARD_ROWS * sum(ROW_BYTES)  # shards 2, 3, 4
    assert f"({local_bytes} bytes)" in msg and "planes [1, 0]" in msg


def test_bad_reader_knobs_are_rejected():
    with pytest.raises(ValueError):
        ple_mmap.PageCacheRows(
            table_rows=10,
            shard_rows=5,
            shard_start=0,
            shard_end=10,
            row_bytes=(8,),
            prewarm="yes",
        )
    with pytest.raises(ValueError):
        ple_mmap.PageCacheRows(
            table_rows=10,
            shard_rows=5,
            shard_start=0,
            shard_end=10,
            row_bytes=(8,),
            chunk=0,
        )


def _b12x_hash_reference():
    """b12x's exact PyTorch hash oracle: the installed package, or a checkout
    named by B12X_SRC (it only needs torch)."""
    import os

    pytest.importorskip("torch")
    try:
        from b12x.sequence.ple_hash import reference

        return reference
    except ImportError:
        src = os.environ.get("B12X_SRC")
        if not src:
            pytest.skip("needs b12x or B12X_SRC=<b12x checkout>")
        path = Path(src) / "b12x/sequence/ple_hash/reference.py"
        spec = importlib.util.spec_from_file_location("ple_hash_reference", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def test_cpu_hash_matches_b12x_hash_bit_for_bit():
    """r8 tp1-ple-cpuhash: the NumPy hash must give the GPU kernel's ids
    exactly; b12x's reference is the oracle its kernel tests compare to."""
    import torch

    ref = _b12x_hash_reference()
    vocab, eos, order, per_order = 248320, 248044, 3, 8
    multipliers = ref.ple_multipliers(
        vocab_size=vocab, max_order=order, dense_layer_ordinal=0
    )
    sizes, offsets = ref.ple_table_geometry(
        base_size=20_000_000, dense_layer_ordinal=0, total_heads=16
    )
    rng = np.random.default_rng(11)
    for case in range(300):
        num_seqs = int(rng.integers(1, 6))
        lens = rng.integers(0, 7, num_seqs)
        lens[-int(rng.integers(0, 2)) or num_seqs :] = 0  # padded requests
        qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        num_tokens = int(qsl[-1])
        window = num_tokens + int(rng.integers(0, 4))  # graph-padded launch
        tokens = rng.integers(0, vocab, window, dtype=np.int64)
        history = rng.integers(0, vocab, (num_seqs + 2, order - 1), dtype=np.int64)
        for arr in (tokens, history.reshape(-1)):  # EOS boundaries
            arr[rng.random(arr.shape) < 0.15] = eos
        got = ple_mmap.ple_hash_ids(
            tokens,
            np.pad(qsl, (0, 3)),
            history,
            num_seqs,
            num_tokens,
            eos_token_id=eos,
            multipliers=multipliers.numpy(),
            prime_sizes=sizes.numpy(),
            table_offsets=offsets.numpy(),
            heads_per_order=per_order,
        )
        want = ref.ple_hash_packed_reference(
            torch.from_numpy(tokens[:num_tokens]),
            torch.from_numpy(qsl),
            torch.from_numpy(history[:num_seqs]),
            eos_token_id=eos,
            multipliers=multipliers,
            prime_sizes=sizes,
            table_offsets=offsets,
            heads_per_order=per_order,
        ).numpy()
        np.testing.assert_array_equal(got[:num_tokens], want, err_msg=f"case {case}")
        assert (got[num_tokens:] == -1).all()
