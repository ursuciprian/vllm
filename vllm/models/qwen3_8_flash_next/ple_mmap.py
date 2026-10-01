# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve the Qwen3.8-Flash-Next PLE table through the page cache.

Idea from dime-online/qwen3.8-Flash-DGX-UltraFast (Apache-2.0,
recipe/build/image/src/vllm_ple_mmap.py): on a single GB10 the table does not
fit next to the other weights, but a token reads only 16 of its rows. Their
patch mmaps an FP8 table and gathers on the CPU. Here the table stays in the
checkpoint's safetensors shards and b12x's disk-table contract is kept (GPU
hash, positional per-lookup row cache in mapped host memory, compact-row
decode); only the O_DIRECT io_uring reader is replaced by buffered reads, so
hot rows stay in the evictable page cache instead of coming from NVMe every
step.
"""

from __future__ import annotations

import functools
import logging
import math
import mmap
import os
import re
import resource
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.lib.stride_tricks import as_strided

logger = logging.getLogger("vllm.ple_mmap")

# Batches above this many lookups (prefill) are split across threads so
# page-cache misses are in flight concurrently; decode batches stay serial.
_PARALLEL_LOOKUPS = 2048
_CHUNK = 1024


def _env_int(name: str, default: int) -> int:
    # Reader knobs are read from os.environ, not vllm/envs.py, on purpose: they
    # change no b12x plan or graph, so they must stay out of the compile key.
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _host_counters() -> dict[str, int]:
    """Process page faults, host-wide NVMe reads completed and page cache."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    out = dict(majflt=usage.ru_majflt, minflt=usage.ru_minflt, nvme_reads=0,
               mem_available=0, cached=0)
    try:
        with open("/proc/diskstats") as fh:
            for line in fh:
                fields = line.split()
                if re.fullmatch(r"nvme\d+n\d+", fields[2]):
                    out["nvme_reads"] += int(fields[3])
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, value = line.split(":", 1)
                if key in ("MemAvailable", "Cached"):
                    name = "mem_available" if key == "MemAvailable" else "cached"
                    out[name] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return out


class ReaderStats:
    """VLLM_PLE_MMAP_STATS=N: log the reader's cost every N table reads.

    ``wait`` is the CPU blocked on the GPU before the gather (ids ready);
    ``gather`` is the host row copy itself, which the next graph waits for.
    """

    def __init__(self, every: int) -> None:
        self.every = every
        self._host = _host_counters()
        self._reset()

    def _reset(self) -> None:
        self.reads = self.lookups = self.unique = 0
        self.wait_s = self.gather_s = self.max_gather_s = 0.0
        self.t0 = time.monotonic()

    def add(self, ids: np.ndarray, wait_s: float, gather_s: float) -> None:
        self.reads += 1
        self.lookups += ids.size
        self.unique += np.unique(ids).size
        self.wait_s += wait_s
        self.gather_s += gather_s
        self.max_gather_s = max(self.max_gather_s, gather_s)
        if self.reads >= self.every:
            self.flush()

    def flush(self) -> None:
        if not self.reads:
            return
        host, n = _host_counters(), self.reads
        elapsed = max(time.monotonic() - self.t0, 1e-9)
        delta = {k: host[k] - self._host[k] for k in ("majflt", "minflt", "nvme_reads")}
        logger.info(
            "PLE reader: %d reads, %.1f lookups/read (%.1f unique), sync wait "
            "%.3f ms/read, gather %.3f ms/read (max %.3f), majflt %.2f/read, "
            "minflt %.2f/read, nvme %.0f reads/s, MemAvailable %.2f GiB, "
            "Cached %.2f GiB",
            n, self.lookups / n, self.unique / n, 1e3 * self.wait_s / n,
            1e3 * self.gather_s / n, 1e3 * self.max_gather_s,
            delta["majflt"] / n, delta["minflt"] / n,
            delta["nvme_reads"] / elapsed, host["mem_available"] / 2**30,
            host["cached"] / 2**30,
        )
        self._host = host
        self._reset()


class PageCacheRows:
    """Positional row gather over checkpoint row planes.

    Plane 0 holds weight rows, plane 1 (optional) scale rows. Shard ``s`` of a
    plane covers table rows ``[s * shard_rows, s * shard_rows + rows)`` stored
    contiguously at a byte offset of one file. ``gather`` writes the row of
    lookup ``i`` into row ``i`` of each output and zeros lookups outside this
    rank's ``[shard_start, shard_end)``, like b12x's O_DIRECT reader.
    """

    def __init__(
        self,
        *,
        table_rows: int,
        shard_rows: int,
        shard_start: int,
        shard_end: int,
        row_bytes: tuple[int, ...],
        workers: int = 32,
    ) -> None:
        if not 0 <= shard_start <= shard_end <= table_rows or shard_rows <= 0:
            raise ValueError("invalid PLE table geometry")
        self.table_rows = table_rows
        self.shard_rows = shard_rows
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.row_bytes = row_bytes
        self.shard_count = math.ceil(table_rows / shard_rows)
        self.workers = workers
        self._sources: dict[tuple[int, int], tuple[str, int]] = {}
        self._frozen = False
        self._pool: ThreadPoolExecutor | None = None

    def shard_row_count(self, shard: int) -> int:
        return min(self.shard_rows, self.table_rows - shard * self.shard_rows)

    def add_source(self, plane: int, shard: int, path: str, offset: int) -> bool:
        """Register one shard plane; returns False when it is not local."""
        if self._frozen:
            raise RuntimeError("cannot change PLE sources after freeze")
        if not 0 <= plane < len(self.row_bytes):
            raise ValueError("PLE table has no such row plane")
        if not 0 <= shard < self.shard_count or offset < 0:
            raise ValueError("PLE source shard or offset is out of range")
        first = shard * self.shard_rows
        if first >= self.shard_end or first + self.shard_rows <= self.shard_start:
            return False
        if (plane, shard) in self._sources:
            raise ValueError("PLE source shard/plane is already registered")
        end = offset + self.shard_row_count(shard) * self.row_bytes[plane]
        if end > os.stat(path).st_size:
            raise ValueError(f"PLE source range exceeds {path}")
        self._sources[(plane, shard)] = (os.fspath(path), offset)
        return True

    def freeze(self) -> None:
        if self._frozen:
            return
        first = self.shard_start // self.shard_rows
        last = math.ceil(self.shard_end / self.shard_rows)
        paths = sorted({path for path, _ in self._sources.values()})
        file_index = {path: i for i, path in enumerate(paths)}
        maps = []
        for path in paths:
            mapped = np.memmap(path, dtype=np.uint8, mode="r")
            # Rows are scattered: fault in one page per miss, no readahead.
            mapped._mmap.madvise(mmap.MADV_RANDOM)
            maps.append(mapped)
        self._file_of, self._base_of, self._windows = [], [], []
        for plane, row_bytes in enumerate(self.row_bytes):
            file_of = np.full(self.shard_count, -1, dtype=np.int64)
            base_of = np.zeros(self.shard_count, dtype=np.int64)
            for shard in range(first, last):
                source = self._sources.get((plane, shard))
                if source is None:
                    raise ValueError(f"missing PLE plane {plane} shard {shard}")
                file_of[shard] = file_index[source[0]]
                base_of[shard] = source[1]
            self._file_of.append(file_of)
            self._base_of.append(base_of)
            # Row k of a window is file bytes [k, k + row_bytes): one index
            # gathers rows at arbitrary byte offsets without copying the file.
            self._windows.append(
                [
                    as_strided(
                        m,
                        shape=(m.size - row_bytes + 1, row_bytes),
                        strides=(1, 1),
                        writeable=False,
                    )
                    for m in maps
                ]
            )
        self._maps = maps
        self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="ple-mmap")
        self._frozen = True

    def gather(self, ids: np.ndarray, outs: tuple[np.ndarray, ...]) -> None:
        if not self._frozen:
            raise RuntimeError("PLE rows are not frozen")
        count = ids.shape[0]
        local = (ids >= self.shard_start) & (ids < self.shard_end)
        valid = np.flatnonzero(local)
        valid_ids = ids[valid]
        shard = valid_ids // self.shard_rows
        row = valid_ids - shard * self.shard_rows
        jobs = []
        for plane, out in enumerate(outs):
            out[:count][~local] = 0
            row_bytes = self.row_bytes[plane]
            file_of = self._file_of[plane][shard]
            offsets = self._base_of[plane][shard] + row * row_bytes
            for file in np.unique(file_of):
                pick = file_of == file
                window = self._windows[plane][file]
                targets, sources = valid[pick], offsets[pick]
                for start in range(0, targets.size, _CHUNK):
                    jobs.append(
                        (
                            out,
                            window,
                            targets[start : start + _CHUNK],
                            sources[start : start + _CHUNK],
                        )
                    )
        if count > _PARALLEL_LOOKUPS and len(jobs) > 1:
            list(self._pool.map(lambda job: _copy_rows(*job), jobs))
        else:
            for job in jobs:
                _copy_rows(*job)


def _copy_rows(out, window, targets, sources) -> None:
    # Fancy indexing copies row by row without the GIL, so page faults overlap
    # across threads. (take() would first copy the strided window whole.)
    out[targets] = window[sources]


@functools.cache
def _page_cache_disk_table_cls():
    import threading

    import torch
    from b12x.sequence._shared.disk_table import DiskRowCache, MappedHostAllocation
    from b12x.sequence.ple_embedding._disk import DiskTable

    class PageCacheRowCache(DiskRowCache):
        """b12x DiskRowCache with the O_DIRECT reader swapped for PageCacheRows."""

        def __init__(self, layout, shard_rows: int) -> None:  # noqa: D107
            # The base constructor builds the native io_uring reader; set up the
            # same staging and transaction state without it.
            caps = layout.caps
            device = torch.device(caps.device)
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            self.device = device
            self.max_lookups = caps.max_tokens * layout.head_count
            self.table_rows = layout.padded_vocab_size
            self.shard_start = layout.shard_start
            self.shard_end = layout.shard_end
            self.shard_rows = shard_rows
            self.weight_row_bytes = (
                layout.weight_shape[1] * layout.weight_dtype.itemsize
            )
            self.scale_row_bytes = (
                layout.head_dim // 16 if caps.quant_mode == "nvfp4_group16" else 0
            )
            self.shard_count = math.ceil(self.table_rows / shard_rows)
            self._backend = "page_cache"
            self._gds = self._native = self._reader = None
            self.ids_host = torch.empty(
                (self.max_lookups,), dtype=torch.int64, device="cpu", pin_memory=True
            )
            self._ids_buffer = memoryview(self.ids_host.numpy())
            self._weight_allocation = MappedHostAllocation(
                (self.max_lookups, self.weight_row_bytes), torch.uint8, device
            )
            self.weight = self._weight_allocation.device_view
            self.weight_host = self._weight_allocation.host_view
            self._weight_buffer = None
            self._scale_allocation = self.scale = self.scale_host = None
            self._scale_buffer = None
            planes = [self.weight_host.numpy()]
            if self.scale_row_bytes:
                self._scale_allocation = MappedHostAllocation(
                    (self.max_lookups, self.scale_row_bytes), torch.uint8, device
                )
                self.scale = self._scale_allocation.device_view
                self.scale_host = self._scale_allocation.host_view
                planes.append(self.scale_host.numpy())
            self._planes = tuple(planes)
            self._rows = PageCacheRows(
                table_rows=self.table_rows,
                shard_rows=shard_rows,
                shard_start=self.shard_start,
                shard_end=self.shard_end,
                row_bytes=tuple(p.shape[1] for p in planes),
                workers=min(32, os.cpu_count() or 1),
            )
            every = _env_int("VLLM_PLE_MMAP_STATS", 0)
            self._stats = ReaderStats(every) if every > 0 else None
            self._sources = set()
            self._frozen = False
            self._lock = threading.RLock()
            self._transaction_thread = None
            self._transaction_stream = None
            self._ids_ready = torch.cuda.Event()
            self._cache_done = torch.cuda.Event()
            self._cache_used = False
            self._closed = False

        def add_shard(self, shard_index, path, offset, *, scale=False) -> None:
            self._require_open()
            with self._lock:
                if self._frozen:
                    raise RuntimeError("cannot change disk shards after binding")
                if scale and not self.scale_row_bytes:
                    raise ValueError("disk table has no row scale plane")
                if self._rows.add_source(int(scale), shard_index, path, offset):
                    self._sources.add((scale, shard_index))

        def freeze(self) -> None:
            with self._lock:
                super().freeze()
                self._rows.freeze()

        def _read_staged(self, count: int) -> None:
            with torch.cuda.device(self.device):
                t0 = time.perf_counter()
                self._ids_ready.synchronize()
                t1 = time.perf_counter()
                ids = self.ids_host.numpy()[:count]
                self._rows.gather(ids, self._planes)
                if self._stats is not None:
                    self._stats.add(ids, t1 - t0, time.perf_counter() - t1)

        def stats(self) -> dict[str, int | float]:
            return {"cache_bytes": sum(p.nbytes for p in self._planes)}

    class PageCacheDiskTable(DiskTable):
        def __init__(self, layout, shard_rows: int) -> None:  # noqa: D107
            self.layout = layout
            self._cache = PageCacheRowCache(layout, shard_rows)
            self.weight = self._cache.weight.view(layout.weight_dtype)
            self.weight_host = self._cache.weight_host.view(layout.weight_dtype)
            scale, scale_host = self._cache.scale, self._cache.scale_host
            self.weight_scale = (
                scale.view(torch.float8_e4m3fn) if scale is not None else None
            )
            self.weight_scale_host = (
                scale_host.view(torch.float8_e4m3fn) if scale_host is not None else None
            )

    return PageCacheDiskTable


def make_page_cache_disk_table(layout, shard_rows: int):
    """Build a b12x ``DiskTable`` whose rows come from the page cache."""
    return _page_cache_disk_table_cls()(layout, shard_rows)
