import cv2
import time
import heapq
import queue
import shutil
import itertools
import threading
import collections
import concurrent.futures
import numpy as np
import tensorstore as ts
from pathlib import Path

from insegt3d.volume.io import read_multiscale_zarr, read_multiscale_masks, make_ts_context, remap_labels, is_http_url, HTTP_REQUEST_CONCURRENCY
from insegt3d.volume.interp import write_nearest, read_nearest, read_trilinear
from insegt3d.volume.intensity import coarsest_small_level, robust_percentile_range

_sample_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

class _RequestBudget:
    """
    Remote chunk requests that may be outstanding at once, shared by every slicer. Cancelling a tensorstore read leaves
    its HTTP requests queued, so requests only start within this limit, from queues that a newer view simply abandons.
    """

    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def acquire(self, share=1.0):
        """A request may start while less than `share` of the limit is in use."""
        with self._lock:
            if self.used >= self.limit * share:
                return False
            self.used += 1
            return True

    def release(self):
        with self._lock:
            self.used -= 1

# Every request starts at once, so none waits in tensorstore's queue where it could not be cancelled
_remote_requests = _RequestBudget(HTTP_REQUEST_CONCURRENCY)

# Chunks already read from each remote level, which tensorstore now serves from its cache without a request
_fetched_chunks = {}

# Each finer level waits this many slice pixels behind the next coarser one, so a finer level's centre is requested
# before a coarser level's edges
_LEVEL_PRIORITY_PIXELS = 128

def _remote_chunks(volume):
    """Chunk shape and already read chunks of a volume read over HTTP, or None for in-memory and local volumes."""
    if isinstance(volume, np.ndarray) or volume.kvstore is None:
        return None
    spec = volume.kvstore.spec().to_json()
    if spec['driver'] != 'http':
        return None
    return np.array(volume.chunk_layout.read_chunk.shape), _fetched_chunks.setdefault(spec['base_url'] + spec['path'], set())

def _chunk_indices(lo, hi, chunk):
    first, last = np.asarray(lo) // chunk, (np.asarray(hi) - 1) // chunk
    return list(itertools.product(*(range(a, b + 1) for a, b in zip(first.tolist(), last.tolist()))))

_RemoteLevel = collections.namedtuple('_RemoteLevel', 'volume chunk fetched rank voxel_size offset')

class _ChunkRequests:
    """
    The remote chunks one stream still needs, requested nearest the view centre first while the shared budget allows.
    A remote tile is read, from tensorstore's cache, once all its chunks have arrived, so the view refines chunk by chunk.
    """

    def __init__(self, camera, zoom, wake):
        self._origin, self._v, self._w, self._zoom = camera.origin, camera.v, camera.w, zoom
        self._wake = wake
        self._levels = []
        self._heap = []
        self._waiting = {}  # (level, chunk key) -> tiles that still need the chunk
        self._missing = {}  # tile -> number of its chunks that have not arrived
        self._arrived = queue.SimpleQueue()
        self._in_flight = 0
        self.ready = collections.deque()

    @property
    def pending(self):
        return bool(self._heap or self._in_flight or self.ready)

    def add_level(self, volume, chunk, fetched, rank, voxel_size, offset):
        self._levels.append(_RemoteLevel(volume, chunk, fetched, rank, voxel_size, offset))
        return len(self._levels) - 1

    def add_tile(self, tile, level, keys):
        missing = [key for key in keys if key not in self._levels[level].fetched]
        if not missing:
            self.ready.append(tile)
            return
        self._missing[tile] = len(missing)
        for key in missing:
            waiting = self._waiting.setdefault((level, key), [])
            if not waiting:
                heapq.heappush(self._heap, (self._priority(level, key), level, key))
            waiting.append(tile)

    def request(self, share, superseded):
        """Requests the most urgent chunks the budget allows, skipping ones read meanwhile or needed only by superseded tiles."""
        while self._heap:
            _, level, key = self._heap[0]
            tiles = self._waiting.get((level, key))
            if tiles is None or key in self._levels[level].fetched:
                heapq.heappop(self._heap)
                self._arrive(level, key)
            elif all(superseded(tile) for tile in tiles):
                heapq.heappop(self._heap)
                for tile in self._waiting.pop((level, key)):
                    self._missing.pop(tile, None)
            elif _remote_requests.acquire(share):
                heapq.heappop(self._heap)
                self._read(level, key)
            else:
                return

    def collect(self):
        """Moves the tiles whose last chunk has arrived to `ready`."""
        while not self._arrived.empty():
            self._in_flight -= 1
            self._arrive(*self._arrived.get())

    def _priority(self, level, key):
        """Slice pixels from the view centre to the chunk's nearest point, plus `_LEVEL_PRIORITY_PIXELS` per source rank."""
        remote = self._levels[level]
        lo = remote.offset + np.asarray(key) * remote.chunk * remote.voxel_size
        nearest = np.clip(self._origin, lo, lo + remote.chunk * remote.voxel_size) - self._origin
        return float(np.hypot(nearest @ self._v, nearest @ self._w)) / self._zoom + remote.rank * _LEVEL_PRIORITY_PIXELS

    def _read(self, level, key):
        remote = self._levels[level]
        lo = np.asarray(key) * remote.chunk
        hi = np.minimum(lo + remote.chunk, remote.volume.shape)
        self._in_flight += 1
        remote.volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].read().add_done_callback(lambda future: self._on_read(level, key, future))

    def _on_read(self, level, key, future):
        # Runs on a tensorstore thread, so only shared state is touched here and the stream does the bookkeeping
        _remote_requests.release()
        if not future.cancelled() and future.exception() is None:
            self._levels[level].fetched.add(key)
        self._arrived.put((level, key))
        self._wake()

    def _arrive(self, level, key):
        for tile in self._waiting.pop((level, key), ()):
            if tile in self._missing:
                self._missing[tile] -= 1
                if not self._missing[tile]:
                    del self._missing[tile]
                    self.ready.append(tile)

def _bounding_box(center, axes, half_extents):
    extent = sum(np.abs(axis) * half for axis, half in zip(axes, half_extents))
    return np.floor(center - extent).astype(np.int64), np.ceil(center + extent).astype(np.int64) + 1

def _fitting_level(voxel_sizes, zoom, normal):
    """
    Coarsest level whose voxels are no larger than `zoom` on every axis in the slice plane. Level 0 always qualifies.
    The axis the slice faces is left out, so a level downsampled less along it is not passed over.
    """
    # An axis counts unless the slice is within about 2.5 degrees of facing it
    in_plane = np.abs(normal) < 0.999
    fits = np.all((voxel_sizes <= np.maximum(zoom, voxel_sizes[0]))[:, in_plane], axis=1)
    return int(np.flatnonzero(fits)[-1])

class VolumeSlicer:

    def __init__(self, cache_size_mb=16384, ts_context=None):

        self.zarr_path = None
        self.mask_path = None

        self.images = None
        self.masks = None
        self.predictions = None
        self.shapes = None
        self.scales = None
        self.translations = None
        self.voxel_sizes = None
        self.offsets = None

        self._fallback = None

        self.ts_context = ts_context if ts_context is not None else make_ts_context(cache_size_mb)

    def initialize(self, zarr_path, mask_path, camera, center_camera=False, prediction_path=None):

        self.zarr_path = zarr_path
        self.mask_path = mask_path

        self.images, self.scales, self.translations = read_multiscale_zarr(
            self.zarr_path,
            ts_context=self.ts_context
        )
        self.voxel_sizes, self.offsets = self._to_world(self.scales, self.translations)

        self.shapes = np.array([image.shape for image in self.images], dtype=int)

        # Masks are created on the first annotation (see set_data)
        if mask_path is not None and Path(mask_path).exists():
            self.masks = self.read_masks()
        else:
            self.masks = None

        self.predictions = self._load_predictions(prediction_path)

        if center_camera:
            camera.reset(self.world_shape)

        self._fallback = None

    def _to_world(self, scales, translations):
        """Scales and translations in world units (level 0's smallest voxel side), with level 0 at the origin."""
        unit = self.scales[0].min()
        return scales / unit, (translations - self.translations[0]) / unit

    @property
    def world_shape(self):
        return self.shapes[0] * self.voxel_sizes[0]

    @property
    def remote(self):
        return is_http_url(self.zarr_path)

    def read_masks(self):
        return read_multiscale_masks(self.mask_path, [image.shape for image in self.images], self.scales, self.translations, ts_context=self.ts_context)

    def sources(self, levels, mask=False):
        volumes = self.masks if mask else self.images
        return [(level, volumes[level], self.voxel_sizes[level], self.offsets[level]) for level in levels]

    @property
    def fallback(self):
        """Small level held in memory, drawn first while finer levels load."""
        if self._fallback is None:
            # Remote volumes use the smaller level that intensity_stats has already read, so it costs no requests
            max_dim = 128 if self.remote else 256
            level = coarsest_small_level(self.shapes, max_voxels=max_dim ** 3)
            self._fallback = (level, np.asarray(self.images[level]), self.voxel_sizes[level], self.offsets[level])
        return self._fallback

    def reset_masks(self, masks_root):
        shutil.rmtree(masks_root, ignore_errors=True)
        # A fresh context drops cached chunks of the deleted masks
        self.ts_context = ts.Context(self.ts_context.spec)
        self.masks = None

    def relabel_masks(self, masks_root, lut):
        remap_labels(masks_root, lut)
        self.ts_context = ts.Context(self.ts_context.spec)
        if self.masks is not None:
            self.masks = self.read_masks()

    @property
    def has_prediction(self):
        return self.predictions is not None

    def refresh_prediction(self, prediction_path):
        self.predictions = self._load_predictions(prediction_path)

    def _load_predictions(self, prediction_path):
        if prediction_path is None or not Path(prediction_path).exists():
            return None

        predictions, scales, translations = read_multiscale_zarr(str(prediction_path), ts_context=self.ts_context, recheck="open")
        return (predictions, *self._to_world(scales, translations))

    def intensity_stats(self, max_dim=128, bins=256):
        """
        Histogram, value range and display window of the whole volume at low resolution together with a central crop at
        full resolution. Coarse levels are smoothed by averaging and some pyramids are scaled differently from level 0,
        so the window spans both.
        """
        size = np.minimum((max_dim // 4, max_dim, max_dim), self.shapes[0])
        z, y, x = (self.shapes[0] - size) // 2
        dz, dy, dx = size

        # Both reads run at once, which matters for remote volumes
        overview = self.images[coarsest_small_level(self.shapes, max_voxels=max_dim ** 3)].read()
        detail = self.images[0][z:z + dz, y:y + dy, x:x + dx].read()
        overview, detail = overview.result(), detail.result()

        value_range = float(min(overview.min(), detail.min())), float(max(overview.max(), detail.max()))
        counts = np.histogram(overview, bins=bins, range=value_range)[0] + np.histogram(detail, bins=bins, range=value_range)[0]
        low, high = robust_percentile_range(overview)
        detail_low, detail_high = robust_percentile_range(detail)
        return counts, value_range, (min(low, detail_low), max(high, detail_high))

    @staticmethod
    def apply_patches(patches, undo):
        for vol, box, index, before, after in (reversed(patches) if undo else patches):
            sub = np.ascontiguousarray(vol[box])
            sub.flat[index] = before if undo else after
            vol[box] = sub

    def level(self, camera, zoom=None, level_modifier=0):
        zoom = camera.zoom if zoom is None else zoom
        return _fitting_level(self.voxel_sizes, zoom * 2.0 ** level_modifier, camera.u)

    def cached_level(self, camera, levels, extent):
        """Finest of the remote `levels` whose chunks in view have all been read, so it draws without requests, or None."""
        d0, d1, top, bottom, left, right = (abs(float(value)) * camera.zoom for value in extent)
        for level in sorted(levels):
            chunk_shape, fetched = _remote_chunks(self.images[level])
            voxel_size, offset = self.voxel_sizes[level], self.offsets[level]
            # Padded by a voxel like the tiles of an interpolated stream
            pad = float(voxel_size.max())
            axes = tuple(axis / voxel_size for axis in camera.uvw)
            lo, hi = _bounding_box((camera.origin - offset) / voxel_size, axes, (max(d0, d1) + pad, max(top, bottom) + pad, max(left, right) + pad))
            if fetched.issuperset(_chunk_indices(np.maximum(lo, 0), np.minimum(hi, self.shapes[level]), chunk_shape)):
                return level
        return None

    def get_data(
        self,
        camera,
        extent,
        out_shape,
        mask=False,
        prediction=False,
        order=0,
        zoom_override=None,
        level=None,
    ):
        """Samples `extent` (d0, d1, top, bottom, left, right in slice pixels) around the camera, as (D, H, W) or (H, W) if D == 1."""

        dtype = np.uint8 if mask or prediction else np.float32
        if self.images is None:
            return np.zeros(out_shape[-2:], dtype=dtype)

        zoom = zoom_override if zoom_override is not None else camera.zoom
        level = self.level(camera, zoom) if level is None else level

        D = max(1, int(np.ceil(extent[1] - extent[0])))
        D, H, W = out_shape if len(out_shape) == 3 else (D, *out_shape)
        output = np.zeros((D, H, W), dtype=dtype)

        if (mask and self.masks is None) or (prediction and self.predictions is None):
            return output[0] if D == 1 else output

        if prediction:
            volumes, voxel_sizes, offsets = self.predictions
            level = _fitting_level(voxel_sizes, zoom, camera.u)
            sources = [(level, volumes[level], voxel_sizes[level], offsets[level])]
        else:
            sources = self.sources([level], mask=mask)

        for _ in self.stream(camera, output, sources, extent, zoom=zoom, order=order):
            pass

        return output[0] if D == 1 else output

    def stream(
        self,
        camera,
        output,
        sources,
        extent,
        zoom=None,
        keep_level=None,
        deadline=None,
        order=0,
        tile_hw=128,
        max_inflight=256,
        budget_share=1.0,
    ):
        """
        Generator that fills `output` from `sources` tile by tile, yielding as reads complete.
        Later sources take priority, and each source's tiles are read centre-first. Stops after
        `deadline`. Local reads that are superseded or unfinished are cancelled below `keep_level`.
        Remote tiles wait for their chunks (see _ChunkRequests), which only start while less than
        `budget_share` of the shared request budget is in use. Yields whether the remote tiles that
        were already cached at the start are all drawn.
        """
        zoom = zoom if zoom is not None else camera.zoom

        d0, d1, top, bottom, left, right = (float(value) * zoom for value in extent)
        D, H, W = output.shape

        dd = 0.0 if D == 1 else (d1 - d0) / (D - 1)
        dy = 0.0 if H == 1 else (bottom - top) / (H - 1)
        dx = 0.0 if W == 1 else (right - left) / (W - 1)

        th = tw = max(8, int(tile_hw))

        tiles = []
        cached = {}  # position -> rank of the finest remote tile there whose chunks are all cached
        done = queue.Queue()
        chunks = _ChunkRequests(camera, zoom, wake=lambda: done.put(None))

        for rank, (level, volume, voxel_size, offset) in enumerate(sources):

            axes = tuple((axis / voxel_size).astype(np.float32) for axis in camera.uvw)
            origin = ((camera.origin - offset) / voxel_size).astype(np.float32)
            pad = float(voxel_size.max()) if order == 1 else 0.0

            Is, Js, Ks = volume.shape
            remote = _remote_chunks(volume)
            if remote is not None:
                chunk_shape, fetched = remote
                chunk_level = chunks.add_level(volume, chunk_shape, fetched, rank, voxel_size, offset)

            d_lo, d_hi = (d0, d0) if pad == 0 and D == 1 else (d0 - pad, d0 + (D - 1) * dd + pad)

            for y0 in range(0, H, th):
                h_tile = min(th, H - y0)
                y_start = top + y0 * dy
                y_end   = y_start + (h_tile - 1) * dy

                for x0 in range(0, W, tw):
                    w_tile = min(tw, W - x0)
                    x_start = left + x0 * dx
                    x_end   = x_start + (w_tile - 1) * dx

                    center = origin + (d_lo + d_hi) / 2 * axes[0] + (y_start + y_end) / 2 * axes[1] + (x_start + x_end) / 2 * axes[2]
                    half_extents = (abs(d_hi - d_lo) / 2, abs(y_end - y_start) / 2 + pad, abs(x_end - x_start) / 2 + pad)
                    mn, mx = _bounding_box(center, axes, half_extents)

                    i0_raw, j0_raw, k0_raw = mn
                    i1_raw, j1_raw, k1_raw = mx

                    if i1_raw <= 0 or i0_raw >= Is or j1_raw <= 0 or j0_raw >= Js or k1_raw <= 0 or k0_raw >= Ks:
                        continue

                    i0, i1 = max(0, int(i0_raw)), min(Is, int(i1_raw))
                    j0, j1 = max(0, int(j0_raw)), min(Js, int(j1_raw))
                    k0, k1 = max(0, int(k0_raw)), min(Ks, int(k1_raw))

                    # Tiles crossing the volume edge are read clipped and zero-padded back to full size
                    if (i0, i1, j0, j1, k0, k1) == (i0_raw, i1_raw, j0_raw, j1_raw, k0_raw, k1_raw):
                        padding = None
                    else:
                        padding = ((int(i1_raw - i0_raw), int(j1_raw - j0_raw), int(k1_raw - k0_raw)), i0 - int(i0_raw), j0 - int(j0_raw), k0 - int(k0_raw))

                    local_origin = origin - np.array([i0_raw, j0_raw, k0_raw], dtype=np.float32)
                    out_tile = output[:, y0:y0 + h_tile, x0:x0 + w_tile]
                    distance = (y0 + h_tile / 2 - H / 2) ** 2 + (x0 + w_tile / 2 - W / 2) ** 2
                    grid = axes + (d0, dd, y_start, dy, x_start, dx)
                    chunk_keys = None if remote is None else (chunk_level, _chunk_indices((i0, j0, k0), (i1, j1, k1), chunk_shape))
                    if chunk_keys is not None and fetched.issuperset(chunk_keys[1]):
                        cached[(y0, x0)] = rank

                    tiles.append(((rank, distance), (y0, x0), level, volume[i0:i1, j0:j1, k0:k1], out_tile, local_origin, padding, grid, chunk_keys))

        tiles.sort(key=lambda tile: tile[0])

        local = collections.deque()
        for i, tile in enumerate(tiles):
            (rank, _), position, chunk_keys = tile[0], tile[1], tile[8]
            if chunk_keys is None:
                local.append(i)
            # Remote tiles under a finer cached one are neither read nor requested
            elif rank >= cached.get(position, -1):
                chunks.add_tile(i, *chunk_keys)

        undrawn_cached = set(chunks.ready)
        futures = {}
        drawn = {}

        def superseded(i):
            return tiles[i][0][0] <= drawn.get(tiles[i][1], -1)

        def issue(i):
            region = tiles[i][3]
            if isinstance(region, np.ndarray):
                futures[i] = region
                done.put(i)
            else:
                futures[i] = region.read()
                futures[i].add_done_callback(lambda _, i=i: done.put(i))

        def draw(item):
            _, i, subvol = item
            _, _, _, _, out_tile, local_origin, padding, grid, _ = tiles[i]

            if not isinstance(subvol, np.ndarray):
                subvol = subvol.result()
            subvol = np.ascontiguousarray(subvol, dtype=output.dtype)

            if padding is not None:
                shape, oi, oj, ok = padding
                buf = np.zeros(shape, dtype=output.dtype)
                buf[oi:oi + subvol.shape[0], oj:oj + subvol.shape[1], ok:ok + subvol.shape[2]] = subvol
                subvol = buf

            sampler = read_nearest if order == 0 else read_trilinear
            sampler(subvol, out_tile, local_origin, *grid)

        try:
            while futures or local or chunks.pending:

                chunks.collect()
                chunks.request(budget_share, superseded)
                while (chunks.ready or local) and len(futures) < max_inflight:
                    # Local and in-memory tiles first, so the fallback is always drawn before any remote level
                    issue(local.popleft() if local else chunks.ready.popleft())

                late = deadline is not None and time.time() >= deadline
                wait = 0.1 if deadline is None else max(0.0, min(0.1, deadline - time.time()))

                # None only wakes the loop for an arrived chunk
                ready = []
                try:
                    ready.append(done.get(timeout=wait))
                except queue.Empty:
                    pass
                while not done.empty():
                    ready.append(done.get_nowait())

                # Per output tile, draw only the highest-priority read that improves on what is drawn
                best = {}
                for i in ready:
                    if i is None:
                        continue
                    undrawn_cached.discard(i)
                    (rank, _), key = tiles[i][0], tiles[i][1]
                    subvol = futures.pop(i, None)
                    if subvol is not None and rank > max(drawn.get(key, -1), best.get(key, (-1,))[0]):
                        best[key] = (rank, i, subvol)

                for key, (rank, _, _) in best.items():
                    drawn[key] = rank

                list(_sample_pool.map(draw, sorted(best.values(), key=lambda b: tiles[b[1]][0][1])))

                # Drop reads that can no longer improve their tile, cancelling those below keep_level
                for i in [i for i in futures if superseded(i)]:
                    future = futures.pop(i)
                    undrawn_cached.discard(i)
                    if isinstance(future, ts.Future) and (keep_level is None or tiles[i][2] < keep_level):
                        future.cancel()

                yield not undrawn_cached

                if late:
                    return
        finally:
            for i, future in futures.items():
                if isinstance(future, ts.Future) and (keep_level is None or tiles[i][2] < keep_level):
                    future.cancel()

    def set_data(self, camera, data, extent, tile_hw=64, thickness=2):
        """Writes the 2D `data` into every mask level and returns the changed voxels as undo patches."""

        if self.images is None:
            return []

        if self.masks is None:
            self.masks = self.read_masks()

        _, _, top, bottom, left, right = map(float, extent)
        H, W = max(1, int(np.ceil(bottom - top))), max(1, int(np.ceil(right - left)))

        zoom = float(camera.zoom)
        top, left = top * zoom, left * zoom
        dy = 0.0 if H == 1 else (bottom * zoom - top) / (H - 1)
        dx = 0.0 if W == 1 else (right * zoom - left) / (W - 1)

        mask = cv2.resize(np.asarray(data, dtype=np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        data0 = np.repeat(mask[None], thickness, axis=0)

        step = max(1, tile_hw - 1)
        tiles = [
            (y0, x0, min(tile_hw, H - y0), min(tile_hw, W - x0))
            for y0 in range(0, H, step) for x0 in range(0, W, step)
            if mask[y0:y0 + tile_hw, x0:x0 + tile_hw].any()
        ]

        def write_level(level):
            voxel_size, offset = self.voxel_sizes[level], self.offsets[level]
            vol = self.masks[level]
            shape = np.array(vol.shape, dtype=np.int64)
            shard = vol.chunk_layout.write_chunk.shape[-3:]
            axes = tuple((axis / voxel_size).astype(np.float32) for axis in camera.uvw)
            origin = ((camera.origin - offset) / voxel_size).astype(np.float32)
            dd = np.linalg.norm(voxel_size * camera.u) / np.linalg.norm(camera.u)
            grid = tuple((axis * voxel_size).astype(np.float32) for axis in camera.uvw) + (-0.5 * (thickness - 1) * dd, dd, top, dy, left, dx)

            blocks = {}

            for y0, x0, h_tile, w_tile in tiles:
                y_start = top + y0 * dy
                y_end   = y_start + (h_tile - 1) * dy
                x_start = left + x0 * dx
                x_end   = x_start + (w_tile - 1) * dx

                center = origin + (y_start + y_end) / 2 * axes[1] + (x_start + x_end) / 2 * axes[2]
                half_extents = ((thickness - 1) / 2 * dd, abs(y_end - y_start) / 2, abs(x_end - x_start) / 2)
                mn, mx = _bounding_box(center, axes, half_extents)
                mn, mx = np.maximum(mn, 0), np.minimum(mx, shape)

                if np.any(mx <= mn):
                    continue

                # Merge tiles starting in the same shard into one read-modify-write box
                key = tuple(mn // shard)
                block = blocks.get(key)
                blocks[key] = (mn, mx) if block is None else (np.minimum(block[0], mn), np.maximum(block[1], mx))

            patches = []

            for mn, mx in blocks.values():
                box = tuple(slice(int(i), int(j)) for i, j in zip(mn, mx))

                sub = np.ascontiguousarray(vol[box])
                before = sub.copy()

                write_nearest(sub, data0, (origin - mn).astype(np.float32), *grid)

                changed = np.flatnonzero(sub != before)
                if changed.size:
                    vol[box] = sub
                    patches.append((vol, box, changed, before.flat[changed], sub.flat[changed]))

            return patches

        return [patch for level in range(len(self.masks)) for patch in write_level(level)]
