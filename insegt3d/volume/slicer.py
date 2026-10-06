import cv2
import time
import queue
import shutil
import concurrent.futures
import numpy as np
import tensorstore as ts
from pathlib import Path

from insegt3d.volume.io import read_multiscale_zarr, read_multiscale_masks, make_ts_context, remap_labels
from insegt3d.volume.interp import write_nearest, read_nearest, read_trilinear
from insegt3d.volume.intensity import coarsest_small_level, robust_percentile_range

_sample_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def _bounding_box(center, axes, half_extents):
    extent = sum(np.abs(axis) * half for axis, half in zip(axes, half_extents))
    return np.floor(center - extent).astype(np.int64), np.ceil(center + extent).astype(np.int64) + 1

def _fitting_level(voxel_sizes, zoom):
    """Coarsest level whose voxels are no larger than `zoom` on every axis. Level 0 always qualifies."""
    fits = np.all(voxel_sizes <= np.maximum(zoom, voxel_sizes[0]), axis=1)
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

    def read_masks(self):
        return read_multiscale_masks(self.mask_path, [image.shape for image in self.images], self.scales, self.translations, ts_context=self.ts_context)

    def sources(self, levels, mask=False):
        volumes = self.masks if mask else self.images
        return [(level, volumes[level], self.voxel_sizes[level], self.offsets[level]) for level in levels]

    @property
    def fallback(self):
        """Small level held in memory, drawn first while finer levels load."""
        if self._fallback is None:
            level = coarsest_small_level(self.shapes, max_voxels=256 ** 3)
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
        volume = np.asarray(self.images[coarsest_small_level(self.shapes, max_voxels=max_dim ** 3)])
        value_range = float(volume.min()), float(volume.max())
        counts, _ = np.histogram(volume, bins=bins, range=value_range)
        return counts, value_range, robust_percentile_range(volume)

    @staticmethod
    def apply_patches(patches, undo):
        for vol, box, index, before, after in (reversed(patches) if undo else patches):
            sub = np.ascontiguousarray(vol[box])
            sub.flat[index] = before if undo else after
            vol[box] = sub

    def level(self, zoom, level_modifier=0):
        return _fitting_level(self.voxel_sizes, zoom * 2.0 ** level_modifier)

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
        level = self.level(zoom) if level is None else level

        D = max(1, int(np.ceil(extent[1] - extent[0])))
        D, H, W = out_shape if len(out_shape) == 3 else (D, *out_shape)
        output = np.zeros((D, H, W), dtype=dtype)

        if (mask and self.masks is None) or (prediction and self.predictions is None):
            return output[0] if D == 1 else output

        if prediction:
            volumes, voxel_sizes, offsets = self.predictions
            level = _fitting_level(voxel_sizes, zoom)
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
    ):
        """
        Generator that fills `output` from `sources` tile by tile, yielding as reads complete.
        Later sources take priority, and each source's tiles are read centre-first. Stops after
        `deadline`. Reads that are superseded or unfinished are cancelled below `keep_level`.
        """
        zoom = zoom if zoom is not None else camera.zoom

        d0, d1, top, bottom, left, right = (float(value) * zoom for value in extent)
        D, H, W = output.shape

        dd = 0.0 if D == 1 else (d1 - d0) / (D - 1)
        dy = 0.0 if H == 1 else (bottom - top) / (H - 1)
        dx = 0.0 if W == 1 else (right - left) / (W - 1)

        th = tw = max(8, int(tile_hw))

        tiles = []

        for rank, (level, volume, voxel_size, offset) in enumerate(sources):

            axes = tuple((axis / voxel_size).astype(np.float32) for axis in camera.uvw)
            origin = ((camera.origin - offset) / voxel_size).astype(np.float32)
            pad = float(voxel_size.max()) if order == 1 else 0.0

            Is, Js, Ks = volume.shape

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

                    tiles.append(((rank, distance), (y0, x0), level, volume[i0:i1, j0:j1, k0:k1], out_tile, local_origin, padding, grid))

        tiles.sort(key=lambda tile: tile[0])

        done = queue.Queue()
        futures = {}
        drawn = {}
        issued = 0

        def draw(item):
            _, i, subvol = item
            _, _, _, _, out_tile, local_origin, padding, grid = tiles[i]

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
            while futures or issued < len(tiles):

                while issued < len(tiles) and len(futures) < max_inflight:
                    region = tiles[issued][3]
                    if isinstance(region, np.ndarray):
                        futures[issued] = region
                        done.put(issued)
                    else:
                        futures[issued] = region.read()
                        futures[issued].add_done_callback(lambda _, i=issued: done.put(i))
                    issued += 1

                late = deadline is not None and time.time() >= deadline
                wait = 0.1 if deadline is None else max(0.0, min(0.1, deadline - time.time()))

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
                    (rank, _), key = tiles[i][0], tiles[i][1]
                    subvol = futures.pop(i, None)
                    if subvol is not None and rank > max(drawn.get(key, -1), best.get(key, (-1,))[0]):
                        best[key] = (rank, i, subvol)

                for key, (rank, _, _) in best.items():
                    drawn[key] = rank

                list(_sample_pool.map(draw, sorted(best.values(), key=lambda b: tiles[b[1]][0][1])))

                # Drop reads that can no longer improve their tile, cancelling those below keep_level
                for i in [i for i in futures if tiles[i][0][0] <= drawn.get(tiles[i][1], -1)]:
                    future = futures.pop(i)
                    if isinstance(future, ts.Future) and (keep_level is None or tiles[i][2] < keep_level):
                        future.cancel()

                yield

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
