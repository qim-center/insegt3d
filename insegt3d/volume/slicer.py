import cv2
import numpy as np
import threading

from insegt3d.volume.io import read_multiscale_zarr
from insegt3d.volume.interp import write_nearest, read_nearest, read_trilinear

class VolumeSlicer(object):

    def __init__(self, max_undo=50, cache_size_mb=30000):

        self.zarr_path = None

        self.images = None
        self.masks = None
        self.shapes = None

        self._level = None

        # TensorStore cache settings
        self.ts_context = {
            'cache_pool': {'total_bytes_limit': int(cache_size_mb * 1024**2)}
        }

        # Undo/redo (stores sparse volume patches)
        self._undo_stack = []
        self._redo_stack = []
        self._max_undo = max_undo
        self._edit_lock = threading.Lock()

    def initialize(self, zarr_path, mask_path, camera, center_camera=False):

        self.zarr_path = zarr_path

        # Reset undo/redo history
        self._undo_stack.clear()
        self._redo_stack.clear()

        # Read multiscale volume
        self.images, self.masks = read_multiscale_zarr(
            self.zarr_path,
            mask_path=mask_path,
            ts_context=self.ts_context
        )

        # Get shapes
        self.shapes = np.array([image.shape for image in self.images], dtype=int)

        # Center camera
        if center_camera:
            camera.reset(self.shapes[0])

        self._level = 0

    def _add_to_history(self, edit):

        if not edit:
            return

        self._undo_stack.append(edit)

        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

        self._redo_stack.clear()

    def _apply_edit(self, edit, which):

        patches = edit[::-1] if which == "before" else edit

        for level, is_mask, (i0, i1, j0, j1, k0, k1), before, after in patches:
            vol = self.masks[level] if is_mask else self.images[level]
            vol[i0:i1, j0:j1, k0:k1] = (before if which == "before" else after)

    def undo(self):
        """
        Undo the most recent set_data() call.
        """
        with self._edit_lock:
            if not self._undo_stack:
                return False
            edit = self._undo_stack.pop()
            self._apply_edit(edit, "before")
            self._redo_stack.append(edit)
            return True

    def redo(self):
        """
        Redo the most recently undone set_data() call.
        """
        with self._edit_lock:
            if not self._redo_stack:
                return False
            edit = self._redo_stack.pop()
            self._apply_edit(edit, "after")
            self._undo_stack.append(edit)
            return True

    def get_data(
        self,
        camera,
        extent=(0, 0, -256, 256, -256, 256),
        out_shape=None,
        level_modifier=0,
        scale_depth=False,
        mask=False,
        order=0,
        axis=0,
        zoom_override=None,
        tile_hw=64,
        rescale=False
    ):
        """
        Tiled sampler.
        Returns:
        (H,W) if D==1 else (D,H,W)
        """

        zoom = zoom_override if zoom_override is not None else camera.zoom

        num_levels = len(self.images)
        base_level = int(np.floor(np.log2(max(zoom, 1e-8))))
        level = int(np.clip(base_level + level_modifier, 0, num_levels - 1))

        volume = self.masks[level] if mask else self.images[level]

        d0, d1, top, bottom, left, right = map(float, extent)

        raw_D = max(1, int(np.ceil(d1 - d0)))
        raw_H = max(1, int(np.ceil(bottom - top)))
        raw_W = max(1, int(np.ceil(right - left)))

        if out_shape is None:
            D, H, W = raw_D, raw_H, raw_W
        else:
            if len(out_shape) == 2:
                H, W = map(int, out_shape)
                D = raw_D
            elif len(out_shape) == 3:
                D, H, W = map(int, out_shape)
            else:
                raise ValueError("out_shape must be None, (H,W), or (D,H,W).")

        D = max(1, int(D))
        H = max(1, int(H))
        W = max(1, int(W))

        s_view = zoom / (2 ** level)

        if scale_depth:
            sd0 = d0 * s_view
            sd1 = d1 * s_view
        else:
            sd0, sd1 = d0, d1

        stop    = top * s_view
        sbottom = bottom * s_view
        sleft   = left * s_view
        sright  = right * s_view

        dd = 0.0 if D == 1 else (sd1 - sd0) / (D - 1)
        dy = 0.0 if H == 1 else (sbottom - stop) / (H - 1)
        dx = 0.0 if W == 1 else (sright - sleft) / (W - 1)

        normal_axis, a0, a1 = camera.slice_axes(axis=axis)

        origin = (camera.origin / (2 ** level)).astype(np.float32)

        pad = 1 if order == 1 else 0

        scale = float(2 ** level)
        inv_scale = 1.0 / scale

        Is, Js, Ks = volume.shape

        output = np.zeros((D, H, W), dtype=np.float32)

        th = tw = max(8, int(tile_hw))

        def depth_samples_for_bounds():
            if pad == 0 and D == 1:
                return (sd0,)
            return (sd0 - pad, sd0 + (D - 1) * dd + pad)

        ds_bounds = depth_samples_for_bounds()

        if not hasattr(self, "_pad_cache"):
            self._pad_cache = {} 

        for y0 in range(0, H, th):
            h_tile = min(th, H - y0)
            y_start = stop + y0 * dy
            y_end   = y_start + (h_tile - 1) * dy if h_tile > 1 else y_start

            for x0 in range(0, W, tw):
                w_tile = min(tw, W - x0)
                x_start = sleft + x0 * dx
                x_end   = x_start + (w_tile - 1) * dx if w_tile > 1 else x_start

                corners = []
                if pad == 0 and D == 1:
                    for y in (y_start, y_end):
                        for x in (x_start, x_end):
                            p0 = camera.world_coords(sd0 * scale, y * scale, x * scale)
                            corners.append(p0 * inv_scale)
                else:
                    for d in ds_bounds:
                        for y in (y_start - pad, y_end + pad):
                            for x in (x_start - pad, x_end + pad):
                                p0 = camera.world_coords(d * scale, y * scale, x * scale)
                                corners.append(p0 * inv_scale)

                pts = np.stack(corners, axis=0)
                if not np.isfinite(pts).all():
                    continue 

                mn = np.floor(pts.min(axis=0)).astype(np.int64)
                mx = (np.ceil(pts.max(axis=0)).astype(np.int64) + 1)

                i0_raw, j0_raw, k0_raw = mn
                i1_raw, j1_raw, k1_raw = mx

                if i1_raw <= 0 or i0_raw >= Is or j1_raw <= 0 or j0_raw >= Js or k1_raw <= 0 or k0_raw >= Ks:
                    continue

                if (0 <= i0_raw) and (i1_raw <= Is) and (0 <= j0_raw) and (j1_raw <= Js) and (0 <= k0_raw) and (k1_raw <= Ks):
                    i0, i1 = int(i0_raw), int(i1_raw)
                    j0, j1 = int(j0_raw), int(j1_raw)
                    k0, k1 = int(k0_raw), int(k1_raw)

                    subvol = np.ascontiguousarray(volume[i0:i1, j0:j1, k0:k1])
                    if subvol.dtype != np.float32:
                        subvol = subvol.astype(np.float32, copy=False)

                    shift = np.array([i0, j0, k0], dtype=np.float32)
                    local_origin = origin - shift

                else:
                    i0 = max(0, int(i0_raw))
                    i1 = min(Is, int(i1_raw))
                    j0 = max(0, int(j0_raw))
                    j1 = min(Js, int(j1_raw))
                    k0 = max(0, int(k0_raw))
                    k1 = min(Ks, int(k1_raw))
                    if i1 <= i0 or j1 <= j0 or k1 <= k0:
                        continue

                    Pi = int(i1_raw - i0_raw)
                    Pj = int(j1_raw - j0_raw)
                    Pk = int(k1_raw - k0_raw)
                    if Pi <= 0 or Pj <= 0 or Pk <= 0:
                        continue

                    key = (Pi, Pj, Pk)
                    buf = self._pad_cache.get(key)
                    if buf is None:
                        buf = np.zeros(key, dtype=np.float32)
                        self._pad_cache[key] = buf
                    else:
                        buf.fill(0.0) 

                    sub = np.ascontiguousarray(volume[i0:i1, j0:j1, k0:k1])
                    if sub.dtype != np.float32:
                        sub = sub.astype(np.float32, copy=False)

                    oi = int(i0 - i0_raw) 
                    oj = int(j0 - j0_raw)
                    ok = int(k0 - k0_raw)
                    buf[oi:oi + sub.shape[0], oj:oj + sub.shape[1], ok:ok + sub.shape[2]] = sub

                    subvol = buf
                    shift_raw = np.array([i0_raw, j0_raw, k0_raw], dtype=np.float32)
                    local_origin = origin - shift_raw

                out_tile = output[:, y0:y0 + h_tile, x0:x0 + w_tile]

                if order == 0:
                    read_nearest(
                        subvol, out_tile,
                        local_origin, normal_axis, a0, a1,
                        sd0, dd, y_start, dy, x_start, dx
                    )
                elif order == 1:
                    read_trilinear(
                        subvol, out_tile,
                        local_origin, normal_axis, a0, a1,
                        sd0, dd, y_start, dy, x_start, dx
                    )
                else:
                    raise ValueError(f"Unsupported interpolation order={order}. Use 0 or 1.")

        # if rescale:
        #    output = self._rescale_image(output)

        return output[0] if D == 1 else output

    def _rescale_image(self, image):
        v_min, v_max = -0.002, 0.007
        np.clip(image, v_min, v_max, out=image)
        image -= v_min
        image *= 255.0 / (v_max - v_min)
        image = image.astype(np.uint8, copy=False)
        return image
 
    def set_data(
        self,
        camera,
        data,
        extent=(0, 0, -256, 256, -256, 256),
        out_shape=None,
        level_modifier=0,
        scale_depth=False,
        mask=True,
        axis=0,
        zoom_override=None,
        tile_hw=64,
        order=0,
        thickness=2,
    ):
        """
        Tiled writer.
        Writes to level 0.
        """

        zoom = zoom_override if zoom_override is not None else camera.zoom
        num_levels = len(self.images)
        base_level = int(np.floor(np.log2(max(float(zoom), 1e-8))))
        view_level = int(np.clip(base_level + level_modifier, 0, num_levels - 1))

        L = 0
        vol = self.masks[L] if mask else self.images[L]

        d0, d1, top, bottom, left, right = map(float, extent)

        data = np.asarray(data)
        if data.ndim == 2:
            data = np.repeat(data[None, ...], int(thickness), axis=0)
            d0 = -0.5 * float(thickness)
            d1 =  0.5 * float(thickness)

        if data.ndim != 3:
            raise ValueError("data must be 2D (H,W) or 3D (D,H,W).")

        D_in, H_in, W_in = map(int, data.shape)

        raw_D = max(1, int(np.ceil(d1 - d0)))
        raw_H = max(1, int(np.ceil(bottom - top)))
        raw_W = max(1, int(np.ceil(right - left)))

        if out_shape is None:
            Dg, Hg, Wg = raw_D, raw_H, raw_W
        else:
            if len(out_shape) == 2:
                Hg, Wg = map(int, out_shape)
                Dg = raw_D
            elif len(out_shape) == 3:
                Dg, Hg, Wg = map(int, out_shape)
            else:
                raise ValueError("out_shape must be None, (H,W), or (D,H,W).")

        Dg = max(1, int(Dg))
        Hg = max(1, int(Hg))
        Wg = max(1, int(Wg))

        s_view = float(zoom) / (2 ** view_level)

        if scale_depth:
            sd0 = d0 * s_view
            sd1 = d1 * s_view
        else:
            sd0, sd1 = d0, d1

        stop    = top * s_view
        sbottom = bottom * s_view
        sleft   = left * s_view
        sright  = right * s_view

        to_level0 = float(2 ** view_level)

        if scale_depth:
            d0_0 = sd0 * to_level0
            d1_0 = sd1 * to_level0
        else:
            d0_0, d1_0 = sd0, sd1

        top_0    = stop    * to_level0
        bottom_0 = sbottom * to_level0
        left_0   = sleft   * to_level0
        right_0  = sright  * to_level0

        dd = 0.0 if Dg == 1 else (d1_0 - d0_0) / (Dg - 1)
        dy = 0.0 if Hg == 1 else (bottom_0 - top_0) / (Hg - 1)
        dx = 0.0 if Wg == 1 else (right_0 - left_0) / (Wg - 1)

        if (D_in, H_in, W_in) != (Dg, Hg, Wg):
            data0 = np.empty((Dg, Hg, Wg), np.uint8)
            z_map = None if D_in == Dg else np.rint(np.linspace(0, D_in - 1, Dg)).astype(np.int64)
            for zi in range(Dg):
                src_z = zi if z_map is None else int(z_map[zi])
                data0[zi] = cv2.resize(
                    data[src_z].astype(np.uint8, copy=False),
                    (Wg, Hg),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            data0 = data.astype(np.uint8, copy=False)

        normal_axis, a0, a1 = camera.slice_axes(axis=axis)
        origin = camera.origin.astype(np.float32)

        if order != 0:
            raise ValueError(
                f"Unsupported interpolation order={order} for set_data. "
                "Currently only order=0 (nearest) is implemented."
            )

        pad = 0

        Is, Js, Ks = vol.shape

        edit = []

        def depth_samples_for_bounds():
            if pad == 0 and Dg == 1:
                return (d0_0,)
            return (d0_0 - pad, d0_0 + (Dg - 1) * dd + pad)

        ds_bounds = depth_samples_for_bounds()

        def _clip(i0, i1, j0, j1, k0, k1):
            i0 = max(0, int(i0))
            j0 = max(0, int(j0))
            k0 = max(0, int(k0))
            i1 = min(Is, int(i1))
            j1 = min(Js, int(j1))
            k1 = min(Ks, int(k1))
            if i1 <= i0:
                i1 = min(Is, i0 + 1)
            if j1 <= j0:
                j1 = min(Js, j0 + 1)
            if k1 <= k0:
                k1 = min(Ks, k0 + 1)
            return i0, i1, j0, j1, k0, k1

        step = max(1, tile_hw - 1)
        for y0 in range(0, Hg, step):
            h_tile = min(tile_hw, Hg - y0)
            y_start = top_0 + y0 * dy
            y_end   = y_start + (h_tile - 1) * dy if h_tile > 1 else y_start

            for x0 in range(0, Wg, step):
                w_tile = min(tile_hw, Wg - x0)
                x_start = left_0 + x0 * dx
                x_end   = x_start + (w_tile - 1) * dx if w_tile > 1 else x_start

                tile = data0[:, y0:y0 + h_tile, x0:x0 + w_tile]
                if tile.max() == 0:
                    continue

                corners = []
                if pad == 0 and Dg == 1:
                    for y in (y_start, y_end):
                        for x in (x_start, x_end):
                            corners.append(camera.world_coords(d0_0, y, x))
                else:
                    for d in ds_bounds:
                        for y in (y_start - pad, y_end + pad):
                            for x in (x_start - pad, x_end + pad):
                                corners.append(camera.world_coords(d, y, x))

                pts = np.stack(corners, axis=0)
                mn = np.floor(pts.min(axis=0)).astype(np.int64)
                mx = (np.ceil(pts.max(axis=0)).astype(np.int64) + 1)

                i0, j0, k0 = mn
                i1, j1, k1 = mx
                i0, i1, j0, j1, k0, k1 = _clip(i0, i1, j0, j1, k0, k1)

                with self._edit_lock:
                    sub = np.ascontiguousarray(vol[i0:i1, j0:j1, k0:k1])
                    before = sub.copy()

                    local_origin = origin - np.array([i0, j0, k0], dtype=np.float32)

                    write_nearest(
                        sub, tile,
                        local_origin, normal_axis, a0, a1,
                        d0_0, dd,
                        y_start, dy,
                        x_start, dx,
                    )

                    after = sub.copy()
                    vol[i0:i1, j0:j1, k0:k1] = sub

                edit.append((0, mask, (i0, i1, j0, j1, k0, k1), before, after))

        # Add edit to history
        with self._edit_lock:
            self._add_to_history(edit)