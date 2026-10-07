import base64
import math
import asyncio
import threading
import cv2
import numpy as np
from pathlib import Path
from functools import wraps
from datetime import timedelta

from nicegui import run, ui as nicegui_ui

from insegt3d.ml import predict2d as predict
from insegt3d.volume.io import resolve_zarr_inputs, volume_folder_name
from insegt3d.volume.camera import AXIS_VIEWS

def _prediction_label_path(project_path, zarr_ref) -> Path:
    """Argmax label pyramid that ml.predict2d.predict_volume writes for a volume."""
    return Path(project_path) / 'predictions' / volume_folder_name(zarr_ref) / 'labels'

def level_label(level, factors):
    if np.all(factors == factors[0]):
        return f'Level {level} - ' + ('full resolution' if factors[0] == 1 else f'1/{factors[0]:g} resolution')
    return f'Level {level} - ' + ', '.join(f'{axis} 1/{factor:g}' for axis, factor in zip('zyx', factors))

def on_view(method):
    """Runs the method on the event loop thread, and only while a browser view is open."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        def call():
            if self.ui is not None:
                method(self, *args, **kwargs)
        self.scheduler.call_soon(call)
    return wrapper

class CallbackManager:

    def __init__(self, state, services, renderer, scheduler):
        self.ui = None
        self.client = None
        self.trainer = None

        self.state = state
        self.services = services
        self.renderer = renderer
        self.scheduler = scheduler

        self.nav = self.state.nav
        self.ui_state = self.state.ui
        self.data = self.state.data
        self.annot = self.state.annot
        self.camera = self.state.camera
        self.train = self.state.train

        self.slicer = self.services.slicer
        self.history = self.services.history

        self._histogram_counts = None
        self._histogram_range = None
        self._plane_idx = None

        self._open_lock = asyncio.Lock()
        self._predict_cancel_event = threading.Event()

        self.scheduler.register_async("sync_navigator", self._sync_navigator, max_hz=15)

    @on_view
    def notify(self, message, type='warning'):
        with self.client:
            nicegui_ui.notify(message, type=type)

    async def _sync_navigator(self):
        if self.ui is not None:
            self.ui.navigator.sync()

    def _request_slice_update(self):
        self.scheduler.request("nav_hires")
        self.scheduler.request("sync_navigator")

    async def _open_active_volume(self, update_histogram=True):
        zarr_ref = self.data.active_zarr

        async with self._open_lock:
            await run.io_bound(
                self.slicer.initialize, zarr_ref, Path(self.data.project_path) / 'masks' / volume_folder_name(zarr_ref), self.camera,
                center_camera=True, prediction_path=_prediction_label_path(self.data.project_path, zarr_ref),
            )
            self.history.clear()

            if update_histogram:
                await self._update_histogram()

        self.refresh_view()

    def refresh_view(self):
        if self.ui is None or self.slicer.images is None:
            return

        self.ui.navigator.initialize(self.slicer.world_shape)
        self._refresh_intensity_range()
        self._refresh_prediction_overlay_availability()
        self._refresh_level_options()
        self.update_properties()
        self._plane_idx = None
        self.ui.label_plane.text = 'Annotated slices'

    def _refresh_level_options(self):
        num_levels = len(self.slicer.images) if self.slicer.images is not None else 0

        if not self.train.model_locked and num_levels:
            self.train.level = min(self.train.level, num_levels - 1)
        self.ui.select_level.set_options(self.level_options(), value=self.train.level)

        self.train.level_available = self.train.level < num_levels
        self.ui.button_predict.set_enabled(self.train.level_available)
        self.ui.label_live_train_status.text = '' if self.train.level_available else (
            f'Training and prediction disabled: this volume has no level {self.train.level}.')

    def level_options(self):
        options = {}
        if self.slicer.images is not None:
            options = {level: level_label(level, factors) for level, factors in enumerate(self.slicer.voxel_sizes / self.slicer.voxel_sizes[0])}
        options.setdefault(self.train.level, f'Level {self.train.level}')
        return options

    def on_viewport_resize(self, e):
        d = e.args['detail']
        h, w = d['h'], d['w']
        self.ui_state.viewport_shape = (h, w)
        self.nav.slice_shape = self._clamp_slice_shape(h, w)

        self.ui.slider_brush_size.props['max'] = self.ui_state.max_brush_size()
        self.set_brush_size()

        if self.slicer.zarr_path is not None:
            self._request_slice_update()

    def _clamp_slice_shape(self, h, w):
        max_pixels = self.nav.max_slice_megapixels * 1e6
        num_pixels = h * w

        if num_pixels <= max_pixels:
            return (h, w)

        scale = (max_pixels / num_pixels) ** 0.5
        return (max(1, round(h * scale)), max(1, round(w * scale)))

    async def load_zarr_files(self):
        try:
            zarr_files = await run.io_bound(resolve_zarr_inputs, self.data.input_path)
        except (FileNotFoundError, ValueError) as e:
            nicegui_ui.notify(str(e), type='warning')
            return

        if zarr_files == self.data.zarr_files:
            return

        self.data.zarr_files = zarr_files
        self.data.zarr_idx = 0

        # set_options only fires select_scan if the selection changes, so call it directly otherwise
        selection_changed = self.ui.select_scan.value != 0
        self.ui.select_scan.set_options(self.scan_options(), value=0)

        if not selection_changed:
            await self.select_scan()

    def scan_options(self):
        return {i: Path(str(f).rstrip("/")).stem for i, f in enumerate(self.data.zarr_files)} or {0: 'None'}

    async def select_scan(self):

        self.data.zarr_idx = int(self.ui.select_scan.value or 0)
        if self.data.active_zarr is None:
            return

        try:
            await self._open_active_volume()
        except Exception:
            nicegui_ui.notify(f'Could not open {self.data.active_zarr}.', type='negative')
            self.data.zarr_files = []
            self.ui.select_scan.set_options(self.scan_options(), value=0)
            raise

        self._request_slice_update()

    @on_view
    def _refresh_prediction_overlay_availability(self):
        available = self.slicer.has_prediction
        self.ui.checkbox_saved_prediction_overlay.set_enabled(available)
        self.ui.slider_saved_prediction_opacity.set_enabled(available)
        if not available:
            self.ui.checkbox_saved_prediction_overlay.value = False

    async def _update_histogram(self):
        counts, value_range, (low, high) = await run.io_bound(self.slicer.intensity_stats)
        self._histogram_counts = counts
        self._histogram_range = value_range

        v_min, v_max = value_range

        self.ui_state.intensity_low = float(np.clip(low, v_min, v_max))
        self.ui_state.intensity_high = float(np.clip(high, v_min, v_max))

    def _refresh_intensity_range(self):
        v_min, v_max = self._histogram_range
        step = max((v_max - v_min) / 500.0, 1e-6)

        self.ui.range_intensity.min = v_min
        self.ui.range_intensity.max = v_max
        self.ui.range_intensity.step = step
        self.ui.range_intensity.value = {'min': self.ui_state.intensity_low, 'max': self.ui_state.intensity_high}
        self.ui.range_intensity.update()

        self._refresh_histogram_image()
        self._refresh_intensity_label()
        self.renderer.refresh_intensity_scaling()

    def update_intensity_range(self):
        value = self.ui.range_intensity.value
        low = float(value['min'])
        high = float(value['max'])

        if high <= low:
            return

        self.ui_state.intensity_low = low
        self.ui_state.intensity_high = high

        self._refresh_histogram_image()
        self._refresh_intensity_label()
        self.renderer.refresh_intensity_scaling()

    def _refresh_intensity_label(self):
        low = self.ui_state.intensity_low
        high = self.ui_state.intensity_high

        v_min = self.ui.range_intensity.min
        v_max = self.ui.range_intensity.max
        span = max(v_max - v_min, 1e-9)

        low_pct = float(np.clip((low - v_min) / span, 0.0, 1.0)) * 100.0
        high_pct = float(np.clip((high - v_min) / span, 0.0, 1.0)) * 100.0

        self.ui.label_intensity_low.text = f'{low:.4g}'
        self.ui.label_intensity_low.style(f'left: {low_pct}%')

        self.ui.label_intensity_high.text = f'{high:.4g}'
        self.ui.label_intensity_high.style(f'left: {high_pct}%')

    def _refresh_histogram_image(self):
        canvas = self._render_histogram_image(
            self._histogram_counts,
            self._histogram_range,
            self.ui_state.intensity_low,
            self.ui_state.intensity_high
        )
        _, encoded = cv2.imencode('.png', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(encoded).decode('ascii')
        self.ui.image_histogram.set_source(f'data:image/png;base64,{b64}')

    @staticmethod
    def _render_histogram_image(counts, value_range, low, high, width=440, height=64):
        background = (245, 245, 245)
        bar_color = (165, 176, 191)
        dim_color = (223, 226, 231)
        marker_color = (220, 38, 38)

        canvas = np.full((height, width, 3), background, dtype=np.uint8)

        if counts is None or value_range is None or counts.sum() == 0:
            return canvas

        v_min, v_max = value_range
        span = max(v_max - v_min, 1e-6)

        n_bins = len(counts)
        bin_edges = v_min + (np.arange(n_bins + 1) / n_bins) * span
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Log scale so a single dominant bin (e.g. background) doesn't flatten the rest
        heights = np.log1p(counts.astype(np.float32))
        max_height = heights.max()
        if max_height > 0:
            heights = heights / max_height * (height - 2)

        for i, h in enumerate(heights):
            x0 = int(i * width / n_bins)
            x1 = max(x0 + 1, int((i + 1) * width / n_bins))
            y0 = height - 1 - int(h)
            color = bar_color if low <= bin_centers[i] <= high else dim_color
            cv2.rectangle(canvas, (x0, max(0, y0)), (x1 - 1, height - 1), color, thickness=-1)

        def x_for_value(v):
            frac = (v - v_min) / span
            return int(np.clip(frac, 0.0, 1.0) * (width - 1))

        cv2.line(canvas, (x_for_value(low), 0), (x_for_value(low), height - 1), marker_color, 1)
        cv2.line(canvas, (x_for_value(high), 0), (x_for_value(high), height - 1), marker_color, 1)

        return canvas

    def on_pick_color(self, i):
        self.annot.color_idx = int(i)
        self.refresh_button_palette()

    @on_view
    def refresh_button_palette(self):
        self.annot.color_idx = min(self.annot.color_idx, self.train.num_classes - 1)

        for i, button in enumerate(self.ui.button_palette):
            border = 'black' if i == self.annot.color_idx else 'transparent'
            button.set_visibility(i < self.train.num_classes)
            button.style(f'background:{self.annot.colors[i]} !important; border:2px solid {border};')

        self.ui.button_add_class.set_visibility(self.train.num_classes < len(self.annot.colors))
        self.ui.button_remove_class.set_visibility(self.train.num_classes > 2)

    def add_class(self):
        if not self.train.changing_classes:
            self.train.changing_classes = True
            self.scheduler.request("change_classes")

    async def remove_class(self):
        if self.train.changing_classes:
            return

        i = self.annot.color_idx
        annotations = [a for a in self.services.tracker.annotations() if a.class_idx == i + 1]
        volumes = len({a.volume_path for a in annotations})
        self.ui.label_remove_class.text = (
            f'Remove class {i + 1}? Its {len(annotations)} annotation(s) in {volumes} volume(s) will be erased and later '
            f'classes renumbered, keeping their colours. Existing predictions keep the old classes. This cannot be undone.'
        )
        if await self.ui.dialog_remove_class:
            self.train.changing_classes = True
            self.scheduler.request("change_classes", i)

    def _clamp_brush_size(self, size):
        return max(1, min(size, self.ui_state.max_brush_size()))

    def update_brush_size(self):
        self.annot.brush_size = self._clamp_brush_size(self.ui.slider_brush_size.value)

    def set_brush_size(self):
        self.annot.brush_size = self._clamp_brush_size(self.annot.brush_size)
        self.ui.slider_brush_size.value = self.annot.brush_size

    def toggle_live_prediction(self):
        checkbox = self.ui.checkbox_prediction_overlay
        checkbox.value = not checkbox.value

    def toggle_crosshair(self):
        checkbox = self.ui.checkbox_crosshair
        checkbox.value = not checkbox.value

    def align_view(self, axis):
        self.camera.set_view(self.camera.origin, self.camera.zoom, AXIS_VIEWS[axis])
        self._request_slice_update()

    def center_view(self):
        if self.slicer.images is not None:
            self.camera.reset(self.slicer.world_shape)
            self._request_slice_update()

    async def go_to(self):
        if self.slicer.images is None:
            return

        fields = self.ui.numbers_go_to
        for field, value in zip(fields, (*self.camera.origin / self.slicer.voxel_sizes[0], *self.camera.u)):
            field.value = round(float(value), 3)
        if not await self.ui.dialog_go_to:
            return

        location, normal = np.split(np.array([field.value or 0 for field in fields], dtype=np.float64), 2)
        self.camera.set_view(location * self.slicer.voxel_sizes[0], self.camera.zoom, self.camera.uvw)
        if normal.any():
            self.camera.look_along(normal)
        self._request_slice_update()

    def _annotated_planes(self):
        planes = {}
        for a in self.services.tracker.annotations():
            if a.volume_path == str(self.slicer.zarr_path):
                # Strokes with the same normal (up to sign) and offset along it lie on the same plane
                u = a.camera.u * np.sign(a.camera.u[np.argmax(np.abs(a.camera.u))])
                planes.setdefault((*np.round(u, 3), round(float(u @ a.camera.origin))), []).append(a)
        return list(planes.values())

    def step_plane(self, step):
        planes = self._annotated_planes()
        if not planes:
            return

        start = len(planes) if self._plane_idx is None else self._plane_idx
        self._plane_idx = (start + step) % len(planes)
        annotations = planes[self._plane_idx]

        # Frame every stroke on the plane, but never zoom in past full resolution
        ref = annotations[-1].camera
        axes = np.stack([ref.v, ref.w])
        corners = np.array([
            a.camera.origin + sv * a.extent[3] * a.camera.v + sw * a.extent[5] * a.camera.w
            for a in annotations for sv in (-1, 1) for sw in (-1, 1)
        ])
        rel = (corners - ref.origin) @ axes.T
        lo, hi = rel.min(axis=0), rel.max(axis=0)
        zoom = max(1.0, 1.2 * float(np.max((hi - lo) / self.nav.slice_shape)))

        self.camera.set_view(ref.origin + ((lo + hi) / 2) @ axes, zoom, ref.uvw)
        self.ui_state.mask.visible = True
        self.ui.label_plane.text = f'Annotated slice {self._plane_idx + 1} / {len(planes)}'
        self._request_slice_update()

    @on_view
    def update_properties(self):
        origin = self.camera.origin / self.slicer.voxel_sizes[0]
        u, v, w = self.camera.uvw
        zoom = 1 / self.camera.zoom
        volume_shape = self.slicer.shapes[0]

        def fmt(vec):
            return f'{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}'

        self.ui.label_origin.text = f'z: {origin[0]:.02f}, y: {origin[1]:.02f}, x: {origin[2]:.02f}'
        self.ui.label_rotation_u.text = f'u  {fmt(u)}'
        self.ui.label_rotation_v.text = f'v  {fmt(v)}'
        self.ui.label_rotation_w.text = f'w  {fmt(w)}'
        self.ui.label_zoom.text = f'{zoom:.02f}'
        self.ui.label_shape.text = f'z: {volume_shape[0]}, y: {volume_shape[1]}, x: {volume_shape[2]}'

    def predict_volumes(self):
        self._predict_cancel_event.clear()
        self.train.predicting = True
        self.ui.button_cancel_predict.set_enabled(True)
        self.ui.button_cancel_predict.set_text('Cancel')
        self._show_predict_status(f'Preparing {len(self.data.zarr_files)} volume(s)...')
        self.scheduler.request("predict_volumes")

    def cancel_predict_volumes(self):
        self._predict_cancel_event.set()
        self.ui.button_cancel_predict.set_enabled(False)
        self.ui.button_cancel_predict.set_text('Cancelling...')

    @on_view
    def _show_predict_status(self, status, progress=0.0, chunks=''):
        self.ui.label_predict_status.text = status
        self.ui.progress_predict.value = progress
        self.ui.label_predict_chunks.text = chunks

    def _on_predict_progress(self, volume_name, vol_idx, num_volumes, block_idx, num_blocks, eta_seconds):
        self._show_predict_status(
            f'Predicting {volume_name}... - ({vol_idx + 1}/{num_volumes})',
            (block_idx / num_blocks) if num_blocks else 0.0,
            f'{block_idx}/{num_blocks} blocks · ETA {timedelta(seconds=round(eta_seconds))}',
        )

    def run_predict_volumes(self):
        self.train.predicting = True
        live_prediction_visible = self.ui_state.prediction.visible
        self.ui_state.prediction.visible = False

        for job_name in ("live_train", "live_predict"):
            self.scheduler.jobs[job_name].cancel()

        cancelled_volume = None
        try:
            self.trainer.predict_volumes(
                self.data.zarr_files,
                Path(self.data.project_path),
                export_tiff=self.train.export_tiff,
                progress_callback=self._on_predict_progress,
                cancel_event=self._predict_cancel_event
            )
        except predict.PredictionCancelled as e:
            cancelled_volume = e.volume_name
        except Exception as e:
            self.notify(f'Prediction failed: {e}', type='negative')
            raise
        finally:
            self.train.predicting = False
            self.ui_state.prediction.visible = live_prediction_visible

            if self.data.active_zarr is not None:
                label_path = _prediction_label_path(self.data.project_path, self.data.active_zarr)
                self.slicer.refresh_prediction(label_path)
                self._refresh_prediction_overlay_availability()

            self.scheduler.request("live_predict")
            self._request_slice_update()

        if cancelled_volume is not None:
            self.notify(f'Prediction cancelled — removed partial output for {cancelled_volume}.')
        elif self._predict_cancel_event.is_set():
            self.notify('Prediction cancelled.')

    def select_architecture(self):
        self.train.architecture = self.ui.select_architecture.value
        self._rebuild_model_if_unlocked()

    def select_encoder(self):
        self.train.encoder_name = self.ui.select_encoder.value
        self._rebuild_model_if_unlocked()

    def update_level(self):
        self.train.level = int(self.ui.select_level.value)
        self._refresh_level_options()
        self._rebuild_model_if_unlocked()

    def _round_up_odd(self, value, lo, hi):
        value = min(max(math.ceil(value or lo), lo), hi)
        return value if value % 2 else value + 1

    def update_slab_size(self):
        self.train.slab_size = self._round_up_odd(self.ui.number_slab_size.value, 1, 15)
        self.ui.number_slab_size.value = self.train.slab_size
        self._rebuild_model_if_unlocked()

    def update_scnp_size(self):
        self.train.scnp_size = self._round_up_odd(self.ui.number_scnp_size.value, 3, 15)
        self.ui.number_scnp_size.value = self.train.scnp_size

    def _rebuild_model_if_unlocked(self):
        if not self.train.model_locked:
            self.scheduler.request("reset_model")

    @on_view
    def set_model_lock(self):
        enabled = not self.train.model_locked
        self.ui.select_architecture.set_enabled(enabled)
        self.ui.select_encoder.set_enabled(enabled)
        self.ui.number_slab_size.set_enabled(enabled)
        self.ui.select_level.set_enabled(enabled)

    @on_view
    def update_live_train_progress(self, step, total_steps, loss):
        self.ui.progress_live_train.value = (step / total_steps) if total_steps else 0.0
        self.ui.label_live_train_status.text = f'Step {step}/{total_steps} · loss {loss:.4f}'

    @on_view
    def show_live_train_hint(self, text):
        self.ui.progress_live_train.value = 0.0
        self.ui.label_live_train_status.text = text

    async def reset_annotations(self):
        if not await self.ui.dialog_reset_annotations:
            return

        await run.io_bound(self.history.reset, Path(self.data.project_path) / 'masks')

        if self.data.active_zarr is not None:
            await self._open_active_volume(update_histogram=False)

        self._request_slice_update()
