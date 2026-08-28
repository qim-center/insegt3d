import hashlib
import base64
import shutil
import threading
import cv2
import numpy as np
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nicegui import ui as nicegui_ui

from insegt3d.app.scheduler import JobSpec
from insegt3d.ml import predict2d as predict

def _is_http_url(s: str) -> bool:
    p = urlparse(s.strip())
    return p.scheme in ("http", "https")

def _url_id(url: str, length: int = 10) -> str:
    """Short, filesystem-safe stable id for a URL."""
    digest = hashlib.sha256(url.encode("utf-8")).digest()
    token = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return token[:length]

# Keyed by the toggle widget's value (see ui.py's `toggle_annotation_mode`).
_ANNOTATION_MODES = {0: 'draw', 1: 'save', 2: 'flood', 3: 'mask_fill'}
_ANNOTATION_TOGGLE_VALUES = {mode: value for value, mode in _ANNOTATION_MODES.items()}

def _mask_folder_name(zarr_ref) -> str:
    """
    Local: use folder name as-is.
    Remote: <last_path_component>__<short_hash> (unique per URL).
    """
    s = str(zarr_ref).strip().rstrip("/")
    if _is_http_url(s):
        last = urlparse(s).path.rstrip("/").split("/")[-1] or "remote"
        return f"{last}__{_url_id(s)}"
    return Path(s).name

def _prediction_label_path(project_path, zarr_ref) -> Path:
    """
    Path to the argmax label pyramid ml.predict2d.predict_volume writes for
    a volume, mirroring predict_all_volumes' `predictions/<zarr_file.name>`
    layout.
    """
    zarr_name = Path(str(zarr_ref).rstrip("/")).name
    return Path(project_path) / 'predictions' / zarr_name / 'labels'

class CallbackManager:

    def __init__(self, state, services, renderer, scheduler):
        self.ui = None
        
        self.client = None
        self.state = state
        self.services = services
        self.renderer = renderer
        self.scheduler = scheduler

        # State references
        self.nav = self.state.nav
        self.ui_state = self.state.ui
        self.data = self.state.data
        self.annot = self.state.annot
        self.camera = self.state.camera
        self.train = self.state.train

        # Service references
        self.slicer = self.services.slicer
        self.tracker = self.services.tracker

        # Intensity histogram, computed once per loaded scan
        self._histogram_counts = None
        self._histogram_range = None

        self._predict_exec = ThreadPoolExecutor(max_workers=1)
        self._predict_cancel_event = threading.Event()
        self.scheduler.register_sync(
            "predict_volumes",
            fn=self._run_predict_volumes,
            spec=JobSpec(
                mode="drop",
                executor=self._predict_exec,
                sequential_executor=False,
            ),
        )

    def _request_slice_update(self):
        self.scheduler.request("nav_hires")
        if "sync_navigator" in self.scheduler.jobs:
            self.scheduler.request("sync_navigator")

    def _open_active_volume(self, update_histogram=True):
        """
        Points the slicer and navigator at data.active_zarr, along with its
        mask folder and any prediction labels already written for it.
        """
        mask_path = Path(self.data.project_path) / 'masks' / _mask_folder_name(self.data.active_zarr)
        label_path = _prediction_label_path(self.data.project_path, self.data.active_zarr)

        self.slicer.initialize(
            self.data.active_zarr, mask_path, self.camera,
            center_camera=True, prediction_path=label_path,
        )
        self.ui.navigator.initialize(self.slicer.shapes[0], self.scheduler)

        if update_histogram:
            self._update_histogram()

        self._refresh_prediction_overlay_availability()

    def on_viewport_resize(self, e):
        d = e.args['detail']
        h, w = d['h'], d['w']
        self.ui_state.viewport_shape = (h, w)
        self.nav.slice_shape = self._clamp_slice_shape(h, w)

        self.ui.slider_brush_size.props(f'max={self.ui_state.max_brush_size()}')
        self.set_brush_size()

        if self.slicer.zarr_path is not None:
            self._request_slice_update()

    def _clamp_slice_shape(self, h, w):
        max_pixels = self.nav.max_slice_megapixels * 1e6
        num_pixels = h * w

        if num_pixels <= max_pixels or num_pixels <= 0:
            return (h, w)

        scale = (max_pixels / num_pixels) ** 0.5
        return (max(1, round(h * scale)), max(1, round(w * scale)))

    def _fail_zarr_load(self, message):
        self.ui.select_scan.options = {}
        self.ui.select_scan.update()
        nicegui_ui.notify(message, type='warning')

    def load_zarr_files(self):
        input_path = (self.ui.input_path.value or "").strip()

        if not input_path:
            self._fail_zarr_load('Enter a path or URL to load.')
            return

        if "," in input_path:
            # Multiple comma separated remote urls
            parts = [p.strip() for p in input_path.split(",") if p.strip()]
            if parts and all(_is_http_url(p) for p in parts):
                self.data.zarr_files = parts
            else:
                self._fail_zarr_load('All comma-separated paths must be http(s) URLs.')
                return
        elif _is_http_url(input_path):
            # Single remote URL
            self.data.zarr_files = [input_path]
        else:
            # Single local zarr file or folder containing multiple zarr files
            root = Path(input_path)

            if root.suffix == ".zarr":
                self.data.zarr_files = [root]

            elif root.exists() and root.is_dir():
                self.data.zarr_files = sorted(root.glob("*.zarr"))
                if not self.data.zarr_files:
                    self._fail_zarr_load(f"No .zarr files found in '{input_path}'.")
                    return
            else:
                self._fail_zarr_load(f"Path not found: '{input_path}'.")
                return

        self.data.zarr_idx = 0

        self._open_active_volume()

        # Update UI select with zarr names
        # Local: ".../something.zarr" -> "something"
        # Remote: ".../xray" -> "xray"
        names = [Path(str(f).rstrip("/")).stem for f in self.data.zarr_files]
        self.ui.select_scan.options = dict(enumerate(names))
        self.ui.select_scan.update()

        # Request slice update
        self._request_slice_update()

    def select_scan(self):

        self.data.zarr_idx = int(self.ui.select_scan.value)

        self._open_active_volume()
        self._request_slice_update()

    def _refresh_prediction_overlay_availability(self):
        available = self.slicer.has_prediction
        self.ui.checkbox_saved_prediction_overlay.set_enabled(available)
        self.ui.slider_saved_prediction_opacity.set_enabled(available)
        if not available and self.ui.checkbox_saved_prediction_overlay.value:
            self.ui.checkbox_saved_prediction_overlay.value = False
            self.ui_state.saved_prediction.visible = False

    def _update_histogram(self):
        counts, value_range = self.slicer.compute_histogram()
        self._histogram_counts = counts
        self._histogram_range = value_range

        if value_range is not None:
            v_min, v_max = value_range

            robust_range = self.slicer.get_robust_intensity_range()
            if robust_range is not None:
                low, high = robust_range
                low = float(np.clip(low, v_min, v_max))
                high = float(np.clip(high, v_min, v_max))
            else:
                low, high = v_min, v_max

            self.ui_state.intensity_low = low
            self.ui_state.intensity_high = high

            step = max((v_max - v_min) / 500.0, 1e-6)

            self.ui.range_intensity.min = v_min
            self.ui.range_intensity.max = v_max
            self.ui.range_intensity.step = step
            self.ui.range_intensity.value = {'min': low, 'max': high}
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
        bar_color = (165, 176, 191)      # slate-300ish: bins inside the selected window
        dim_color = (223, 226, 231)      # gray-200ish: bins outside the selected window
        marker_color = (220, 38, 38)     # red-600: window boundaries

        canvas = np.full((height, width, 3), background, dtype=np.uint8)

        if counts is None or value_range is None or counts.sum() == 0:
            return canvas

        v_min, v_max = value_range
        span = max(v_max - v_min, 1e-6)

        n_bins = len(counts)
        bin_edges = v_min + (np.arange(n_bins + 1) / n_bins) * span
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Log scale so a single dominant bin (e.g. background) doesn't flatten the rest.
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

    def refresh_button_palette(self):
        """
        Enables one palette button per class and outlines the active one.
        """
        num_classes = int(self.train.num_classes)

        if num_classes > 0:
            self.annot.color_idx = max(0, min(int(self.annot.color_idx), num_classes - 1))
        else:
            self.annot.color_idx = 0

        for i, button in enumerate(self.ui.button_palette):
            enabled = i < num_classes
            button.set_enabled(enabled)

            if enabled:
                border = 'black' if i == self.annot.color_idx else 'transparent'
                button.style(f'opacity:1.0; filter:none; border:2px solid {border};')
            else:
                button.style('opacity:0.25; filter:grayscale(100%); border:2px solid transparent;')

    def undo(self):
        self.slicer.undo()
        self.tracker.undo()
        self._request_slice_update()

    def redo(self):
        self.slicer.redo()
        self.tracker.redo()
        self._request_slice_update()

    def toggle_annotation_mode(self):
        self.annot.annotating = False
        
        mode = _ANNOTATION_MODES.get(self.ui.toggle_annotation_mode.value)
        if mode is not None:
            self.annot.mode = mode

    def set_annotation_mode(self):
        self.annot.annotating = False
        
        toggle_value = _ANNOTATION_TOGGLE_VALUES.get(self.annot.mode)
        if toggle_value is not None:
            self.ui.toggle_annotation_mode.value = toggle_value

    def _clamp_brush_size(self, size):
        return max(1, min(size, self.ui_state.max_brush_size()))

    def update_brush_size(self):
        self.annot.brush_size = self._clamp_brush_size(self.ui.slider_brush_size.value)

    def set_brush_size(self):
        self.annot.brush_size = self._clamp_brush_size(self.annot.brush_size)
        self.ui.slider_brush_size.value = self.annot.brush_size

    def toggle_prediction_overlay(self):
        self.ui_state.prediction.visible = self.ui.checkbox_prediction_overlay.value
        self.renderer.update()

    def set_prediction_overlay(self):
        self.ui.checkbox_prediction_overlay.value = self.ui_state.prediction.visible
        self.renderer.update()

    def update_prediction_opacity(self):
        self.ui_state.prediction.alpha = self.ui.slider_prediction_opacity.value
        self.renderer.update()

    def toggle_mask_overlay(self):
        self.ui_state.mask.visible = self.ui.checkbox_mask_overlay.value
        self.renderer.update()

    def update_mask_opacity(self):
        self.ui_state.mask.alpha = self.ui.slider_mask_opacity.value
        self.renderer.update()

    def toggle_saved_prediction_overlay(self):
        self.ui_state.saved_prediction.visible = self.ui.checkbox_saved_prediction_overlay.value
        self.renderer.update()

    def update_saved_prediction_opacity(self):
        self.ui_state.saved_prediction.alpha = self.ui.slider_saved_prediction_opacity.value
        self.renderer.update()

    def update_properties(self):
        origin = self.camera.origin
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
        self.scheduler.request("predict_volumes")

    def cancel_predict_volumes(self):
        self._predict_cancel_event.set()
        self.ui.button_cancel_predict.set_enabled(False)
        self.ui.button_cancel_predict.set_text('Cancelling...')

    @staticmethod
    def _format_duration(seconds):
        if seconds is None or seconds < 0:
            return '--'

        seconds = int(round(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        if hours:
            return f'{hours}h {minutes:02d}m'
        if minutes:
            return f'{minutes}m {seconds:02d}s'
        return f'{seconds}s'

    def _on_predict_progress(self, volume_name, vol_idx, num_volumes, block_idx, num_blocks, eta_seconds):
        self.ui.label_predict_status.text = f'Predicting {volume_name}... - ({vol_idx + 1}/{num_volumes})'
        self.ui.progress_predict.value = (block_idx / num_blocks) if num_blocks else 0.0
        self.ui.label_predict_chunks.text = f'{block_idx}/{num_blocks} blocks · ETA {self._format_duration(eta_seconds)}'

    def _run_predict_volumes(self):
        self.train.predicting = True

        for job_name in ("live_train", "live_predict"):
            if job_name in self.scheduler.jobs:
                self.scheduler.jobs[job_name].cancel()

        self.ui.button_predict.set_enabled(False)
        self.ui.button_predict.set_text('Predicting...')
        self.ui.button_load.set_enabled(False)
        self.ui.button_reset_model.set_enabled(False)
        self.ui.checkbox_export_tiff.set_enabled(False)
        self.ui.checkbox_prediction_overlay.set_enabled(False)
        self.ui.slider_prediction_opacity.set_enabled(False)
        self.ui_state.prediction.visible = False
        self.set_prediction_overlay()

        self.ui.label_predict_status.text = f'Preparing {len(self.data.zarr_files)} volume(s)...'
        self.ui.label_predict_status.set_visibility(True)
        self.ui.progress_predict.value = 0.0
        self.ui.progress_predict.set_visibility(True)
        self.ui.label_predict_chunks.text = ''
        self.ui.label_predict_chunks.set_visibility(True)
        self.ui.button_cancel_predict.set_visibility(True)

        cancelled_volume = None
        try:
            predict.predict_all_volumes(
                self.data.zarr_files,
                self.data.project_path,
                num_classes=self.train.num_classes,
                export_tiff=self.train.export_tiff,
                progress_callback=self._on_predict_progress,
                cancel_event=self._predict_cancel_event
            )
        except predict.PredictionCancelled as e:
            cancelled_volume = e.volume_name
        finally:
            self.train.predicting = False

            self.ui.label_predict_status.set_visibility(False)
            self.ui.progress_predict.set_visibility(False)
            self.ui.label_predict_chunks.set_visibility(False)
            self.ui.button_cancel_predict.set_visibility(False)
            self.ui.button_cancel_predict.set_enabled(True)
            self.ui.button_cancel_predict.set_text('Cancel')

            self.ui.button_predict.set_enabled(True)
            self.ui.button_predict.set_text('Predict')
            self.ui.button_load.set_enabled(True)
            self.ui.button_reset_model.set_enabled(True)
            self.ui.checkbox_export_tiff.set_enabled(True)
            self.ui.checkbox_prediction_overlay.set_enabled(True)
            self.ui.slider_prediction_opacity.set_enabled(True)
            self.ui_state.prediction.visible = True
            self.set_prediction_overlay()

            if self.data.active_zarr is not None:
                label_path = _prediction_label_path(self.data.project_path, self.data.active_zarr)
                self.slicer.refresh_prediction(label_path)
                self._refresh_prediction_overlay_availability()

            if "live_predict" in self.scheduler.jobs:
                self.scheduler.request("live_predict")

            self._request_slice_update()

        if cancelled_volume is not None:
            with self.client:
                nicegui_ui.notify(f'Prediction cancelled — removed partial output for {cancelled_volume}.', type='warning')
        elif self._predict_cancel_event.is_set():
            with self.client:
                nicegui_ui.notify('Prediction cancelled.', type='warning')

    def close(self):
        self._predict_exec.shutdown(wait=False, cancel_futures=True)

    def toggle_export_tiff(self):
        self.train.export_tiff = self.ui.checkbox_export_tiff.value

    def toggle_live_training(self):
        self.train.live_training_enabled = self.ui.checkbox_live_training.value

    def select_architecture(self):
        self.train.architecture = self.ui.select_architecture.value
        self._rebuild_model_if_unlocked()

    def select_encoder(self):
        self.train.encoder_name = self.ui.select_encoder.value
        self._rebuild_model_if_unlocked()

    def _rebuild_model_if_unlocked(self):
        if self.train.model_locked:
            return
        if "reset_model" in self.scheduler.jobs:
            self.scheduler.request("reset_model")

    def set_model_lock(self):
        enabled = not self.train.model_locked
        self.ui.select_architecture.set_enabled(enabled)
        self.ui.select_encoder.set_enabled(enabled)

    def update_live_train_progress(self, step, total_steps, loss):
        self.ui.progress_live_train.value = (step / total_steps) if total_steps else 0.0
        self.ui.label_live_train_status.text = f'Step {step}/{total_steps} · loss {loss:.4f}'

    def update_learning_rate(self):
        self.train.lr = float(self.ui.number_learning_rate.value)

    def update_batch_size(self):
        self.train.batch_size = int(self.ui.number_batch_size.value)

    def update_steps_per_epoch(self):
        self.train.steps_per_epoch = int(self.ui.number_steps_per_epoch.value)

    def reset_model(self):
        if "reset_model" in self.scheduler.jobs:
            self.scheduler.request("reset_model")

    def reset_annotations(self):
        self.tracker.reset()

        masks_root = Path(self.data.project_path) / 'masks'
        if masks_root.exists():
            shutil.rmtree(masks_root)

        if self.data.active_zarr is not None:
            self._open_active_volume(update_histogram=False)

        self._request_slice_update()
