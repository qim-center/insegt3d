
import hashlib
import base64
from urllib.parse import urlparse

from pathlib import Path

from interactive_unet.ml import predict2d as predict

def _is_http_url(s: str) -> bool:
    p = urlparse(s.strip())
    return p.scheme in ("http", "https")

def _url_id(url: str, length: int = 10) -> str:
    """Short, filesystem-safe stable id for a URL."""
    digest = hashlib.sha256(url.encode("utf-8")).digest()
    token = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return token[:length]

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

class CallbackManager:

    def __init__(self, state, services, renderer, scheduler):
        self.ui = None
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

    def _request_slice_update(self):
        self.scheduler.request("nav_hires")
        if "sync_navigator" in self.scheduler.jobs.keys():
            self.scheduler.request("sync_navigator")
    
    def on_viewport_resize(self, e):
        d = e.args['detail']
        self.nav.slice_shape = (d['h'], d['w'])
        self.ui_state.viewport_shape = (d['h'], d['w'])
        if self.slicer.zarr_path is not None:
            self._request_slice_update()

    def load_zarr_files(self):
        input_path = (self.ui.input_path.value or "").strip()

        if "," in input_path:
            # Multiple comma separated remote urls
            parts = [p.strip() for p in input_path.split(",") if p.strip()]
            if parts and all(_is_http_url(p) for p in parts):
                self.data.zarr_files = parts
            else:
                self.ui.select_scan.options = {}
                self.ui.select_scan.update()
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
                    self.ui.select_scan.options = {}
                    self.ui.select_scan.update()
                    return
            else:
                self.ui.select_scan.options = {}
                self.ui.select_scan.update()
                return

        # Reset zarr index
        self.data.zarr_idx = 0

        # Initialize volume
        mask_name = _mask_folder_name(self.data.active_zarr)
        mask_path = Path(self.data.project_path) / "masks" / mask_name
        self.slicer.initialize(self.data.active_zarr, mask_path, self.camera, center_camera=True)
        self.ui.navigator.initialize(self.slicer.shapes[0], self.scheduler)

        # Update UI select with zarr names
        # Local: ".../something.zarr" -> "something"
        # Remote: ".../xray" -> "xray"
        names = [Path(str(f).rstrip("/")).stem for f in self.data.zarr_files]
        self.ui.select_scan.options = dict(enumerate(names))
        self.ui.select_scan.update()

        # Request slice update
        self._request_slice_update()

    def select_scan(self):

        # Update zarr index if different from current
        idx = int(self.ui.select_scan.value)
        if idx != self.data.zarr_idx:
            self.data.zarr_idx = idx

        # Update slicer service
        mask_path = Path(self.data.project_path) / 'masks' / Path(self.data.active_zarr).name
        self.slicer.initialize(self.data.active_zarr, mask_path, self.camera, center_camera=True)
        self.ui.navigator.initialize(self.slicer.shapes[0], self.scheduler)

        # Request slice update
        self._request_slice_update()

    def update_num_classes(self):
        self.train.num_classes = self.ui.select_num_classes.value
        self.refresh_button_palette()

    def on_pick_color(self, i):
        self.annot.color_idx = int(i)
        self.refresh_button_palette()

    def refresh_button_palette(self):
        n = int(self.train.num_classes)

        if n > 0:
            self.annot.color_idx = max(0, min(int(self.annot.color_idx), n - 1))
        else:
            self.annot.color_idx = 0

        for i, b in enumerate(self.ui.button_palette):
            active = i < n
            b.set_enabled(active)

            if active:
                b.style(
                    f'opacity:1.0; filter:none; '
                    f'border:2px solid {"black" if i == self.annot.color_idx else "transparent"};'
                )
            else:
                b.style('opacity:0.25; filter:grayscale(100%); border:2px solid transparent;')

    def undo(self):
        self.slicer.undo()
        self.tracker.undo()
        self._request_slice_update()

    def redo(self):
        self.slicer.redo()
        self.tracker.redo()
        self._request_slice_update()

    def toggle_annotation_mode(self):
        if self.ui.toggle_annotation_mode.value == 0:
            self.annot.mode = 'draw'
        if self.ui.toggle_annotation_mode.value == 1:
            self.annot.mode = 'save'
        if self.ui.toggle_annotation_mode.value == 2:
            self.annot.mode = 'flood'

    def set_annotation_mode(self):
        if self.annot.mode == 'draw':
            self.ui.toggle_annotation_mode.value = 0
        if self.annot.mode == 'save':
            self.ui.toggle_annotation_mode.value = 1
        if self.annot.mode == 'flood':
            self.ui.toggle_annotation_mode.value = 2
    
    def update_brush_size(self, i):
        self.annot.brush_size = self.ui.slider_brush_size.value

    def set_brush_size(self):
        self.ui.slider_brush_size.value = min(self.annot.brush_size, self.ui.slider_brush_size._props.get('max'))

    def toggle_prediction_overlay(self):
        self.ui_state.prediction.visible = self.ui.checkbox_prediction_overlay.value
        self.renderer.update()

    def set_prediction_overlay(self):
        self.ui.checkbox_prediction_overlay.value = self.ui_state.prediction.visible
        self.renderer.update()

    def update_properties(self):
        origin = self.camera.origin
        u, v, w = self.camera.uvw
        zoom = 1 / self.camera.zoom
        volume_shape = self.slicer.shapes[0]

        def fmt(vec):
            return f'{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}'

        self.ui.label_origin.text = f'z: {origin[0]:.02f}, y: {origin[1]:.02f}, x: {origin[2]:.02f}'
        self.ui.label_rotation_u.text = (f'{fmt(u)}')
        self.ui.label_rotation_v.text = (f'{fmt(v)}')
        self.ui.label_rotation_w.text = (f'{fmt(w)}')
        self.ui.label_zoom.text = f'{zoom:.02f}'
        self.ui.label_shape.text = f'z: {volume_shape[0]}, y: {volume_shape[0]}, x: {volume_shape[0]}'

    def predict_volumes(self):
        predict.predict_all_volumes(self.data.zarr_files, self.data.project_path, num_classes=self.train.num_classes)
