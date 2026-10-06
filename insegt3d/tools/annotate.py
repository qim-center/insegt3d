import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from insegt3d.tools.base_tool import BaseTool
from insegt3d.volume.interp import ERASE

BRUSH_MODES = ('draw', 'erase', 'keep')
MODE_KEYS = {'b': 'draw', 'e': 'erase', 'g': 'mask_fill', 'f': 'flood', 'k': 'keep'}

class AnnotatorTool(BaseTool):

    def __init__(self, state, services, renderer, scheduler, callbacks):
        super().__init__(state, services, renderer, scheduler, callbacks)

        self.ui = state.ui
        self.annot = state.annot
        self.nav = state.nav
        self.camera = state.camera
        self.train = state.train
        self.history = services.history

        self._prev_mode = None
        self._erasing = False

        self.strokes = []

        self._cursor_x = 0
        self._cursor_y = 0
        self._overlay_hidden = False

        edits = ThreadPoolExecutor(max_workers=1)
        self.scheduler.register_sync("write_mask", self._do_write_mask, mode="queue", executor=edits)
        self.scheduler.register_sync("undo", self._do_undo, mode="queue", executor=edits)
        self.scheduler.register_sync("redo", self._do_redo, mode="queue", executor=edits)
        self.scheduler.register_async("update_annotation_overlay", self._push_overlay, max_hz=60)

    async def on_pointer(self, e):

        if e.mouse and e.wheel and not e.ctrl:
            factor = 1.1 ** (-1 if e.delta_y > 0 else 1)
            self.annot.brush_size *= factor
            self.callbacks.set_brush_size()

        if not e.ctrl and self.annot.mode in BRUSH_MODES and (e.primary or e.eraser) and e.down:
            self.annot.annotating = True
            self._erasing = e.eraser or self.annot.mode == 'erase'

        if self.annot.annotating and self.annot.mode in BRUSH_MODES:
            self._handle_stroke(e, keeping=self.annot.mode == 'keep' and not self._erasing)

        self._cursor_x, self._cursor_y = e.x, e.y

        # The brush overlay is hidden while ctrl (navigation) is held
        if not e.ctrl or e.ctrl != self._overlay_hidden:
            self.scheduler.request("update_annotation_overlay")
        self._overlay_hidden = bool(e.ctrl)

    async def _push_overlay(self):
        self.renderer.update_svg(self._get_overlay())

    async def on_key(self, e):
        if e.action.repeat:
            return

        if e.action.keydown and e.key == "Shift":
            self._prev_mode = self.annot.mode
            self.annot.mode = 'keep'
            self._abandon_stroke()
        if e.action.keyup and e.key == "Shift" and self._prev_mode is not None:
            self.annot.mode = self._prev_mode
            self._abandon_stroke()
            self._prev_mode = None

        if e.modifiers.ctrl and e.action.keydown:
            if e.key == "z":
                self.scheduler.request("undo")
            elif e.key == "y":
                self.scheduler.request("redo")
            return

        if e.key == "c" and e.action.keydown:
            self.annot.next_color(self.train.num_classes)
            self.callbacks.refresh_button_palette()
        elif e.key.name in MODE_KEYS and e.action.keydown:
            self.annot.mode = MODE_KEYS[e.key.name]
            self._abandon_stroke()
        # Keys 1-9 select classes 1-9, 0 selects class 10
        elif e.key.number is not None and e.action.keydown and (e.key.number - 1) % 10 < self.train.num_classes:
            self.callbacks.on_pick_color((e.key.number - 1) % 10)

        if e.key == "d" and e.action.keydown:
            self.callbacks.toggle_live_prediction()

        self.renderer.update_svg(self._get_overlay())

    def _handle_stroke(self, e, keeping):

        drawing = e.primary or e.eraser

        if (drawing and e.down) or ((e.mouse or e.pen) and e.move):
            self._add_point(e.x, e.y)

        elif drawing and e.up:
            self._end_path(keeping=keeping)
            self.annot.annotating = False

    def _abandon_stroke(self):
        self.annot.annotating = False
        self.strokes.clear()

    def _add_point(self, x, y):
        a = self.annot
        label = ERASE if self._erasing else a.color_idx + 1
        # Start a new segment when brush size or class changes, continuing from the last point
        if not self.strokes or self.strokes[-1][:2] != (a.brush_size, label):
            start = self.strokes[-1][2][-1] if self.strokes else (x, y)
            self.strokes.append((a.brush_size, label, [start]))
        self.strokes[-1][2].append((x, y))

    def _end_path(self, keeping):
        mask = self._create_mask(self.strokes, keeping=keeping)
        self.scheduler.request("write_mask", mask, self.camera.copy())
        self.strokes.clear()

    def _get_overlay(self):
        if self._overlay_hidden:
            return ''

        a = self.annot
        opacity = self.ui.annotation.alpha

        stroke = "".join(
            f'<polyline points="{" ".join(f"{x},{y}" for x, y in points)}" fill="none" '
            f'stroke="{"white" if label == ERASE else a.colors[label - 1]}" stroke-width="{size}" stroke-linecap="round" stroke-linejoin="round" />'
            for size, label, points in self.strokes
        )
        color = 'white' if a.mode == 'erase' else a.colors[a.color_idx]
        cursor = (
            f'<circle cx="{self._cursor_x}" cy="{self._cursor_y}" r="{a.brush_size/2}" '
            f'fill="{color}" stroke="{color}" opacity="{opacity}" />'
        )
        return f'<g opacity="{opacity}">{stroke}</g>{cursor}'

    def _do_write_mask(self, mask, camera=None):

        if mask is None or self.train.changing_classes:
            return

        if camera is None:
            camera = self.camera.copy()

        self.history.commit(camera, mask, self.nav.extent)

        self.scheduler.request("nav_overlays")
        self.scheduler.request("live_train")

    def _do_undo(self):
        self.history.undo()
        self.scheduler.request("nav_overlays")

    def _do_redo(self):
        self.history.redo()
        self.scheduler.request("nav_overlays")

    def _create_mask(self, strokes, keeping=False):
        slice_h, slice_w = self.nav.slice_shape
        view_h, view_w = self.ui.viewport_shape

        scale = np.array([slice_w / view_w, slice_h / view_h])

        mask = np.zeros((slice_h, slice_w), np.uint8)

        for brush_size, label, points in strokes:
            points = (np.array(points) * scale).astype(np.int32)
            cv2.polylines(mask, [points], False, label, max(1, int(np.rint(brush_size * scale[1]))))

        # Keep mode paints the live prediction under the stroke instead of the class
        if keeping:
            prediction = self.renderer.prediction_in_viewport(self.camera.version)
            if prediction is None:
                return None
            mask = (mask > 0) * cv2.resize(prediction, (slice_w, slice_h), interpolation=cv2.INTER_NEAREST)

        return mask
