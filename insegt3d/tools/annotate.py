import cv2
import numpy as np

from insegt3d.app.scheduler import JobSpec
from insegt3d.tools.base_tool import BaseTool

class AnnotatorTool(BaseTool):

    def __init__(self, state, services, renderer, scheduler, callbacks):
        super().__init__(state, services, renderer, scheduler, callbacks)

        # State references
        self.ui = state.ui
        self.annot = state.annot
        self.pointer = state.pointer
        self.nav = state.nav
        self.camera = state.camera
        self.train = state.train
        self.slicer = services.slicer
        self.tracker = services.tracker

        self._prev_mode = None

        self.path = []
        self.svg_parts = []

        self._cursor_x = 0
        self._cursor_y = 0

        self.write_mask_job = "write_mask"

        self.register_latest_job(self.write_mask_job, self._do_write_mask)

        self.scheduler.register_async(
            "update_annotation_overlay",
            fn=self._push_overlay,
            spec=JobSpec(
                max_hz=60,
                mode="latest",
            ),
        )

    async def on_pointer(self, e):

        if e.mouse and e.wheel and not e.ctrl:
            factor = 1.1 ** (-1 if e.delta_y > 0 else 1)
            self.annot.brush_size *= factor
            self.callbacks.set_brush_size()

        if not e.ctrl and ((e.mouse and e.button == 0) or e.pen) and e.down:
            self.annot.annotating = True

        if self.annot.annotating and self.annot.mode in ('draw', 'save'):
            self._handle_stroke(e, saving=self.annot.mode == 'save')

        self._cursor_x, self._cursor_y = e.x, e.y
        self.scheduler.request("update_annotation_overlay")

    async def _push_overlay(self):
        self.renderer.update_svg(self._get_overlay())

    async def on_key(self, e):
        if e.action.repeat:
            return

        if e.action.keydown and e.key == "Shift":
            self._prev_mode = self.annot.mode
            self.annot.mode = 'save'
            self.callbacks.set_annotation_mode()
        if e.action.keyup and e.key == "Shift":
            self.annot.mode = self._prev_mode
            self.callbacks.set_annotation_mode()
            self._prev_mode = None

        if e.modifiers.ctrl and e.action.keydown:
            if e.key == "z":
                self.slicer.undo()
                self.tracker.undo()
            elif e.key == "y":
                self.slicer.redo()
                self.tracker.redo()
            self.scheduler.request("nav_hires")
            return

        if e.key == "x" and e.action.keydown:
            self.annot.previous_color(self.train.num_classes)
            self.callbacks.refresh_button_palette()
        elif e.key == "c" and e.action.keydown:
            self.annot.next_color(self.train.num_classes)
            self.callbacks.refresh_button_palette()

        if e.key == "d" and e.action.keydown:
            self.ui.prediction.visible = not self.ui.prediction.visible
            self.callbacks.set_prediction_overlay()
            self.renderer.update()

        self.renderer.update_svg(self._get_overlay())

    def _handle_stroke(self, e, saving):
        
        p = self.pointer
        drawing = (e.mouse and e.button == 0) or e.pen

        if (drawing and e.down) or ((e.mouse or e.pen) and e.move):
            self._continue_path(p.x, p.y, e.x, e.y)

        elif drawing and e.up:
            self._end_path(saving=saving)
            self.annot.annotating = False

    def _continue_path(self, x0, y0, x1, y1):
        a = self.annot
        r = a.brush_size * 0.5
        css = a.color_css

        self.path.append([x0, y0, x1, y1, a.brush_size, a.color_idx])
        self.svg_parts.append(
            f'<circle cx="{x0}" cy="{y0}" r="{r}" fill="{css}" stroke="{css}" />'
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
            f'stroke="{css}" stroke-width="{a.brush_size}" fill="none" />'
        )

    def _end_path(self, saving):
        mask = self._create_mask(self.path, saving=saving)
        self.scheduler.request(self.write_mask_job, mask)
        self.path.clear()
        self.svg_parts.clear()

    def _get_overlay(self):
        a = self.annot
        opacity = self.ui.annotation.alpha

        stroke = "".join(self.svg_parts)
        cursor = (
            f'<circle cx="{self._cursor_x}" cy="{self._cursor_y}" r="{a.brush_size/2}" '
            f'fill="{a.color_css}" stroke="{a.color_css}" opacity="{opacity}" />'
        )
        return f'<g opacity="{opacity}">{stroke}</g>{cursor}'

    def _do_write_mask(self, mask):

        if mask is None:
            return

        h, w = self.nav.slice_shape
        half_h, half_w = h // 2, w // 2
        extent = (0, 0, -half_h, half_h, -half_w, half_w)

        self.slicer.set_data(self.camera, mask, extent=extent)
        self.tracker.on_annotation_commit(
            self.slicer.zarr_path, self.camera, mask, extent
        )

        self.scheduler.request("nav_hires")
        self.scheduler.request("live_train")

    def _create_mask(self, path, saving=False):
        slice_h, slice_w = self.nav.slice_shape
        view_h, view_w = self.ui.viewport_shape

        scale_x = slice_w / view_w
        scale_y = slice_h / view_h

        mask = np.zeros((slice_h, slice_w), np.uint8)

        for i, (x0, y0, x1, y1, brush_size, color_idx) in enumerate(path):

            x0 = int(x0 * scale_x)
            x1 = int(x1 * scale_x)
            y0 = int(y0 * scale_y)
            y1 = int(y1 * scale_y)

            radius = int(np.rint(brush_size * 0.5))
            thickness = int(np.rint(brush_size))
            label = color_idx + 1

            cv2.circle(mask, (x0, y0), radius, label, -1)
            cv2.line(mask, (x0, y0), (x1, y1), label, thickness)
            if i == len(path) - 1:
                cv2.circle(mask, (x1, y1), radius, label, -1)

        if saving:
            mask = (mask > 0) * self.renderer.prediction_in_viewport()

        return mask
