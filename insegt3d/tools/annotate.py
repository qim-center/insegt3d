import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

        self._mask_exec = ThreadPoolExecutor(max_workers=1)
        self.scheduler.register_sync(
            "write_mask",
            fn=self._do_write_mask,
            spec=JobSpec(
                max_hz=60,
                mode="latest",
                executor=self._mask_exec,
                sequential_executor=False,
            ),
        )

    async def on_pointer(self, e):

        if e.mouse and e.wheel:
            factor = 1.1 ** (-1 if e.delta_y > 0 else 1)
            self.annot.brush_size *= factor
            self.callbacks.set_brush_size()

        if not e.ctrl and ((e.mouse and e.button == 0) or e.pen) and e.down:
            self.annot.annotating = True

        if self.annot.annotating:
            if self.annot.mode == 'draw':
                self._handle_draw(e)
            elif self.annot.mode == 'save':
                self._handle_save(e)

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

        self.renderer.update_svg(self._get_overlay())

    def _handle_draw(self, e):
        p = self.pointer

        if ((e.mouse and e.button == 0) or e.pen) and e.down:
            self._start_path(p.x, p.y, e.x, e.y)

        elif (e.mouse or e.pen) and e.move:
            self._continue_path(p.x, p.y, e.x, e.y)

        elif ((e.mouse and e.button == 0) or e.pen) and e.up:
            self._end_path(saving=False)
            self.annot.annotating = False

    def _handle_save(self, e):
        p = self.pointer

        if ((e.mouse and e.button == 0) or e.pen) and e.down:
            self._start_path(p.x, p.y, e.x, e.y)

        elif (e.mouse or e.pen) and e.move:
            self._continue_path(p.x, p.y, e.x, e.y)

        elif ((e.mouse and e.button == 0) or e.pen) and e.up:
            self._end_path(saving=True)
            self.annot.annotating = False

    def _start_path(self, x0, y0, x1, y1):
        self._continue_path(x0, y0, x1, y1)

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
        self.scheduler.request("write_mask", mask)
        self.path.clear()
        self.svg_parts.clear()

    def _get_overlay(self):
        p = self.pointer
        a = self.annot
        opacity = self.ui.annotation.alpha

        stroke = "".join(self.svg_parts)
        cursor = (
            f'<circle cx="{p.x}" cy="{p.y}" r="{a.brush_size/2}" '
            f'fill="{a.color_css}" stroke="{a.color_css}" opacity="{opacity}" />'
        )
        return f'<g opacity="{opacity}">{stroke}</g>{cursor}'

    def _do_write_mask(self, mask):

        if mask is None:
            return
            
        camera = self.state.camera
        h, w = self.nav.slice_shape
        half_h = h // 2
        half_w = w // 2
        extent = (0, 0, -half_h, half_h, -half_w, half_w)

        self.slicer.set_data(camera, mask, extent=extent)
        self.tracker.on_annotation_commit(
            self.slicer.zarr_path, camera, mask, extent
        )

        self.scheduler.request("nav_hires")
        self.scheduler.request("live_train")

    def _create_mask(self, path, saving=False):
        slice_h, slice_w = self.nav.slice_shape
        view_h, view_w = self.ui.viewport_shape

        scale_x = slice_w / view_w
        scale_y = slice_h / view_h

        mask = np.zeros((slice_h, slice_w), np.uint8)

        for i, (x0, y0, x1, y1, bs, idx) in enumerate(path):

            x0 = int(x0 * scale_x)
            x1 = int(x1 * scale_x)
            y0 = int(y0 * scale_y)
            y1 = int(y1 * scale_y)

            scale = 1

            r = int(np.rint(bs * scale * 0.5))
            t = int(np.rint(bs * scale))
            label = idx + 1

            cv2.circle(mask, (x0, y0), r, label, -1)
            cv2.line(mask, (x0, y0), (x1, y1), label, t)
            if i == len(path) - 1:
                cv2.circle(mask, (x1, y1), r, label, -1)

        if saving:
            pred = self.renderer._to_viewport_shape(
                self.renderer.raw_prediction, resize=False
            )
            print(self.renderer.raw_prediction.shape, (slice_w, slice_h), mask.shape)
            mask = (mask > 0) * pred

        return mask
