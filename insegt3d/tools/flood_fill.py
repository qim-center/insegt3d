import cv2
import numpy as np

from insegt3d.tools.base_tool import BaseTool

class FloodFillTool(BaseTool):

    def __init__(self, state, services, renderer, scheduler, callbacks, annotator):
        super().__init__(state, services, renderer, scheduler, callbacks)

        # State references
        self.ui = state.ui
        self.annot = state.annot
        self.annotator = annotator

        self.flood_x = None
        self.flood_y = None
        self.tolerance = 10

        self.register_latest_job("flood_fill", self._do_flood_fill)

    async def on_pointer(self, e):

        if not e.ctrl and ((e.mouse and e.button == 0) or e.pen) and e.down:
            self.annot.annotating = True

        if self.annot.annotating and self.annot.mode == 'flood':
            self._handle_flood_fill(e)

    def _handle_flood_fill(self, e):
        filling = (e.mouse and e.button == 0) or e.pen

        if filling and e.down:
            self.flood_x = e.x
            self.flood_y = e.y

        elif (e.mouse or e.pen) and e.move:
            # Drag distance from the seed point sets the fill tolerance
            dx = e.x - self.flood_x
            dy = e.y - self.flood_y
            self.tolerance = (dx * dx + dy * dy) ** 0.5
            self.scheduler.request("flood_fill")

        elif filling and e.up:
            self.annot.annotating = False
            self.scheduler.request("flood_fill", True)

    def _do_flood_fill(self, write_to_volume=False):

        annotation = self._flood_fill(
            image=self.renderer.image,
            center_x=int(self.flood_x),
            center_y=int(self.flood_y),
            radius=int(self.annot.brush_size // 2),
            color_idx=self.annot.color_idx,
            tolerance=self.tolerance
        )

        # Show annotation overlay while running flood fill tool
        self.ui.annotation.visible = True

        self.renderer.update(annotation=annotation)

        if write_to_volume:
            self.ui.annotation.visible = False
            self.scheduler.request(self.annotator.write_mask_job, annotation)

    def _flood_fill(self, image, center_x, center_y, radius, color_idx, tolerance=10, fill_holes=True, kernel_size=7):
        """
        Flood fills outwards from a seed disc, scaling the tolerance by the
        local texture (intensity spread) under that disc.
        """
        H, W = image.shape[:2]
        label = np.uint8(color_idx + 1)

        # OpenCV floodFill mask must be (H+2, W+2)
        ff_mask = np.zeros((H + 2, W + 2), np.uint8)

        seed = np.zeros((H, W), np.uint8)
        cv2.circle(seed, (center_x, center_y), radius, 255, -1)
        ys, xs = np.where(seed)

        if ys.size == 0:
            return np.zeros((H, W), np.uint8)

        std = float(image[ys, xs].std(axis=0).mean())
        tol = (float(tolerance) * std / 100.0,) * 3

        # Mask-only flood fill; it writes into ff_mask (offset by 1)
        cv2.floodFill(
            image, ff_mask, (center_x, center_y), (0, 0, 0),
            tol, tol, flags=cv2.FLOODFILL_MASK_ONLY
        )

        filled = ff_mask[1:-1, 1:-1] != 0

        if fill_holes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            filled = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

        out = np.zeros((H, W), np.uint8)
        out[filled] = label

        return out
