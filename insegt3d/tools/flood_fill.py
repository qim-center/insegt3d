import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from insegt3d.tools.base_tool import BaseTool

class FloodFillTool(BaseTool):

    def __init__(self, state, services, renderer, scheduler, callbacks):
        super().__init__(state, services, renderer, scheduler, callbacks)

        self.ui = state.ui
        self.annot = state.annot

        self.flood_x = None
        self.flood_y = None
        self.tolerance = 10

        fills = ThreadPoolExecutor(max_workers=1)
        self.scheduler.register_sync("flood_fill", self._do_flood_fill, max_hz=60, executor=fills)
        self.scheduler.register_sync("commit_flood_fill", self._do_flood_fill, mode="queue", executor=fills)

    async def on_pointer(self, e):

        if not e.ctrl and self.annot.mode == 'flood' and e.primary and e.down:
            self.annot.annotating = True

        if self.annot.annotating and self.annot.mode == 'flood':
            self._handle_flood_fill(e)

    def _handle_flood_fill(self, e):
        if e.primary and e.down:
            self.flood_x = e.x
            self.flood_y = e.y
            self.tolerance = 10

        elif (e.mouse or e.pen) and e.move:
            # Drag distance from the seed point sets the fill tolerance
            dx = e.x - self.flood_x
            dy = e.y - self.flood_y
            self.tolerance = (dx * dx + dy * dy) ** 0.5
            self.scheduler.request("flood_fill", self.flood_x, self.flood_y, self.tolerance)

        elif e.primary and e.up:
            self.annot.annotating = False
            self.scheduler.request("commit_flood_fill", self.flood_x, self.flood_y, self.tolerance, write_to_volume=True)

    def _do_flood_fill(self, x, y, tolerance, write_to_volume=False):

        scale = self.renderer.image.shape[0] / self.ui.viewport_shape[0]

        annotation = self._flood_fill(
            image=self.renderer.image,
            center_x=int(x * scale),
            center_y=int(y * scale),
            radius=int(self.annot.brush_size // 2 * scale),
            color_idx=self.annot.color_idx,
            tolerance=tolerance
        )

        self.ui.annotation.visible = True

        self.renderer.update(annotation=annotation)

        if write_to_volume:
            self.ui.annotation.visible = False
            self.scheduler.request("write_mask", annotation)

    def _flood_fill(self, image, center_x, center_y, radius, color_idx, tolerance):
        """Flood fills from a seed disc, with the tolerance scaled by the intensity spread under it."""
        H, W = image.shape[:2]
        label = np.uint8(color_idx + 1)

        # OpenCV's flood fill mask has a 1 pixel border
        ff_mask = np.zeros((H + 2, W + 2), np.uint8)

        seed = np.zeros((H, W), np.uint8)
        cv2.circle(seed, (center_x, center_y), radius, 255, -1)
        ys, xs = np.where(seed)

        if ys.size == 0:
            return np.zeros((H, W), np.uint8)

        std = float(image[ys, xs].std(axis=0).mean())
        tol = (float(tolerance) * std / 100.0,) * 3

        cv2.floodFill(
            image, ff_mask, (center_x, center_y), (0, 0, 0),
            tol, tol, flags=cv2.FLOODFILL_MASK_ONLY
        )

        filled = ff_mask[1:-1, 1:-1] != 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        filled = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        return filled * label
