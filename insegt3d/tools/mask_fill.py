import cv2
import numpy as np

from insegt3d.tools.base_tool import BaseTool

class MaskFillTool(BaseTool):

    def __init__(self, state, services, renderer, scheduler, callbacks):
        super().__init__(state, services, renderer, scheduler, callbacks)

        self.annot = state.annot

        self.scheduler.register_sync("mask_fill", self._do_mask_fill, mode="queue")

    async def on_pointer(self, e):

        if not e.ctrl and self.annot.mode == 'mask_fill' and e.primary and e.down:
            self.scheduler.request("mask_fill", int(e.x), int(e.y))

    def _do_mask_fill(self, x, y):

        annotation = self._mask_fill(
            mask=self.renderer.mask,
            center_x=x,
            center_y=y,
            color_idx=self.annot.color_idx,
        )

        self.scheduler.request("write_mask", annotation)

    def _mask_fill(self, mask, center_x, center_y, color_idx):
        """Fills the same-coloured region around the seed, or returns None if it reaches the border (not enclosed)."""
        label = np.uint8(color_idx + 1)

        seed_color = mask[center_y, center_x]
        same = np.all(mask == seed_color, axis=2).astype(np.uint8)

        cv2.floodFill(same, None, (center_x, center_y), 2)

        filled = (same == 2)

        if filled[[0, -1]].any() or filled[:, [0, -1]].any():
            return None

        return filled * label
