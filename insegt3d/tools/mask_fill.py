import cv2
import numpy as np

from insegt3d.tools.base_tool import BaseTool

class MaskFillTool(BaseTool):

    def __init__(self, state, services, renderer, scheduler, callbacks, annotator):
        super().__init__(state, services, renderer, scheduler, callbacks)

        # State references
        self.annot = state.annot
        self.annotator = annotator

        self.fill_x = None
        self.fill_y = None

        self.register_latest_job("mask_fill", self._do_mask_fill)

    async def on_pointer(self, e):

        if not e.ctrl and self.annot.mode == 'mask_fill':
            self._handle_mask_fill(e)

    def _handle_mask_fill(self, e):
        if ((e.mouse and e.button == 0) or e.pen) and e.down:
            self.fill_x = e.x
            self.fill_y = e.y
            self.scheduler.request("mask_fill")

    def _do_mask_fill(self):

        if self.annot.mode != 'mask_fill':
            return

        if self.fill_x is None or self.fill_y is None:
            return

        annotation = self._mask_fill(
            mask=self.renderer.mask,
            center_x=int(self.fill_x),
            center_y=int(self.fill_y),
            color_idx=self.annot.color_idx,
        )

        self.scheduler.request(self.annotator.write_mask_job, annotation)

    def _mask_fill(self, mask, center_x, center_y, color_idx):
        """
        Fills the connected region of same-coloured mask pixels around the
        seed point. Returns None when that region reaches the image border,
        which means it isn't actually enclosed by annotations.
        """
        H, W = mask.shape[:2]
        label = np.uint8(color_idx + 1)

        ff_mask = np.zeros((H + 2, W + 2), np.uint8)

        seed_color = mask[center_y, center_x]
        same = np.all(mask == seed_color, axis=2).astype(np.uint8)

        cv2.floodFill(same, ff_mask, (center_x, center_y), 2)

        filled = (same == 2)

        touches_border = (
            np.any(filled[0, :]) or
            np.any(filled[-1, :]) or
            np.any(filled[:, 0]) or
            np.any(filled[:, -1])
        )

        if touches_border:
            return None

        out = np.zeros((H, W), np.uint8)
        out[filled] = label

        return out
