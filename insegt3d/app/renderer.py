import cv2
import base64
import numpy as np
from numba import njit, prange


class ViewportRenderer:
    
    def __init__(self, state):

        self.state = state
        self.viewport = None

        # State references
        self.ui = self.state.ui

        self.image = self._init_zero_image()
        self.mask = self._init_zero_image()
        self.annotation = self._init_zero_image()
        self.prediction = self._init_zero_image()
        self.raw_prediction = self._init_zero_image()[:,:,0]
        self.viewport_image = self._init_zero_image()
    
    def _init_zero_image(self):
        return np.zeros(self.ui.viewport_shape + (3,), dtype=np.uint8)

    def clear_image(self):
        self.image = self._init_zero_image()

    def clear_mask(self):
        self.mask = self._init_zero_image()

    def clear_annotation(self):
        self.annotation = self._init_zero_image()

    def clear_prediction(self):
        self.prediction = self._init_zero_image()

    def clear_viewport_image(self):
        self.viewport_image = self._init_zero_image()

    def update_svg(self, content):
        self.viewport.content = content

    def update(self, image=None, mask=None, annotation=None, prediction=None):

        # Get some state variables
        ui = self.state.ui
        camera = self.state.camera

        # Check inputs
        if image is not None:
            self.image = self._process_image(image)
        if mask is not None:
            self.mask = self._process_mask(mask)
        if annotation is not None:
            self.annotation = self._process_mask(annotation)
        if prediction is not None:
            self.raw_prediction = prediction.copy()
            self.prediction = self._process_mask(prediction, resize=False)

        # Build overlay_list and alpha_list
        overlay_list = []
        alpha_list = []

        if ui.mask.visible:
            overlay_list.append(self.mask)
            alpha_list.append(ui.mask.alpha)

        if ui.annotation.visible:
            overlay_list.append(self.annotation)
            alpha_list.append(ui.annotation.alpha)

        if ui.prediction.visible:
            overlay_list.append(self.prediction)
            alpha_list.append(ui.prediction.alpha)

        # Build overlay
        viewport_image = self._build_overlay(self.image, overlay_list, alpha_list)

        # Draw orientation widget
        if ui.show_orientation:
            viewport_image = self._draw_orientation_widget(
                viewport_image,
                camera.uvw,
                viewport_image.shape[:2],
            )

        # Render to viewport
        self._render_to_viewport(viewport_image)

    def _build_overlay(self, image, overlay_list=None, alpha_list=None):

        if not overlay_list or not alpha_list:
            return image

        masks = np.stack(overlay_list, axis=0)

        alphas = np.asarray(alpha_list, dtype=np.float32)
        alphas_256 = np.round(alphas * 256).astype(np.int32)

        return overlay_rgb_masks_numba(image, masks, alphas_256)

    def _render_to_viewport(self, viewport_image):
        jpeg_quality = int(self.state.ui.jpeg_quality)
        bgr = cv2.cvtColor(viewport_image, cv2.COLOR_RGB2BGR)
        _, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        b64 = base64.b64encode(encoded).decode("ascii")
        self.viewport.set_source(f"data:image/jpeg;base64,{b64}")
        self.viewport_image = viewport_image

    def _process_image(self, image):
        image = np.rint(image).astype(np.uint8, copy=False)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return self._to_viewport_shape(image)

    def _process_mask(self, mask, resize=True):
        mask = np.rint(mask).astype(np.int32, copy=False)
        palette_rgb = self.state.annot.palette_rgb
        mask = palette_rgb[np.clip(mask, 0, palette_rgb.shape[0] - 1)]
        return self._to_viewport_shape(mask.astype(np.uint8, copy=False), resize=resize)

    def _to_viewport_shape(self, image, resize=True):
        H, W = self.ui.viewport_shape

        if resize:
            return cv2.resize(image, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            h, w = image.shape[:2]

            # center crop
            y0, x0 = max(0, (h - H)//2), max(0, (w - W)//2)
            cropped = image[y0:y0+min(H,h), x0:x0+min(W,w)]

            # center pad
            out = np.zeros((H, W, *image.shape[2:]), dtype=image.dtype)
            py, px = (H - cropped.shape[0])//2, (W - cropped.shape[1])//2
            out[py:py+cropped.shape[0], px:px+cropped.shape[1]] = cropped

            return out

    def _draw_orientation_widget(self, image, vectors, shape):
        H, W = shape
        cx, cy = W // 2, H // 2
        scale = min(H, W) * 0.4

        # cv2 draws in BGR; but image is RGB.
        # Use RGB colors here (so it looks correct in RGB space).
        colors_rgb = ((255, 0, 0), (0, 255, 0), (0, 0, 255))

        for (_vz, vy, vx), color in zip(vectors, colors_rgb):
            x = int(cx + vx * scale)
            y = int(cy - vy * scale)
            cv2.arrowedLine(image, (cx, cy), (x, y), color, 2, tipLength=0.2)

        return image

@njit(parallel=True)
def overlay_rgb_masks_numba(image, masks, alphas_256):
    h, w, _ = image.shape
    n = masks.shape[0]
    out = image.copy()

    inv = np.empty(n, dtype=np.int32)
    for i in range(n):
        inv[i] = 256 - int(alphas_256[i])

    for y in prange(h):
        for x in range(w):
            for i in range(n):
                if masks[i, y, x, 0] or masks[i, y, x, 1] or masks[i, y, x, 2]:
                    a = int(alphas_256[i])
                    ia = inv[i]
                    for k in range(3):
                        out[y, x, k] = (int(masks[i, y, x, k]) * a + int(out[y, x, k]) * ia + 128) >> 8

    return out
