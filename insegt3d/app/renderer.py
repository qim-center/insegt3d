import cv2
import base64
import threading
import numpy as np
from numba import njit


class ViewportRenderer:

    def __init__(self, state, scheduler):

        self.state = state
        self.scheduler = scheduler
        self.viewport = None
        self.overlay = None

        self.ui = self.state.ui

        self.image = self._init_zero_image()
        self.mask = self._init_zero_image()
        self.annotation = self._init_zero_image()
        self.prediction = self._init_zero_image()
        self.saved_prediction = self._init_zero_image()
        self.raw_prediction = None
        self.raw_image = None

        self._lock = threading.Lock()
        # (camera version, quality) of the shown image, so incoming images that compare lower are dropped
        self._stamp = (0, 0)
        # Camera version each overlay was computed for, so overlays from other versions are hidden
        self._versions = {}

    def _init_zero_image(self):
        return np.zeros(self.ui.viewport_shape + (3,), dtype=np.uint8)

    def clear(self, *names):
        for name in names:
            setattr(self, name, self._init_zero_image())
        if "prediction" in names:
            self.raw_prediction = None

    def update_svg(self, content):
        if self.overlay is not None:
            self.overlay.content = content

    def prediction_in_viewport(self, version):
        """Live prediction at viewport size, or None if it is hidden or not from camera `version`."""
        with self._lock:
            if self.raw_prediction is None or not self.ui.prediction.visible or self._versions.get("prediction") != version:
                return None
            return self._to_viewport_shape(self.raw_prediction, resize=False)

    def update(self, image=None, mask=None, annotation=None, prediction=None, saved_prediction=None, version=None, quality=0, fit=True):

        with self._lock:
            if version is None:
                version = self._stamp[0]

            if image is not None:
                if (version, quality) < self._stamp:
                    return
                self._stamp = (version, quality)
                self.raw_image = image
                self.image = self._process_image(image, fit=fit)
            if mask is not None:
                self.mask = self._process_mask(mask)
                self._versions["mask"] = version
            if annotation is not None:
                self.annotation = self._process_mask(annotation)
                self._versions["annotation"] = version
            if prediction is not None:
                self.raw_prediction = prediction
                self.prediction = self._process_mask(prediction, resize=False)
                self._versions["prediction"] = version
            if saved_prediction is not None:
                self.saved_prediction = self._process_mask(saved_prediction)
                self._versions["saved_prediction"] = version

            self._composite_and_render()

    def refresh_intensity_scaling(self):
        with self._lock:
            if self.raw_image is not None:
                self.image = self._process_image(self.raw_image)
            self._composite_and_render()

    def _composite_and_render(self):

        ui = self.state.ui
        image = self.image

        layers = [
            (getattr(self, name), overlay.alpha)
            for name, overlay in (("mask", ui.mask), ("annotation", ui.annotation), ("saved_prediction", ui.saved_prediction), ("prediction", ui.prediction))
            if overlay.visible and self._versions.get(name) == self._stamp[0] and getattr(self, name).shape[:2] == image.shape[:2]
        ]

        if layers:
            masks, alphas = zip(*layers)
            alphas_256 = np.round(np.asarray(alphas, dtype=np.float32) * 256).astype(np.int32)
            image = overlay_rgb_masks_numba(image, np.stack(masks), alphas_256)

        self._render_to_viewport(image)

    def _render_to_viewport(self, viewport_image):
        jpeg_quality = int(self.state.ui.jpeg_quality)
        bgr = cv2.cvtColor(viewport_image, cv2.COLOR_RGB2BGR)
        _, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        b64 = base64.b64encode(encoded).decode("ascii")
        self.scheduler.call_soon(self._set_source, f"data:image/jpeg;base64,{b64}")

    def _set_source(self, source):
        if self.viewport is not None:
            self.viewport.set_source(source)

    def _process_image(self, image, fit=True):
        low = self.state.ui.intensity_low
        high = self.state.ui.intensity_high

        image = image.astype(np.float32, copy=False)
        if high > low:
            image = (image - low) * (255.0 / (high - low))
        else:
            image = np.zeros_like(image)

        image = np.clip(image, 0, 255)
        image = np.rint(image).astype(np.uint8, copy=False)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if not fit:
            return image

        return self._to_viewport_shape(image, interpolation=cv2.INTER_LINEAR)

    def _process_mask(self, mask, resize=True):
        palette_rgb = self.state.annot.palette_rgb
        mask = palette_rgb[np.minimum(mask, palette_rgb.shape[0] - 1)]
        return self._to_viewport_shape(mask, resize=resize)

    def _to_viewport_shape(self, image, resize=True, interpolation=cv2.INTER_NEAREST):
        H, W = self.ui.viewport_shape

        if resize:
            return cv2.resize(image, (W, H), interpolation=interpolation)

        h, w = image.shape[:2]

        # Center crop, then center pad back up to the viewport shape
        y0, x0 = max(0, (h - H) // 2), max(0, (w - W) // 2)
        cropped = image[y0:y0 + min(H, h), x0:x0 + min(W, w)]

        out = np.zeros((H, W, *image.shape[2:]), dtype=image.dtype)
        py, px = (H - cropped.shape[0]) // 2, (W - cropped.shape[1]) // 2
        out[py:py + cropped.shape[0], px:px + cropped.shape[1]] = cropped

        return out

@njit(cache=True, nogil=True)
def overlay_rgb_masks_numba(image, masks, alphas_256):
    h, w, _ = image.shape
    n = masks.shape[0]
    out = image.copy()

    inv = np.empty(n, dtype=np.int32)
    for i in range(n):
        inv[i] = 256 - int(alphas_256[i])

    for y in range(h):
        for x in range(w):
            for i in range(n):
                if masks[i, y, x, 0] or masks[i, y, x, 1] or masks[i, y, x, 2]:
                    # Fixed-point alpha blend: alpha is in 1/256ths, +128 rounds before the shift
                    a = int(alphas_256[i])
                    ia = inv[i]
                    for k in range(3):
                        out[y, x, k] = (int(masks[i, y, x, k]) * a + int(out[y, x, k]) * ia + 128) >> 8

    return out
