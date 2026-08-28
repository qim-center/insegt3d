import numpy as np

from insegt3d.tools.base_tool import BaseTool

class NavigatorTool(BaseTool):

    def __init__(self, state, services, renderer, scheduler, callbacks):
        super().__init__(state, services, renderer, scheduler, callbacks)

        self.panning = False
        self.rotating = False
        self.scrolling = False
        self.zooming = False
        self.navigating = False

        self._local_revision = 0
        self.obliqueness = 0.0

        # Dedicated lanes (sequential per job)
        self.register_latest_job("nav_preview", self._do_preview, max_hz=60)
        self.register_latest_job("nav_hires", self._do_hires, max_hz=10)

    def update_obliqueness(self, u):
        self.obliqueness = np.arccos(np.clip(np.max(np.abs(u)), -1.0, 1.0)) / np.arccos(1 / np.sqrt(3))

    async def on_pointer(self, e):

        s = self.state
        ui, camera, p, nav = s.ui, s.camera, s.pointer, s.nav

        # Navigation is tuned for a 768px slice; keep the feel at other sizes
        scale_factor = min(nav.slice_shape) / 768.0

        dx = (p.x - e.x) * scale_factor
        dy = (p.y - e.y) * scale_factor

        # Mouse navigation only while ctrl is held
        if e.mouse and e.ctrl and not e.shift and not e.alt:
            if e.down:
                self.panning = (e.button == 0)
                self.rotating = (e.button == 1)
                self.scrolling = (e.button == 2)

            if e.move:
                if self.panning:
                    camera.pan(dx, dy)
                    self._request_preview()
                elif self.rotating:
                    camera.rotate(-dx, -dy)
                    self.update_obliqueness(camera.u)
                    self._request_preview()
                elif self.scrolling:
                    dz = np.hypot(dx, dy)
                    camera.scroll(dz if e.y < p.y else -dz)
                    self._request_preview()

            if e.wheel:
                direction = -1 if e.delta_y < 0 else 1
                zoom = 1.1 ** direction
                camera.zoom_by(zoom)
                self._request_preview()
                self._request_hires()

            if e.up:
                self.panning = self.rotating = self.scrolling = False
                self._request_preview()
                self._request_hires()

        # Touch navigation
        elif e.touch:
            if e.down:
                self.panning = e.one_finger
                self.rotating = e.two_finger
                self.zooming = e.two_finger

            if e.move:
                if e.one_finger and self.panning:
                    camera.pan(dx, dy)
                    self._request_preview()

                if e.two_finger:
                    if self.rotating:
                        camera.rotate_axis("u", e.rotation_rad)
                        self._request_preview()
                    if self.zooming:
                        camera.zoom_by(1 / e.zoom_factor)
                        self._request_preview()

            if e.up:
                self.panning = self.rotating = self.scrolling = self.zooming = False
                self._request_preview()
                self._request_hires()

        ui.show_orientation = self.rotating
        self.navigating = self.panning or self.rotating or self.scrolling or self.zooming

    async def on_key(self, e):

        if e.action.keyup and not e.action.repeat:

            if e.key == "Space":
                self.state.camera.randomize()
                self._request_hires()

            if e.key == "Shift":
                self.panning = self.rotating = self.scrolling = self.zooming = self.navigating = False

    def _request_preview(self):
        self.scheduler.request("nav_preview")
        self.scheduler.request("sync_navigator")

    def _request_hires(self):
        self._local_revision = self.state.nav.bump()
        self.scheduler.request("nav_hires")
        self.scheduler.request("sync_navigator")

    def _do_preview(self):
        ui = self.state.ui
        overlays = (ui.mask, ui.annotation, ui.prediction, ui.saved_prediction)

        # Hide overlays while navigating, then restore them afterwards
        was_visible = [overlay.visible for overlay in overlays]
        for overlay in overlays:
            overlay.visible = False

        # Coarser level the further the view is from an axis-aligned slice
        level_modifier = 3 if self.obliqueness > 0.3 else 2

        self.renderer.update(image=self._extract_slice(level_modifier=level_modifier, order=1))
        self.callbacks.update_properties()

        for overlay, visible in zip(overlays, was_visible):
            overlay.visible = visible

    def _do_hires(self):

        # Reset overlays
        self.renderer.clear_annotation()
        self.renderer.clear_prediction()
        self.renderer.clear_saved_prediction()

        rev = self._local_revision
        if self._cancelled(rev):
            return

        image = self._extract_slice(level_modifier=0, order=1)
        if self._cancelled(rev):
            return

        mask = self._extract_slice(level_modifier=0, mask=True)
        if self._cancelled(rev):
            return

        saved_prediction = None
        if self.services.slicer.has_prediction:
            saved_prediction = self._extract_slice(level_modifier=0, prediction=True)
            if self._cancelled(rev):
                return

        self.renderer.update(image=image, mask=mask, saved_prediction=saved_prediction)
        self.callbacks.update_properties()

        self.scheduler.request("live_predict")

    def _cancelled(self, rev):
        return self.navigating or (rev != self.state.nav.revision)

    def _get_output_shape(self, level_modifier):
        """
        Computes proper output_shape depending on level modifier.
        Uses nav.slice_shape = (h, w).
        """
        h, w = self.state.nav.slice_shape

        scaled_h = max(1, int(round(h / (2 ** level_modifier))))
        scaled_w = max(1, int(round(w / (2 ** level_modifier))))

        return (scaled_h, scaled_w)

    def _extract_slice(self, level_modifier, order=0, mask=False, prediction=False):
        h, w = self.state.nav.slice_shape
        half_h, half_w = h // 2, w // 2
        extent = (0, 0, -half_h, half_h, -half_w, half_w)

        return self.services.slicer.get_data(
            self.state.camera,
            extent=extent,
            out_shape=self._get_output_shape(level_modifier),
            level_modifier=level_modifier,
            order=order,
            mask=mask,
            prediction=prediction,
        )
