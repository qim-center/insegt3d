import numpy as np
from concurrent.futures import ThreadPoolExecutor

from insegt3d.app.scheduler import JobSpec
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
        self._low_exec = ThreadPoolExecutor(max_workers=1)
        self._hires_exec = ThreadPoolExecutor(max_workers=1)

        self.scheduler.register_sync(
            "nav_preview",
            fn=self._do_lowres,
            spec=JobSpec(
                max_hz=60,
                mode="latest",
                executor=self._low_exec,
                sequential_executor=False,
            ),
        )

        self.scheduler.register_sync(
            "nav_hires",
            fn=self._do_hires,
            spec=JobSpec(
                max_hz=10,
                mode="latest",
                executor=self._hires_exec,
                sequential_executor=False,
            ),
        )

    def update_obliqueness(self, u):
        self.obliqueness = np.arccos(np.clip(np.max(np.abs(u)), -1.0, 1.0)) / np.arccos(1 / np.sqrt(3))

    async def on_pointer(self, e):

        s = self.state
        ui, annot, camera, p, nav = s.ui, s.annot, s.camera, s.pointer, s.nav
        
        # This is because navigation feels most when calling get_data at scale=512
        scale_factor = (min(nav.slice_shape) / 768.0)

        dx = (p.x - e.x) * scale_factor
        dy = (p.y - e.y) * scale_factor

        # Mouse navigation only when shift is held
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
                annot.brush_size /= zoom
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
                        annot.brush_size *= e.zoom_factor
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

    def _do_lowres(self):
        ui = self.state.ui

        # Save current visibility state
        prev_visibility = {
            'mask': ui.mask.visible,
            'annotation': ui.annotation.visible,
            'prediction': ui.prediction.visible,
        }

        # Hide overlays while navigating
        ui.mask.visible = False
        ui.annotation.visible = False
        ui.prediction.visible = False

        # Choose resolution
        level_modifier = 3 if self.obliqueness > 0.3 else 2
        image = self._extract_slice(level_modifier=level_modifier, order=1)

        # Update renderer
        self.renderer.update(image=image)

        # Update display information
        self.callbacks.update_properties()

        # Restore visibility
        ui.mask.visible = prev_visibility['mask']
        ui.annotation.visible = prev_visibility['annotation']
        ui.prediction.visible = prev_visibility['prediction']

    def _do_hires(self):
        ui = self.state.ui

        # Reset overlays
        self.renderer.clear_annotation()
        self.renderer.clear_prediction()

        rev = self._local_revision
        if self._cancelled(rev):
            return

        image = self._extract_slice(level_modifier=0, order=1)
        if self._cancelled(rev):
            return

        mask = self._extract_slice(level_modifier=0, mask=True)
        if self._cancelled(rev):
            return

        # Show mask overlays after hires update
        ui.mask.visible = True

        # Update renderer
        self.renderer.update(image=image, mask=mask)

        # Update display information
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

    def _extract_slice(self, level_modifier, order=0, mask=False):
        s = self.state
        camera, nav = s.camera, s.nav

        depth_start = 0
        depth_end = 0

        # if s.proj.projection is not None:
        #     half = s.proj.depth // 2
        #     depth_start = -half
        #     depth_end = half + 1

        h, w = nav.slice_shape

        half_h = h // 2
        half_w = w // 2

        extent = (depth_start, depth_end, -half_h, half_h, -half_w, half_w)

        image = self.services.slicer.get_data(
            camera,
            extent=extent,
            out_shape=self._get_output_shape(level_modifier),
            level_modifier=level_modifier,
            order=order,
            mask=mask,
            rescale=not mask
        )

        return image