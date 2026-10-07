import time
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

        self._next_properties = 0.0
        self._pending_steps = 0
        self._last_x = self._last_y = 0

        # nav_preview draws a quick, coarse frame while moving, and nav_hires refines it once the camera settles
        self.scheduler.register_sync("nav_preview", self._do_preview, max_hz=60, idle_after=0.1, idle_kwargs={"settled": True})
        self.scheduler.register_sync("nav_hires", self._do_hires, max_hz=10)
        self.scheduler.register_sync("nav_overlays", self._do_overlays, max_hz=30)
        self.scheduler.register_async("nav_step", self._do_step, max_hz=5)

    async def on_pointer(self, e):

        camera, nav = self.state.camera, self.state.nav

        # Navigation is tuned for a 768px slice, so keep the same feel at other sizes
        scale_factor = min(nav.slice_shape) / 768.0

        dx = (self._last_x - e.x) * scale_factor
        dy = (self._last_y - e.y) * scale_factor

        if e.mouse and e.ctrl and not e.shift and not e.alt:
            if e.down:
                self.panning = (e.button == 0)
                self.rotating = (e.button == 1)
                self.scrolling = (e.button == 2)
                if e.double and e.button == 0:
                    self._recenter(e.x, e.y)

            if e.move:
                if self.panning:
                    camera.pan(dx, dy)
                    self._request_preview()
                elif self.rotating:
                    camera.rotate(-dx, -dy)
                    self._request_preview()
                elif self.scrolling:
                    dz = np.hypot(dx, dy)
                    camera.scroll(dz if e.y < self._last_y else -dz)
                    self._request_preview()

            if e.wheel:
                direction = -1 if e.delta_y < 0 else 1
                zoom = 1.1 ** direction
                camera.zoom_by(zoom)
                self._request_hires()

            if e.up:
                self.panning = self.rotating = self.scrolling = False
                self._request_hires()

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
                self._request_hires()

        self.navigating = self.panning or self.rotating or self.scrolling or self.zooming
        self._last_x, self._last_y = e.x, e.y

    async def on_key(self, e):

        if e.action.keyup and not e.action.repeat:

            if e.key == "Space":
                self.state.camera.randomize()
                self._request_hires()

            if e.key == "Shift":
                self.panning = self.rotating = self.scrolling = self.zooming = self.navigating = False

        if e.action.keydown and not e.modifiers.ctrl and e.key in ("q", "a"):
            self._pending_steps += 1 if e.key == "q" else -1
            self.scheduler.request("nav_step")

        if e.action.keydown and not e.action.repeat and not e.modifiers.ctrl:
            if e.key in ("z", "y", "x"):
                self.callbacks.align_view(e.key.name)
            elif e.key in (",", "."):
                self.callbacks.step_plane(1 if e.key == "." else -1)
            elif e.key == "h":
                self.callbacks.toggle_crosshair()

    def _recenter(self, x, y):
        """Pans the view so the point at viewport pixel (x, y) is in the middle."""
        (slice_h, slice_w), (view_h, view_w) = self.state.nav.slice_shape, self.state.ui.viewport_shape
        self.state.camera.pan((x - view_w / 2) * slice_w / view_w, (y - view_h / 2) * slice_h / view_h)
        self._request_hires()

    def _request_preview(self):
        self.scheduler.request("nav_preview")
        self.scheduler.request("sync_navigator")

    def _request_hires(self):
        self.scheduler.request("nav_preview")
        self.scheduler.request("nav_hires")
        self.scheduler.request("sync_navigator")

    def _do_preview(self, settled=False):
        if self.services.slicer.images is None:
            return

        # Idle callback: the pointer is held still mid-navigation, so refine the frame
        if settled:
            if self.navigating:
                self.scheduler.request("nav_hires")
            return

        camera = self.state.camera.copy()
        # Previews are one level coarser (two when oblique) and show what arrives within 50 ms
        coarsen = 1 + int(np.max(np.abs(camera.u)) < 0.98)
        image, tiles = self._stream_image(camera, deadline=time.time() + 0.05, coarsen=coarsen)
        for _ in tiles:
            pass

        self.renderer.update(image=image, version=camera.version, fit=False)

        if time.time() >= self._next_properties:
            self._next_properties = time.time() + 0.1
            self.callbacks.update_properties()

    def _do_hires(self):

        camera = self.state.camera.copy()
        if self.services.slicer.images is None:
            return

        self.scheduler.request("nav_overlays")
        self.scheduler.request("live_predict")

        image, tiles = self._stream_image(camera)
        next_render = time.time() + 0.1

        for _ in tiles:
            if camera.version != self.state.camera.version:
                return
            if time.time() >= next_render:
                next_render = time.time() + 0.08
                self.renderer.update(image=image, version=camera.version, quality=1)

        self.renderer.update(image=image, version=camera.version, quality=1)
        self.callbacks.update_properties()

    def _do_overlays(self):
        slicer = self.services.slicer
        camera = self.state.camera.copy()
        nav = self.state.nav

        self.renderer.clear("annotation", "saved_prediction")

        mask = slicer.get_data(camera, extent=nav.extent, out_shape=nav.slice_shape, mask=True)
        saved_prediction = None
        if slicer.has_prediction:
            saved_prediction = slicer.get_data(camera, extent=nav.extent, out_shape=nav.slice_shape, prediction=True)

        self.renderer.update(mask=mask, saved_prediction=saved_prediction, version=camera.version)

    async def _do_step(self):
        steps, self._pending_steps = self._pending_steps, 0
        slicer, camera = self.services.slicer, self.state.camera
        if slicer.images is None:
            return
        camera.step(steps * np.linalg.norm(slicer.voxel_sizes[0] * camera.u))
        self._request_hires()

    def _stream_image(self, camera, deadline=None, coarsen=0):
        slicer = self.services.slicer

        # 0 when axis-aligned, 1 along a cube diagonal
        obliqueness = np.arccos(np.clip(np.max(np.abs(camera.u)), -1.0, 1.0)) / np.arccos(1 / np.sqrt(3))

        keep_level = slicer.level(camera.zoom, 3 if obliqueness > 0.3 else 2)
        target = slicer.level(camera.zoom, coarsen)
        # Coarse to fine: the in-memory fallback, then each level down to the target
        fallback = [slicer.fallback] if target < slicer.fallback[0] else []
        sources = fallback + slicer.sources(range(max(slicer.fallback[0] - 1, target), target - 1, -1))

        tile_hw = int(np.clip(256 / (1 + 4 * obliqueness), 96, 256))

        h, w = self.state.nav.slice_shape
        image = np.zeros((max(1, h >> coarsen), max(1, w >> coarsen)), dtype=np.float32)
        tiles = slicer.stream(camera, image[None], sources, self.state.nav.extent, keep_level=keep_level, deadline=deadline, order=1, tile_hw=tile_hw)

        return image, tiles
