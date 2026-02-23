import numpy as np
from nicegui import ui
from concurrent.futures import ThreadPoolExecutor
from interactive_unet.app.scheduler import JobSpec

EPS = 1e-9
PI = np.pi


def euler_xyz(R):
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-6:
        return (
            float(np.arctan2(R[2, 1], R[2, 2])),
            float(np.arctan2(-R[2, 0], sy)),
            float(np.arctan2(R[1, 0], R[0, 0])),
        )
    return (
        float(np.arctan2(-R[1, 2], R[1, 1])),
        float(np.arctan2(-R[2, 0], sy)),
        0.0,
    )


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + EPS)


def wrap_pi(a):
    return (a + PI) % (2 * PI) - PI


def Rx(theta: float) -> np.ndarray:
    """
    Rotation matrix around +X
    """
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]], float)


def add_wire_box(scene, size_xyz, color="#aaaaaa"):
    """
    Create a wireframe box that represents th volume
    """
    sx, sy, sz = map(float, size_xyz)
    hx, hy, hz = sx / 2, sy / 2, sz / 2

    pts = np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz],
            [-hx,  hy, -hz], [hx,  hy, -hz],
            [-hx, -hy,  hz], [hx, -hy,  hz],
            [-hx,  hy,  hz], [hx,  hy,  hz],
        ]
    )
    edges = [
        (0, 1), (0, 2), (2, 3), (1, 3),
        (4, 5), (4, 6), (6, 7), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    with scene.group() as g:
        for i, j in edges:
            ui.scene.line(tuple(pts[i]), tuple(pts[j])).material(color)

    return g


class NavigatorWidget:

    def __init__(self, state, on_change=lambda camera: None, axis_len=1.0, ring_scale=1.3):

        self.camera = state.camera
        self.nav = state.nav

        self._sync_exec = ThreadPoolExecutor(max_workers=1)

        self.on_change = on_change

        self.axis_len = float(axis_len)
        self.ring_radius = self.axis_len * float(ring_scale)

        self.W2S = Rx(+PI / 2)
        self.S2W = self.W2S.T

        self.volume_shape = None
        self.volume_center = None
        self.world_to_scene_scale = None
        self._wire_box = None

        self._last_plane_dims = None
        self._drag_pos = {}

        with ui.scene(
            grid=False,
            on_drag_start=self._drag_start,
            on_drag_end=self._drag_end,
        ).classes('w-full') as self.scene:
            self.scene.style("touch-action: none;")

            self.scene.on("drag", self._drag, ["object_id", "x", "y", "z"])

            with self.scene.group() as self.gizmo:
                self._ring("yz", "#ffb3b3")
                self._ring("xz", "#b3ffb3")
                self._ring("xy", "#b3b3ff")
                self.plane = None

            self.move_u = self._arrow("#ff0000")
            self.move_v = self._arrow("#00ff00")
            self.move_w = self._arrow("#0000ff")

            self.rot_u = self.scene.sphere(0.1).material("#ff0000").draggable()
            self.rot_v = self.scene.sphere(0.1).material("#00ff00").draggable()
            self.rot_w = self.scene.sphere(0.1).material("#0000ff").draggable()

        ui.timer(0.1, self._enable_touch_orbit, once=True)

    def initialize(self, volume_shape, scheduler):

        self.scheduler = scheduler
        self.scheduler.register_sync(
            "sync_navigator",
            fn=self._sync,
            spec=JobSpec(
                max_hz=30,
                mode="latest",
                executor=self._sync_exec,
                sequential_executor=False,
            ),
        )

        self.volume_shape = np.asarray(volume_shape, float)
        self.volume_center = self.volume_shape / 2
        self.world_to_scene_scale = 6.0 / float(np.max(self.volume_shape))

        if self._wire_box is not None:
            self._wire_box.delete()
            self._wire_box = None

        self._wire_box = add_wire_box(
            self.scene,
            self.volume_shape * self.world_to_scene_scale,
        )

        self._last_plane_dims = None
        self._sync()

    def _enable_touch_orbit(self):
        ui.run_javascript(f"""
        (function () {{
          const el = getElement({self.scene.id});
          if (!el || !el.controls || !window.THREE) return;
          const c = el.controls;
          c.enableRotate = true;
          c.enablePan    = true;
          c.enableZoom   = true;
          c.touches.ONE = THREE.TOUCH.ROTATE;
          c.touches.TWO = THREE.TOUCH.DOLLY_PAN;
          c.screenSpacePanning = true;
        }})();
        """)

    def _to_scene(self, p):
        if self.world_to_scene_scale is None:
            return np.zeros(3)
        q = (np.asarray(p, float) - self.volume_center)
        q = self.W2S @ q
        return q * self.world_to_scene_scale

    def _to_world_delta(self, d):
        if self.world_to_scene_scale is None:
            return np.zeros(3)
        q = np.asarray(d, float) / self.world_to_scene_scale
        return self.S2W @ q

    def _ring(self, plane, color, steps=72):
        r = self.ring_radius
        axes = {
            "yz": (np.array([0, 1, 0]), np.array([0, 0, 1])),
            "xz": (np.array([1, 0, 0]), np.array([0, 0, 1])),
            "xy": (np.array([1, 0, 0]), np.array([0, 1, 0])),
        }
        a, b = axes[plane]
        ts = np.linspace(0, 2 * PI, steps + 1)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            ui.scene.line(
                tuple(r * (np.cos(t0) * a + np.sin(t0) * b)),
                tuple(r * (np.cos(t1) * a + np.sin(t1) * b)),
            ).material(color)

    def _arrow(self, color):
        with self.scene.group() as g:
            radius_scale = 1.5
            ui.scene.cylinder(
                radius_scale * 0.02 * self.axis_len, radius_scale * 0.02 * self.axis_len, 0.8 * self.axis_len
            ).move(y=0.4 * self.axis_len).material(color)
            ui.scene.cylinder(
                0, 0.10 * self.axis_len, 0.2 * self.axis_len
            ).move(y=0.9 * self.axis_len).material(color)
        return g.draggable()

    def _ensure_plane(self, plane_w: float, plane_h: float):
        dims = (float(plane_w), float(plane_h))
        if self._last_plane_dims == dims:
            return
        if self.plane:
            self.plane.delete()
        with self.gizmo:
            self.plane = (
                ui.scene.box(dims[0], dims[1], 0.02)
                .material("#4a90e2")
                .rotate(0, -PI / 2, 0)
            )
        self._last_plane_dims = dims

    def _sync(self, skip_id=None):
        if self.volume_shape is None:
            return

        origin_s = self._to_scene(self.camera.origin)

        u, v, w = self.camera.uvw
        u = self.W2S @ np.asarray(u, float)
        v = self.W2S @ np.asarray(v, float)
        w = self.W2S @ np.asarray(w, float)

        h, w_px = self.nav.slice_shape
        plane_w = (w_px * self.camera.zoom) * self.world_to_scene_scale
        plane_h = (h    * self.camera.zoom) * self.world_to_scene_scale

        self._ensure_plane(plane_w, plane_h)

        self.gizmo.move(*origin_s).rotate(*euler_xyz(np.stack([u, v, w], axis=1)))

        def place(obj, d):
            d = unit(d)
            zref = np.array([0, 0, 1]) if abs(d @ [0, 0, 1]) < 0.99 else np.array([1, 0, 0])
            x = unit(np.cross(d, zref))
            z = unit(np.cross(x, d))
            obj.move(*origin_s).rotate(*euler_xyz(np.stack([x, d, z], axis=1)))

        if skip_id != self.move_u.id:
            place(self.move_u, u)
        if skip_id != self.move_v.id:
            place(self.move_v, v)
        if skip_id != self.move_w.id:
            place(self.move_w, -w)

        r = self.ring_radius
        if skip_id != self.rot_u.id:
            self.rot_u.move(*(origin_s + v * r))
        if skip_id != self.rot_v.id:
            self.rot_v.move(*(origin_s - w * r))
        if skip_id != self.rot_w.id:
            self.rot_w.move(*(origin_s + u * r))

        self.on_change(self.camera)

    def _drag_start(self, e):
        self._drag_pos[e.object_id] = np.array([e.x, e.y, e.z], float)

    def _drag(self, e):
        oid = e.args.get("object_id")
        if oid not in self._drag_pos:
            return

        p1 = np.array([e.args["x"], e.args["y"], e.args["z"]], float)
        p0 = self._drag_pos[oid]
        self._drag_pos[oid] = p1

        d = self._to_world_delta(p1 - p0)

        u, v, w = self.camera.uvw 

        if oid == self.move_u.id:
            self.camera.scroll(float(d @ u))
        elif oid == self.move_v.id:
            self.camera.pan(0.0, float(d @ v))
        elif oid == self.move_w.id:
            self.camera.pan(float(d @ w), 0.0)
        elif oid == self.rot_u.id:
            self.camera.rotate_axis("u", wrap_pi(self._ring_angle("u", p1) - self._ring_angle("u", p0)))
        elif oid == self.rot_v.id:
            self.camera.rotate_axis("v", -wrap_pi(self._ring_angle("v", p1) - self._ring_angle("v", p0)))
        elif oid == self.rot_w.id:
            self.camera.rotate_axis("w", wrap_pi(self._ring_angle("w", p1) - self._ring_angle("w", p0)))

        self._sync(skip_id=oid)

        if self.volume_shape is not None:
            self.scheduler.request("nav_preview")

    def _ring_angle(self, axis, p):
        origin_s = self._to_scene(self.camera.origin)
        rel = np.asarray(p, float) - origin_s

        u, v, w = self.camera.uvw
        u = self.W2S @ np.asarray(u, float)
        v = self.W2S @ np.asarray(v, float)
        w = self.W2S @ np.asarray(w, float)

        a, b = {"u": (v, w), "v": (u, w), "w": (u, v)}[axis]
        return float(np.arctan2(rel @ b, rel @ a))

    def _drag_end(self, e):
        self._drag_pos.pop(e.object_id, None)
        self._sync()
        if self.volume_shape is not None:
            self.scheduler.request("nav_hires")
