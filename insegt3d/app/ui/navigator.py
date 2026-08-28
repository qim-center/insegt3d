import json
import numpy as np
from nicegui import ui
from concurrent.futures import ThreadPoolExecutor
from insegt3d.app.scheduler import JobSpec

EPS = 1e-9
PI = np.pi


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + EPS)


def wrap_pi(a):
    return (a + PI) % (2 * PI) - PI


def basis_R(*columns):
    """
    Rotation matrix mapping the local x, y, z axes onto the given columns.
    """
    return np.stack([np.asarray(c, float) for c in columns], axis=1).tolist()


def frame_from_direction(d):
    """
    An orthonormal frame whose local +y axis points along d (arrows are modelled
    along +y).
    """
    d = unit(d)
    zref = np.array([0, 0, 1]) if abs(d @ [0, 0, 1]) < 0.99 else np.array([1, 0, 0])
    x = unit(np.cross(d, zref))
    z = unit(np.cross(x, d))
    return basis_R(x, d, z)


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
    Create a wireframe box that represents the volume
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

    with scene.group() as group:
        for i, j in edges:
            ui.scene.line(tuple(pts[i]), tuple(pts[j])).material(color)

    return group


class NavigatorWidget:

    def __init__(self, state, axis_len=1.0, ring_scale=1.3):

        self.camera = state.camera
        self.nav = state.nav

        self._sync_exec = ThreadPoolExecutor(max_workers=1)

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

        with ui.scene(grid=False).classes('w-full') as self.scene:
            self.scene.style("touch-action: none;")

            with self.scene.group() as self.gizmo:
                self._ring("yz", "#ffb3b3")
                self._ring("xz", "#b3ffb3")
                self._ring("xy", "#b3b3ff")

                self.plane = (
                    ui.scene.box(1.0, 1.0, 0.02)
                    .material("#4a90e2")
                    .rotate(0, -PI / 2, 0)
                    .visible(False)
                )

                self.move_u = self._arrow("#ff0000", (1, 0, 0))
                self.move_v = self._arrow("#00ff00", (0, 1, 0))
                self.move_w = self._arrow("#0000ff", (0, 0, -1))

                r = self.ring_radius
                self.rot_u = self.scene.sphere(0.1).material("#ff0000").move(0, r, 0).draggable()
                self.rot_v = self.scene.sphere(0.1).material("#00ff00").move(0, 0, -r).draggable()
                self.rot_w = self.scene.sphere(0.1).material("#0000ff").move(r, 0, 0).draggable()

        self._drag_event = f"gizmo_drag_{self.scene.id}"
        ui.on(self._drag_event, self._on_gizmo_drag)

        ui.timer(0.1, self._enable_touch_orbit, once=True)
        ui.timer(0.1, self._install_gizmo_drag, once=True)

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

    def close(self):
        self._sync_exec.shutdown(wait=False, cancel_futures=True)

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

    def _install_gizmo_drag(self):
        
        handles = {
            self.move_u.id: "u",
            self.move_v.id: "v",
            self.move_w.id: "w",
            self.rot_u.id: "ru",
            self.rot_v.id: "rv",
            self.rot_w.id: "rw",
        }

        ui.run_javascript(f"""
        (function install() {{
          const el = getElement({self.scene.id});
          if (!el || !el.drag_controls) {{ setTimeout(install, 100); return; }}
          if (el.__insegt3d_gizmo_installed) return;
          el.__insegt3d_gizmo_installed = true;

          // Handles are nested in the gizmo group now; without this DragControls
          // would select the outermost group and we could not tell them apart.
          el.drag_controls.transformGroup = false;

          const handles = {json.dumps(handles)};
          const event_name = {json.dumps(self._drag_event)};
          let key = null, home = null, quat = null, frozen = null;

          const lookup = (object) => {{
            for (let o = object; o; o = o.parent) {{
              if (o.object_id && handles[o.object_id]) return handles[o.object_id];
            }}
            return null;
          }};

          // DragControls froze the parent matrix at grab time; using the same
          // matrix recovers the raw pointer position without folding in any
          // gizmo motion that happened since.
          const report = (phase, object) => {{
            const p = object.position.clone().applyMatrix4(frozen);
            emitEvent(event_name, {{phase: phase, handle: key, x: p.x, y: p.y, z: p.z}});
          }};

          const restore = (object) => {{
            object.position.copy(home);
            object.quaternion.copy(quat);
          }};

          el.drag_controls.addEventListener('dragstart', (event) => {{
            key = lookup(event.object);
            if (!key) return;
            home = event.object.position.clone();
            quat = event.object.quaternion.clone();
            frozen = event.object.parent.matrixWorld.clone();
            report('start', event.object);
          }});

          el.drag_controls.addEventListener('drag', (event) => {{
            if (!key) return;
            report('move', event.object);
            restore(event.object);
          }});

          el.drag_controls.addEventListener('dragend', (event) => {{
            if (!key) return;
            restore(event.object);
            emitEvent(event_name, {{phase: 'end', handle: key}});
            key = null;
          }});
        }})();
        """)

    def _to_scene(self, p):
        if self.world_to_scene_scale is None:
            return np.zeros(3)
        q = (np.asarray(p, float) - self.volume_center)
        q = self.W2S @ q
        return q * self.world_to_scene_scale

    def _scene_axes(self):
        """
        The camera's u, v, w basis expressed in scene coordinates.
        """
        return tuple(self.W2S @ np.asarray(axis, float) for axis in self.camera.uvw)

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

    def _arrow(self, color, direction):
        shaft_radius = 1.5 * 0.02 * self.axis_len

        with self.scene.group() as group:
            ui.scene.cylinder(
                shaft_radius, shaft_radius, 0.8 * self.axis_len
            ).move(y=0.4 * self.axis_len).material(color)
            ui.scene.cylinder(
                0, 0.10 * self.axis_len, 0.2 * self.axis_len
            ).move(y=0.9 * self.axis_len).material(color)

        return group.rotate_R(frame_from_direction(direction)).draggable()

    def _sync(self, *_):
        if self.volume_shape is None:
            return

        origin_s = self._to_scene(self.camera.origin)
        u, v, w = self._scene_axes()

        h, w_px = self.nav.slice_shape
        plane_w = (w_px * self.camera.zoom) * self.world_to_scene_scale
        plane_h = (h    * self.camera.zoom) * self.world_to_scene_scale

        dims = (float(plane_w), float(plane_h))
        if self._last_plane_dims != dims:
            self.plane.scale(dims[0], dims[1], 1.0).visible(True)
            self._last_plane_dims = dims

        self.gizmo.move(*origin_s).rotate_R(basis_R(u, v, w))

    def _on_gizmo_drag(self, e):
        key = e.args.get("handle")
        phase = e.args.get("phase")
        if key is None:
            return

        if phase == "end":
            self._drag_pos.pop(key, None)
            self._sync()
            if self.volume_shape is not None:
                self.scheduler.request("nav_hires")
            return

        p = np.array([e.args["x"], e.args["y"], e.args["z"]], float)

        if phase == "start":
            self._drag_pos[key] = p
            return

        p0 = self._drag_pos.get(key)
        if p0 is None:
            return
        self._drag_pos[key] = p

        self._apply_drag(key, p0, p)

    def _apply_drag(self, key, p0, p1):
        if self.volume_shape is None:
            return

        d = self._to_world_delta(p1 - p0) / float(self.camera.zoom)

        u, v, w = self.camera.uvw

        if key == "u":
            self.camera.scroll(float(d @ u))
        elif key == "v":
            self.camera.pan(0.0, float(d @ v))
        elif key == "w":
            self.camera.pan(float(d @ w), 0.0)
        else:
            axis = {"ru": "u", "rv": "v", "rw": "w"}[key]
            self.camera.rotate_axis(
                axis, wrap_pi(self._ring_angle(axis, p1) - self._ring_angle(axis, p0))
            )

        self._sync()
        self.scheduler.request("nav_preview")

    def _ring_angle(self, axis, p):
        rel = np.asarray(p, float) - self._to_scene(self.camera.origin)

        u, v, w = self._scene_axes()

        a, b = {"u": (v, w), "v": (w, u), "w": (u, v)}[axis]
        return float(np.arctan2(rel @ b, rel @ a))
