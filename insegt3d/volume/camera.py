from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

# u, v, w of each axis-aligned view, kept right-handed (w = u x v)
AXIS_VIEWS = {
    "z": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "y": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "x": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
}

@dataclass
class Camera:
    origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))

    # u is the slice normal, with v and w its vertical and horizontal directions
    u: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    w: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))

    zoom: float = 1.0  # World units per slice pixel, so larger values zoom out
    version: int = 0  # Bumped on every change so stale renders can be discarded

    _VECTOR_FIELDS = ("origin", "u", "v", "w")

    def copy(self):
        cam = self.__class__.__new__(self.__class__)
        for name in self._VECTOR_FIELDS:
            setattr(cam, name, getattr(self, name).copy())
        cam.zoom = self.zoom
        cam.version = self.version
        return cam

    def to_dict(self):
        d = {name: getattr(self, name).tolist() for name in self._VECTOR_FIELDS}
        d["zoom"] = float(self.zoom)
        return d

    @classmethod
    def from_dict(cls, d):
        cam = cls()
        for name in cls._VECTOR_FIELDS:
            setattr(cam, name, np.array(d[name], dtype=np.float32))
        cam.zoom = float(d["zoom"])
        return cam

    def reset(self, world_shape):
        self.set_view(world_shape / 2, 1.0, AXIS_VIEWS["z"])

    def set_view(self, origin, zoom, uvw):
        self.origin = np.asarray(origin, dtype=np.float32)
        self.zoom = float(zoom)
        self.u, self.v, self.w = (np.array(axis, dtype=np.float32) for axis in uvw)
        self.version += 1

    def look_along(self, normal):
        """Turns u to `normal` with the smallest rotation, keeping the in-plane orientation."""
        rot, _ = Rotation.align_vectors([normal], [self.u.astype(np.float64)])
        self._apply_rotation(rot)

    @property
    def uvw(self):
        return self.u, self.v, self.w

    def _normalize(self, v):
        return v / np.linalg.norm(v)

    def _orthonormalize(self):
        """Gram-Schmidt: keeps u, v, w orthonormal and numerically stable."""
        u = self._normalize(self.u)
        v = self.v - (u @ self.v) * u
        v = self._normalize(v)
        w = np.cross(u, v)

        w = self._normalize(w)

        self.u, self.v, self.w = u, v, w

    def _apply_rotation(self, R):
        self.u, self.v, self.w = R.apply(np.stack(self.uvw).astype(np.float64)).astype(np.float32)
        self._orthonormalize()
        self.version += 1

    def randomize(self):
        self.look_along(np.random.normal(size=3))

    def _translate(self, t=(0, 0, 0)):
        t = np.array(t, dtype=np.float32) * float(self.zoom)
        delta_world = t[0] * self.u + t[1] * self.v + t[2] * self.w
        self.origin = self.origin + delta_world
        self.version += 1

    def pan(self, dx, dy):
        self._translate((0, dy, dx))

    def scroll(self, dz):
        self._translate((dz, 0, 0))

    def step(self, slices):
        self.origin = self.origin + float(slices) * self.u
        self.version += 1

    def rotate(self, dx, dy, sensitivity=0.002, tol=1e-8):
        dx = float(dx)
        dy = float(dy)

        drag = -dy * self.v + dx * self.w
        drag_norm = float(np.linalg.norm(drag))
        if drag_norm < tol:
            return

        axis = np.cross(self.u, drag)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < tol:
            return
        axis /= axis_norm

        theta = drag_norm * float(sensitivity)
        self._apply_rotation(Rotation.from_rotvec(axis.astype(np.float64) * theta))

    def rotate_axis(self, axis, angle):
        """Rotates u, v, w around one of its own axes ('u', 'v', or 'w'), `angle` radians, right-hand rule."""
        rot_axis = self._normalize({"u": self.u, "v": self.v, "w": self.w}[axis])
        self._apply_rotation(Rotation.from_rotvec(rot_axis * float(angle)))

    def zoom_by(self, zoom_factor, min_zoom=1 / 64, max_zoom=1024):
        self.zoom = float(np.clip(self.zoom * zoom_factor, min_zoom, max_zoom))
        self.version += 1
