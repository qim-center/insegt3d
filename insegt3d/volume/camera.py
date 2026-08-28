from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class Camera:
    """
    A dataclass that stores parameters of a camera.
    """
    origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))

    u: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    w: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))

    zoom: float = 1.0

    def __post_init__(self):
        self._update_orientation_vectors(self.u)

    def copy(self):
        cls = self.__class__
        cam = cls.__new__(cls)
        cam.origin = self.origin.copy()
        cam.u = self.u.copy()
        cam.v = self.v.copy()
        cam.w = self.w.copy()
        cam.zoom = self.zoom
        return cam

    def to_dict(self):
        return {
            "origin": self.origin.tolist(),
            "u": self.u.tolist(),
            "v": self.v.tolist(),
            "w": self.w.tolist(),
            "zoom": float(self.zoom),
        }

    @classmethod
    def from_dict(cls, d):
        cam = cls()
        cam.origin = np.array(d["origin"], dtype=np.float32)
        cam.u = np.array(d["u"], dtype=np.float32)
        cam.v = np.array(d["v"], dtype=np.float32)
        cam.w = np.array(d["w"], dtype=np.float32)
        cam.zoom = float(d["zoom"])
        return cam

    def reset(self, volume_shape):

        self.origin = volume_shape / 2

        self.u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.w = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self.zoom = 1.0

    @property
    def uvw(self):
        return self.u.astype(np.float32), self.v.astype(np.float32), self.w.astype(np.float32)

    def world_coords(self, d, y, x):
        return self.origin + float(d) * self.u + float(y) * self.v + float(x) * self.w

    def slice_axes(self, axis=0):
        basis = self.uvw
        n = basis[axis]
        a0, a1 = [basis[i] for i in range(3) if i != axis]
        return n, a0, a1

    def _normalize(self, v):
        """
        Converts vector to unit length.
        """
        return v / np.linalg.norm(v)

    def _orthonormalize(self):
        """
        Use Gram-Schmidt algorithm to keep u, v, w orthonormal and more numerically stable.
        """
        u = self._normalize(self.u)
        v = self.v - (u @ self.v) * u
        v = self._normalize(v)
        w = np.cross(u, v)

        w = self._normalize(w)

        self.u, self.v, self.w = u, v, w

    def _update_orientation_vectors(self, rotation_vector):
        """
        Updates the orientation vectors from the given rotation vector (target u direction).
        Uses scipy.spatial.transform for robust rotation handling.
        """
        rotation_vector = self._normalize(rotation_vector)

        # Compute rotation matrix, M
        rot, _ = Rotation.align_vectors(
            [rotation_vector.astype(np.float64)],
            [[1.0, 0.0, 0.0]]
        )
        M = rot.as_matrix().astype(np.float32)

        # Rotate vectors
        self.u = M @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.v = M @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.w = M @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self._orthonormalize()

    def _generate_uniformly_random_unit_vector(self, ndim=3):
        """
        Generates a uniformly random unit vector.
        """

        u = np.random.normal(size=ndim)

        # Regenerate to avoid rounding issues
        while np.linalg.norm(u) < 0.0001:
            u = np.random.normal(size=ndim)

        # Make unit vector
        u = self._normalize(u)

        return u

    def randomize(self):
        """
        Randomizes camera orientation.
        """
        rotation_vector = self._generate_uniformly_random_unit_vector().astype(np.float32)
        self._update_orientation_vectors(rotation_vector)

    def _translate(self, t=(0, 0, 0)):
        """
        Translates origin by the given amount.
        """

        # Adjust translation speed based on zoom
        t = np.array(t, dtype=np.float32) * float(self.zoom)

        # Convert from camera-local coords to world coords using the current basis
        delta_world = t[0] * self.u + t[1] * self.v + t[2] * self.w

        self.origin = self.origin + delta_world

    def pan(self, dx, dy):
        """
        Pans along the v-w plane.
        """
        self._translate((0, dy, dx))

    def scroll(self, dz):
        """
        Scrolls through slices along normal vector, u
        """
        self._translate((dz, 0, 0))

    def rotate(self, dx, dy, sensitivity=0.002, tol=1e-8):
        """
        Rotates about the axis perpendicular to the drag direction.
        """

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

        R = Rotation.from_rotvec(axis.astype(np.float64) * theta)

        self.u = R.apply(self.u.astype(np.float64)).astype(np.float32)
        self.v = R.apply(self.v.astype(np.float64)).astype(np.float32)
        self.w = R.apply(self.w.astype(np.float64)).astype(np.float32)

        self._orthonormalize()

    def rotate_axis(self, axis, angle):
        """
        Rotate camera basis (u, v, w) around one of its own axes.

        Args:
            axis: 'u', 'v', or 'w' (case-insensitive)
            angle: rotation angle in radians (right-hand rule about the chosen axis)
        """

        rot_axis = {"u": self.u, "v": self.v, "w": self.w}[axis]
        rot_axis = self._normalize(rot_axis)

        R = Rotation.from_rotvec(rot_axis * float(angle))

        self.u = R.apply(self.u.astype(np.float64)).astype(np.float32)
        self.v = R.apply(self.v.astype(np.float64)).astype(np.float32)
        self.w = R.apply(self.w.astype(np.float64)).astype(np.float32)

        self._orthonormalize()

    def zoom_by(self, zoom_factor):
        """
        Zoom in / out by the given factor
        """

        self.zoom *= zoom_factor
