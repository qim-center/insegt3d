import os
import cv2
import json
import base64
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, replace

from insegt3d.volume.camera import Camera
from insegt3d.volume.interp import ERASE

def encode_points(points):
    return base64.b64encode(np.asarray(points, np.float32).tobytes()).decode('ascii')

def decode_points(text):
    return np.frombuffer(base64.b64decode(text), np.float32).reshape(-1, 2)

@dataclass
class Annotation:
    volume_path: str
    class_idx: int  # ERASE for erase strokes, which are kept only for replaying
    camera: Camera
    extent: tuple
    # Finest stored mask level, and for replay the brush path as (width, points) segments, or else PNG pixels
    level: int = 0
    stroke: str = ''
    path: list = None

    def replay(self, spacing):
        """Camera, label image and extent that VolumeSlicer.set_data needs to write the stroke again, with a brush path drawn at `spacing`."""
        image, size, top, left = self._draw_path(spacing) if self.path else self._pixels()

        # A square image, as set_data spaces rows and columns alike
        n = len(image)
        camera = self.camera.copy()
        camera.zoom = size * (n - 1) / (n - 0.5)
        top, left = top / camera.zoom, left / camera.zoom
        d0, d1 = self.extent[:2]
        return camera, image, (d0, d1, top, top + n - 0.5, left, left + n - 0.5)

    def _pixels(self):
        """The stored pixels with a blank border, their spacing and the first pixel's center."""
        pixels = cv2.imdecode(np.frombuffer(base64.b64decode(self.stroke), np.uint8), cv2.IMREAD_GRAYSCALE)
        h, w = pixels.shape
        _, _, top, bottom, left, right = self.extent
        # Reconstruct the spacing and first pixel center
        size = max((bottom - top) / max(h - 1, 1), (right - left) / max(w - 1, 1)) or 1.0
        top, left = top - size / 2, left - size / 2

        # The blank border, as set_data writes only up to the edge pixels' centers
        n = max(h, w) + 2
        square = np.zeros((n, n), np.uint8)
        square[1:h + 1, 1:w + 1] = pixels * np.uint8(self.class_idx)
        return square, size, top - size, left - size

    def _draw_path(self, spacing):
        """The brush path drawn at `spacing` with a blank border, and the first pixel's center."""
        segments = [(width, decode_points(points)) for width, points in self.path]
        vertices = np.concatenate([points for _, points in segments])
        margin = max(width for width, _ in segments) / 2 + 2 * spacing
        left, top = vertices.min(axis=0) - margin
        n = int(np.ceil((vertices.max(axis=0) + margin - (left, top)).max() / spacing)) + 1

        image = np.zeros((n, n), np.uint8)
        for width, points in segments:
            # In sixteenths of a pixel, so the path keeps its precision
            pixels = np.rint((points - (left, top)) / spacing * 16).astype(np.int32)
            cv2.polylines(image, [pixels], False, int(self.class_idx), max(1, int(np.rint(width / spacing))), shift=4)
        return image, spacing, top, left

class AnnotationTracker:
    """Records every stroke as a per-class (camera, extent) snapshot with its brush path or pixels in annotations.json."""

    def __init__(self, project_path):

        self.project_path = project_path

        self.annotations_path = Path(self.project_path) / "annotations.json"
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)

        self._history = []

        self._lock = threading.Lock()

        if self.annotations_path.is_file():
            self.load()

    def annotations(self):
        """Annotated regions, leaving out erase strokes."""
        with self._lock:
            return [a for a in self._history if a.class_idx != ERASE]

    def strokes(self):
        """Every stroke in drawing order, erase strokes included."""
        with self._lock:
            return list(self._history)

    def set_level(self, strokes, level):
        with self._lock:
            for a in strokes:
                a.level = level
            self.save()

    def reset(self):
        with self._lock:
            self._history.clear()
            self.save()

    def on_annotation_commit(self, volume_path, camera, mask, write_extent, level, path=None):
        labels = np.unique(mask)
        labels = labels[labels > 0]
        if labels.size == 0:
            return []

        volume_path = str(volume_path)
        # Classes overlapping within one stroke would replay in the wrong order, so only single class strokes keep their path
        path = path if labels.size == 1 else None
        added = []

        with self._lock:
            for cls in labels.tolist():
                cam_saved, extent_saved, pixels, stroke_path = self._sample(camera, mask, cls, write_extent, path)

                ann = Annotation(
                    volume_path=volume_path,
                    class_idx=int(cls),
                    camera=cam_saved,
                    extent=extent_saved,
                    level=level,
                    stroke='' if stroke_path else base64.b64encode(cv2.imencode('.png', pixels.astype(np.uint8))[1]).decode('ascii'),
                    path=stroke_path or None,
                )
                added.append(ann)

            self._history.extend(added)
            self.save()

        return added

    def remove(self, annotations):
        with self._lock:
            self._history = [a for a in self._history if not any(a is b for b in annotations)]
            self.save()

    def restore(self, annotations):
        with self._lock:
            self._history.extend(annotations)
            self.save()

    def relabel(self, lut):
        # Erase strokes are not a class and stay as they are
        lut = np.array(lut)
        lut[ERASE] = ERASE
        with self._lock:
            history = []
            for a in self._history:
                if lut[a.class_idx]:
                    history.append(replace(a, class_idx=int(lut[a.class_idx])))
                elif a.level > 0:
                    # Levels finer than this stroke's still lack it, so it replays as an erase, as relabeling erased its class here
                    history.append(replace(a, class_idx=ERASE))
            self._history = history
            self.save()

    def _sample(self, camera, mask, cls, write_extent, path):
        """Return a centered camera, extent, pixels, and brush path in its frame for `mask == cls`."""
        ys, xs = np.where(mask == cls)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        H, W = mask.shape

        d0, d1, top, bottom, left, right = map(float, write_extent)
        vy = (bottom - top) / max(1, (H - 1))
        vx = (right - left) / max(1, (W - 1))
        corner = np.array([left, top])

        top, bottom = top + (y0 + 0.5) * vy, top + (y1 - 0.5) * vy
        left, right = left + (x0 + 0.5) * vx, left + (x1 - 0.5) * vx
        cd, cy, cx = 0.5 * (d0 + d1), 0.5 * (top + bottom), 0.5 * (left + right)

        cam = camera.copy()
        z = float(cam.zoom)
        cam.origin = cam.origin + (cd * cam.u + cy * cam.v + cx * cam.w) * z
        cam.zoom = 1.0

        extent = (d0 - cd, d1 - cd, top - cy, bottom - cy, left - cx, right - cx)
        if path is not None:
            path = [[width * vy * z, encode_points((points * (vx, vy) + corner - (cx, cy)) * z)] for width, label, points in path if label == cls]
        return cam, tuple(v * z for v in extent), mask[y0:y1, x0:x1] == cls, path

    def _annotation_to_dict(self, a):
        return {
            "volume_path": str(a.volume_path),
            "class_idx": int(a.class_idx),
            "camera": a.camera.to_dict(),
            "extent": list(map(float, a.extent)),
            "level": int(a.level),
            "stroke": a.stroke,
            "path": a.path,
        }

    def _dict_to_annotation(self, d):
        d = dict(d)
        d["camera"] = Camera.from_dict(d["camera"])
        d["extent"] = tuple(d["extent"])
        return Annotation(**d)

    def save(self):
        tmp = str(self.annotations_path) + ".tmp"

        payload = [self._annotation_to_dict(a) for a in self._history]

        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, self.annotations_path)

    def load(self):
        with open(self.annotations_path) as f:
            payload = json.load(f)

        self._history = [self._dict_to_annotation(x) for x in payload]
