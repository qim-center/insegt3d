import os
import json
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, replace

from insegt3d.volume.camera import Camera
from insegt3d.volume.interp import ERASE

@dataclass
class Annotation:
    volume_path: str
    class_idx: int
    camera: Camera
    extent: tuple

class AnnotationTracker:
    """Records per-class annotation regions as (camera, extent) snapshots in annotations.json."""

    def __init__(self, project_path):

        self.project_path = project_path

        self.annotations_path = Path(self.project_path) / "annotations.json"
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)

        self._history = []

        self._lock = threading.Lock()

        if self.annotations_path.is_file():
            self.load()

    def annotations(self):
        with self._lock:
            return list(self._history)

    def reset(self):
        with self._lock:
            self._history.clear()
            self.save()

    def on_annotation_commit(self, volume_path, camera, mask, write_extent):
        labels = np.unique(mask)
        labels = labels[(labels > 0) & (labels != ERASE)]
        if labels.size == 0:
            return []

        volume_path = str(volume_path)
        added = []

        with self._lock:
            for cls in labels.tolist():
                cam_saved, extent_saved = self._sample(camera, mask, cls, write_extent)

                ann = Annotation(
                    volume_path=volume_path,
                    class_idx=int(cls),
                    camera=cam_saved,
                    extent=extent_saved,
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
        with self._lock:
            self._history = [replace(a, class_idx=int(lut[a.class_idx])) for a in self._history if lut[a.class_idx]]
            self.save()

    def _sample(self, camera, mask, cls, write_extent):
        """Bounding box of `mask == cls` as a camera centred on it at zoom 1 and a centred extent, independent of how the user was viewing."""
        ys, xs = np.where(mask == cls)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        H, W = mask.shape

        d0, d1, top, bottom, left, right = map(float, write_extent)
        vy = (bottom - top) / max(1, (H - 1))
        vx = (right - left) / max(1, (W - 1))

        top, bottom = top + (y0 + 0.5) * vy, top + (y1 - 0.5) * vy
        left, right = left + (x0 + 0.5) * vx, left + (x1 - 0.5) * vx
        cd, cy, cx = 0.5 * (d0 + d1), 0.5 * (top + bottom), 0.5 * (left + right)

        cam = camera.copy()
        z = float(cam.zoom)
        cam.origin = cam.origin + (cd * cam.u + cy * cam.v + cx * cam.w) * z
        cam.zoom = 1.0

        extent = (d0 - cd, d1 - cd, top - cy, bottom - cy, left - cx, right - cx)
        return cam, tuple(v * z for v in extent)

    def _annotation_to_dict(self, a):
        return {
            "volume_path": str(a.volume_path),
            "class_idx": int(a.class_idx),
            "camera": a.camera.to_dict(),
            "extent": list(map(float, a.extent)),
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
