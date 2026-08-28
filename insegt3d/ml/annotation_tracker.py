import os
import json
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from insegt3d.volume.camera import Camera

@dataclass
class Annotation:
    time_idx: int
    volume_path: str
    class_idx: int
    camera: Camera
    extent: tuple

class AnnotationTracker:
    """
    Records per-class annotation regions as (camera, extent) snapshots.
    """

    def __init__(self, project_path):

        self.project_path = project_path

        self.annotations_path = Path(self.project_path) / "annotations.json"
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)

        self._time = 0
        self._history = []
        self._redo = []
        self._active = []

        self._lock = threading.Lock()

        if self.annotations_path.is_file():
            self.load(self.annotations_path)

    def annotations(self):
        with self._lock:
            return list(self._active)

    def reset(self):
        with self._lock:
            self._time = 0
            self._history.clear()
            self._redo.clear()
            self._active = []
            self._autosave()

    def on_annotation_commit(self, volume_path, camera, mask, write_extent, axis=0):
        if volume_path is None:
            return None

        labels = np.unique(mask)
        labels = labels[labels > 0]
        if labels.size == 0:
            return None

        H, W = mask.shape
        volume_path = str(volume_path)

        with self._lock:
            self._redo.clear()
            self._time += 1
            time_idx = self._time

            for cls in labels.tolist():
                bbox = self._bbox(mask == cls)
                if bbox is None:
                    continue

                extent_view = self._bbox_to_extent_view(bbox, (H, W), write_extent)

                cam_saved, extent_saved = self._normalize_sample(camera, extent_view)

                ann = Annotation(
                    time_idx=time_idx,
                    volume_path=volume_path,
                    class_idx=int(cls),
                    camera=cam_saved,
                    extent=extent_saved,
                )
                self._history.append(ann)

            self._active = list(self._history)
            self._autosave()

    def undo(self):
        with self._lock:
            if not self._history:
                return None
            ann = self._history.pop()
            self._redo.append(ann)
            self._active = list(self._history)
            self._autosave()
            return ann

    def redo(self):
        with self._lock:
            if not self._redo:
                return None
            ann = self._redo.pop()
            self._history.append(ann)
            self._active = list(self._history)
            self._autosave()
            return ann

    def _autosave(self):
        self.save(self.annotations_path)

    def _bbox(self, mask_bool):
        ys, xs = np.where(mask_bool)
        if ys.size == 0:
            return None
        return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    def _bbox_to_extent_view(self, bbox, shape, write_extent):
        
        d0, d1, top, bottom, left, right = map(float, write_extent)
        H, W = map(int, shape)
        y0, y1, x0, x1 = bbox

        vy = (bottom - top) / max(1, (H - 1))
        vx = (right - left) / max(1, (W - 1))

        top2    = top  + (y0 + 0.5) * vy
        bottom2 = top  + (y1 - 0.5) * vy
        left2   = left + (x0 + 0.5) * vx
        right2  = left + (x1 - 0.5) * vx

        if bottom2 < top2:
            top2, bottom2 = bottom2, top2
        if right2 < left2:
            left2, right2 = right2, left2

        return (d0, d1, top2, bottom2, left2, right2)

    def _normalize_sample(self, camera, extent_view):
        
        cam = camera.copy()
        d0, d1, top, bottom, left, right = map(float, extent_view)

        z = float(cam.zoom)

        cy = 0.5 * (top + bottom)
        cx = 0.5 * (left + right)
        cam.origin = cam.origin + (cy * cam.v + cx * cam.w) * z

        if d1 > d0:
            cd = 0.5 * (d0 + d1)
            cam.origin = cam.origin + (cd * cam.u) * z

        d0 *= z
        d1 *= z
        top *= z
        bottom *= z
        left *= z
        right *= z

        # Set zoom to 1
        cam.zoom = 1.0

        # Center extent around 0
        extent_z1 = self._center_extent((d0, d1, top, bottom, left, right))
        return cam, extent_z1

    def _center_extent(self, extent):
        d0, d1, top, bottom, left, right = map(float, extent)

        cy = 0.5 * (top + bottom)
        cx = 0.5 * (left + right)

        top -= cy
        bottom -= cy
        left -= cx
        right -= cx

        if d1 > d0:
            cd = 0.5 * (d0 + d1)
            d0 -= cd
            d1 -= cd

        return (d0, d1, top, bottom, left, right)

    def _annotation_to_dict(self, a):
        return {
            "time_idx": int(a.time_idx),
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

    def save(self, path):
        path = str(path)
        tmp = path + ".tmp"

        payload = [self._annotation_to_dict(a) for a in self._history]

        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    def load(self, path):
        with open(path) as f:
            payload = json.load(f)

        self._history = [self._dict_to_annotation(x) for x in payload
                         if x.get("volume_path") not in (None, "", "None")]
        self._redo.clear()
        self._active = list(self._history)
        self._time = max((a.time_idx for a in self._history), default=0)
