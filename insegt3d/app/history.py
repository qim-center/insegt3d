import threading
from collections import deque


class EditHistory:

    def __init__(self, slicer, tracker, max_undo=50):
        self.slicer = slicer
        self.tracker = tracker

        self._undo = deque(maxlen=max_undo)
        self._redo = []
        self._lock = threading.Lock()

    def commit(self, camera, mask, extent):
        with self._lock:
            patches = self.slicer.set_data(camera, mask, extent=extent)
            if patches:
                annotations = self.tracker.on_annotation_commit(self.slicer.zarr_path, camera, mask, extent)
                self._undo.append((patches, annotations))
                self._redo.clear()

    def undo(self):
        self._move(self._undo, self._redo, undo=True)

    def redo(self):
        self._move(self._redo, self._undo, undo=False)

    def _move(self, src, dst, undo):
        with self._lock:
            if not src:
                return
            patches, annotations = edit = src.pop()
            self.slicer.apply_patches(patches, undo)
            if undo:
                self.tracker.remove(annotations)
            else:
                self.tracker.restore(annotations)
            dst.append(edit)

    def clear(self):
        with self._lock:
            self._undo.clear()
            self._redo.clear()

    def reset(self, masks_root):
        with self._lock:
            self.tracker.reset()
            self.slicer.reset_masks(masks_root)
            self._undo.clear()
            self._redo.clear()

    def relabel(self, masks_root, lut):
        with self._lock:
            self.slicer.relabel_masks(masks_root, lut)
            self.tracker.relabel(lut)
            self._undo.clear()
            self._redo.clear()
