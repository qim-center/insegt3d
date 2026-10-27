import threading
from pathlib import Path
from collections import deque

from insegt3d.volume.camera import Camera
from insegt3d.volume.io import volume_folder_name
from insegt3d.volume.slicer import VolumeSlicer


class EditHistory:

    def __init__(self, slicer, tracker, train, max_undo=50):
        self.slicer = slicer
        self.tracker = tracker
        self.train = train

        self._undo = deque(maxlen=max_undo)
        self._redo = []
        self._lock = threading.Lock()

    def commit(self, camera, mask, extent, path=None):
        with self._lock:
            if self.slicer.images is None:
                return

            # Use the coarsest available level if the working level is missing
            num_levels = len(self.slicer.images)
            level = min(self.train.level, num_levels - 1)
            for _ in self._replay(self.slicer, level):
                pass

            patches = self.slicer.set_data(camera, mask, extent, range(level, num_levels))
            if patches:
                annotations = self.tracker.on_annotation_commit(self.slicer.zarr_path, camera, mask, extent, level, path)
                self._undo.append((patches, annotations))
                self._redo.clear()

    def change_level(self, level, masks_root, progress):
        """Replay strokes into finer levels, calling progress(replayed, total) as it goes, then set the working level."""
        failed = []
        with self._lock:
            pending = [stroke for stroke in self.tracker.strokes() if stroke.level > level]
            replayed = 0
            if pending:
                progress(replayed, len(pending))

            for volume_path in {stroke.volume_path for stroke in pending}:
                try:
                    slicer = self.slicer
                    if volume_path != str(self.slicer.zarr_path):
                        slicer = VolumeSlicer(ts_context=self.slicer.ts_context)
                        slicer.initialize(volume_path, Path(masks_root) / volume_folder_name(volume_path), Camera())
                    for _ in self._replay(slicer, level):
                        replayed += 1
                        progress(replayed, len(pending))
                except Exception:
                    failed.append(volume_path)

            self.train.level = level
            self._undo.clear()
            self._redo.clear()

        return failed

    def _replay(self, slicer, level):
        """Replay strokes into levels finer than their stored level, yielding after each."""
        strokes = [stroke for stroke in self.tracker.strokes() if stroke.volume_path == str(slicer.zarr_path) and stroke.level > level]
        spacing = slicer.voxel_sizes[level].min()
        for stroke in strokes:
            slicer.set_data(*stroke.replay(spacing), range(level, stroke.level))
            yield
        if strokes:
            self.tracker.set_level(strokes, level)

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
