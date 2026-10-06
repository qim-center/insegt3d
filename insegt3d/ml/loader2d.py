import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from insegt3d.volume.slicer import VolumeSlicer
from insegt3d.volume.io import volume_folder_name
from insegt3d.volume.intensity import robust_normalize

class LiveTrainingDataset:
    """Samples (image, mask, weight) directly from volumes using stored annotations."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.rng = np.random.default_rng()

        self._slicers = {}
        self._ts_context = None

    def batches(self, train, ts_context):
        """Yields batches holding one annotation per class (for a random subset if there are more classes than samples)."""
        if ts_context is not self._ts_context:
            self._slicers.clear()
            self._ts_context = ts_context

        for _ in range(train.steps_per_epoch):
            anns = self.tracker.annotations()

            by_class = {}
            for a in anns:
                by_class.setdefault(a.class_idx, []).append(a)

            if len(by_class) < train.num_classes:
                return

            classes = list(by_class.values())
            self.rng.shuffle(classes)

            chosen = [group[self.rng.integers(len(group))] for group in classes]

            while len(chosen) < train.batch_size:
                chosen.append(anns[self.rng.integers(len(anns))])

            batch = [torch.stack(tensors) for tensors in zip(*(self.sample(ann, train) for ann in chosen[:train.batch_size]))]
            yield [tensor.pin_memory() for tensor in batch] if torch.cuda.is_available() else batch

    def _augment_camera(self, camera, spacing, input_size):
        camera.rotate_axis('u', self.rng.uniform(0, 2 * np.pi))

        max_shift = 0.5 * float(input_size) * spacing
        dy = self.rng.uniform(-max_shift, max_shift)
        dx = self.rng.uniform(-max_shift, max_shift)
        camera.origin = camera.origin + dy * camera.v + dx * camera.w

        camera.zoom = spacing * self.rng.uniform(0.9, 1.1)
        return camera

    def sample(self, ann, train):
        volume_path = str(ann.volume_path)
        mask_path = Path(self.tracker.project_path) / 'masks' / volume_folder_name(ann.volume_path)

        slicer = self._slicers.get(volume_path)
        if slicer is None:
            slicer = VolumeSlicer(ts_context=self._ts_context)
            slicer.initialize(ann.volume_path, mask_path, ann.camera)
            self._slicers[volume_path] = slicer
        elif slicer.masks is None and mask_path.exists():
            slicer.masks = slicer.read_masks()

        size, slab_size, level = train.input_size, train.slab_size, train.level

        # Volume lacks this level: an all-zero weight makes the sample contribute nothing
        if len(slicer.images) <= level:
            return torch.zeros(slab_size, size, size), torch.zeros(train.num_classes, size, size, dtype=torch.long), torch.zeros(1, size, size)

        camera = self._augment_camera(ann.camera.copy(), slicer.voxel_sizes[level].min(), size)

        half = size // 2
        extent = (0, 0, -half, half, -half, half)

        d0 = -(slab_size // 2)

        img = slicer.get_data(
            camera,
            extent=(d0, d0 + slab_size - 1, -half, half, -half, half),
            out_shape=(slab_size, size, size),
            mask=False,
            order=1,
            level=level,
        )

        msk = slicer.get_data(
            camera,
            extent=extent,
            out_shape=(size, size),
            mask=True,
            order=0,
            level=level,
        )

        img = robust_normalize(img.reshape(slab_size, size, size))
        img = np.clip(img * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.05, 0.05), 0.0, 1.0)

        x = torch.from_numpy(img)
        y = torch.from_numpy(msk).long()

        # Label 0 (unannotated) gets zero weight, and labels 1..N become classes 0..N-1
        w = (y != 0).float()[None]

        y = torch.clamp(y - 1, min=0)
        y = F.one_hot(y, num_classes=train.num_classes)
        y = y.permute(2, 0, 1)

        return x, y, w

def prefetch(iterable):
    with ThreadPoolExecutor(max_workers=1) as pool:
        iterator = iter(iterable)
        pending = pool.submit(next, iterator, None)
        while (item := pending.result()) is not None:
            pending = pool.submit(next, iterator, None)
            yield item
