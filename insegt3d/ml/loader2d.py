import numpy as np
from pathlib import Path
import tensorstore as ts

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from insegt3d.volume.slicer import VolumeSlicer
from insegt3d.volume.io import read_multiscale_masks
from insegt3d.volume.intensity import robust_normalize

class LiveTrainingDataset(Dataset):
    """
    Samples (image, mask, weight) directly from volumes using stored annotations.
    """

    def __init__(self, tracker, input_size=512, num_classes=2, axis=0, cache_size_mb=8000):
        self.tracker = tracker
        self.rng = np.random.default_rng()

        self._slicers = {}
        self._ts_context = ts.Context({
            'cache_pool': {'total_bytes_limit': int(cache_size_mb * 1024**2)}
        })

        self.project_path = self.tracker.project_path
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.axis = int(axis)

    def __len__(self):
        return len(self.tracker.annotations())

    def _augment_camera(self, camera):
        """
        Randomizes the sampled view around a stored annotation
        """
        camera.rotate_axis('u', self.rng.uniform(0, 2 * np.pi))

        max_shift = 0.5 * float(self.input_size)
        dy = self.rng.uniform(-max_shift, max_shift)
        dx = self.rng.uniform(-max_shift, max_shift)
        camera.origin = camera.origin + dy * camera.v + dx * camera.w

        camera.zoom = self.rng.uniform(0.9, 1.1)
        return camera

    def __getitem__(self, idx):
        ann = self.tracker.annotations()[idx]

        volume_path = str(ann.volume_path)
        mask_path = Path(self.project_path) / 'masks' / Path(ann.volume_path).name

        slicer = self._slicers.get(volume_path)
        if slicer is None:
            slicer = VolumeSlicer(ts_context=self._ts_context)
            slicer.initialize(ann.volume_path, mask_path, ann.camera)
            self._slicers[volume_path] = slicer
        else:
            level_shapes = [image.shape for image in slicer.images]
            slicer.masks = read_multiscale_masks(mask_path, level_shapes, ts_context=slicer.ts_context)

        camera = self._augment_camera(ann.camera.copy())

        half = self.input_size // 2
        extent = (0, 0, -half, half, -half, half)

        img = slicer.get_data(
            camera,
            extent=extent,
            out_shape=(self.input_size, self.input_size),
            mask=False,
            order=1,
            axis=self.axis,
        )

        msk = slicer.get_data(
            camera,
            extent=extent,
            out_shape=(self.input_size, self.input_size),
            mask=True,
            order=0,
            axis=self.axis,
        )

        x = torch.from_numpy(robust_normalize(img))[None]
        y = torch.from_numpy(msk).long()

        w = (y != 0).float()[None]

        y = torch.clamp(y - 1, min=0)
        y = F.one_hot(y, num_classes=self.num_classes)
        y = y.permute(2, 0, 1)

        return x, y, w

class RecencySampler(Sampler):
    """
    Ensures each batch contains at least one annotation per class.
    Favors recent annotations via temperature.
    """

    def __init__(self, tracker, batch_size=4, steps=20, recency_temp=20.0):
        self.tracker = tracker
        self.batch_size = batch_size
        self.steps = steps
        self.recency_temp = max(recency_temp, 1e-6)  # avoid divide-by-zero below
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            anns = self.tracker.annotations()
            if not anns:
                continue

            times = np.array([a.time_idx for a in anns], dtype=np.float32)
            tmax = times.max()
            weights = np.exp(-(tmax - times) / self.recency_temp)
            weights /= weights.sum()

            by_class = {}
            for i, a in enumerate(anns):
                by_class.setdefault(a.class_idx, []).append(i)

            chosen = []

            # One sample per class first, then fill the rest of the batch
            for idxs in by_class.values():
                idxs = np.array(idxs)
                p = weights[idxs]
                p /= p.sum()
                chosen.append(int(self.rng.choice(idxs, p=p)))

            while len(chosen) < self.batch_size:
                chosen.append(int(self.rng.choice(len(anns), p=weights)))

            yield chosen[:self.batch_size]

def collate_x_y_w(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    w = torch.stack([b[2] for b in batch])
    return x, y, w

def build_dataloader(
    tracker,
    input_size=512,
    num_classes=4,
    batch_size=4,
    steps_per_epoch=20,
    recency_temp=20.0,
    cache_size_mb=8000,
):
    dataset = LiveTrainingDataset(
        tracker=tracker,
        input_size=input_size,
        num_classes=num_classes,
        cache_size_mb=cache_size_mb,
    )

    sampler = RecencySampler(
        tracker=tracker,
        batch_size=batch_size,
        steps=steps_per_epoch,
        recency_temp=recency_temp,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_x_y_w,
        num_workers=0,
        pin_memory=True,
    )
