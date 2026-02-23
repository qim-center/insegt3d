import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from interactive_unet.volume.slicer import VolumeSlicer

class LiveTrainingDataset(Dataset):
    """
    Samples (image, mask, weight) directly from volumes using stored annotations.
    """

    def __init__(self, tracker, input_size=512, num_classes=2, axis=0):
        self.tracker = tracker
        self.slicer = VolumeSlicer()

        self.project_path = self.tracker.project_path
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.axis = int(axis)

    def __len__(self):
        return len(self.tracker.annotations())

    def _apply_random_rotation(self, camera):
        angle = np.random.rand() * 2 * np.pi
        camera.rotate_axis('u', angle)
        return camera

    def _apply_random_shift(self, camera):
        max_shift = 0.5 * float(self.input_size)
        dy = np.random.uniform(-max_shift, max_shift)
        dx = np.random.uniform(-max_shift, max_shift)
        camera.origin = camera.origin + dy * camera.v + dx * camera.w
        return camera

    def _apply_random_zoom(self, camera):
        zoom = np.random.uniform(0.9, 1.1)
        camera.zoom = zoom
        return camera

    def __getitem__(self, idx):
        ann = self.tracker.annotations()[idx]

        mask_path = Path(self.project_path) / 'masks' / Path(ann.volume_path).name
        self.slicer.initialize(ann.volume_path, mask_path, ann.camera)

        camera = ann.camera.copy()
        camera = self._apply_random_rotation(camera)
        camera = self._apply_random_shift(camera)
        camera = self._apply_random_zoom(camera)

        half = self.input_size // 2
        extent = (0, 0, -half, half, -half, half)

        img = self.slicer.get_data(
            camera,
            extent=extent,
            out_shape=(self.input_size, self.input_size),
            mask=False,
            order=1,
            axis=self.axis,
            rescale=True
        )

        msk = self.slicer.get_data(
            camera,
            extent=extent,
            out_shape=(self.input_size, self.input_size),
            mask=True,
            order=0,
            axis=self.axis,
        )

        x = torch.from_numpy(img).float()[None] / 255.0
        y = torch.from_numpy(msk).long()
        w = (y != 0).float()[None]

        # Ensure y is one-hot encoded
        y = torch.clamp(y - 1, min=0)
        y = F.one_hot(y, num_classes=self.num_classes)
        y = y.permute(2, 0, 1)

        return x, y, w, ann.class_idx

class RecencySampler(Sampler):
    """
    Ensures each batch contains at least one annotation per class.
    Favors recent annotations via temperature.
    """

    def __init__(self, tracker, batch_size=4, steps=20, recency_temp=20.0):
        self.tracker = tracker
        self.batch_size = batch_size
        self.steps = steps
        self.recency_temp = recency_temp

    def __len__(self):
        return self.steps

    def __iter__(self):
        anns = self.tracker.annotations()
        if not anns:
            return iter([])

        times = np.array([a.time_idx for a in anns], dtype=np.float32)
        tmax = times.max()
        weights = np.exp(-(tmax - times) / self.recency_temp)
        weights /= weights.sum()

        by_class = {}
        for i, a in enumerate(anns):
            by_class.setdefault(a.class_idx, []).append(i)

        rng = np.random.default_rng()

        for _ in range(self.steps):
            chosen = []

            # Cover each class
            for c, idxs in by_class.items():
                idxs = np.array(idxs)
                p = weights[idxs]
                p /= p.sum()
                chosen.append(int(rng.choice(idxs, p=p)))

            # Fill rest
            while len(chosen) < self.batch_size:
                chosen.append(int(rng.choice(len(anns), p=weights)))

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
):
    dataset = LiveTrainingDataset(
        tracker=tracker,
        input_size=input_size,
        num_classes=num_classes
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
