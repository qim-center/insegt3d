import cv2
import copy
import json
import threading
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch

from insegt3d.ml import metrics, predict2d
from insegt3d.ml.loader2d import LiveTrainingDataset, prefetch
from insegt3d.ml.unet2d import UNet2D, load_checkpoint, save_checkpoint
from insegt3d.volume.intensity import robust_normalize


class LiveTrainer:

    def __init__(self, state, services, renderer, scheduler, callbacks):
        self.state = state
        self.services = services
        self.renderer = renderer
        self.scheduler = scheduler
        self.callbacks = callbacks

        # A CUDA GPU, or the GPU of an Apple Silicon Mac (MPS)
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        self.device = accelerator.type if accelerator else "cpu"

        ts = self.state.train

        self.model_path = Path(self.state.data.project_path) / "model.ckpt"
        self.classes_path = Path(self.state.data.project_path) / "classes.json"
        self.model = None
        self._closed = False
        self._steps_done = 0
        self._loss = None

        if self.classes_path.exists():
            classes = json.loads(self.classes_path.read_text())
            self.state.annot.set_colors(classes + [c for c in self.state.annot.colors if c not in classes])
            ts.num_classes = len(classes)

        checkpoint = None
        if self.model_path.exists():
            checkpoint = load_checkpoint(self.model_path)

            config = checkpoint['config']
            ts.num_classes = config['num_classes']
            ts.architecture = config['architecture']
            ts.encoder_name = config['encoder_name']
            ts.slab_size = config['num_channels']
            ts.level = checkpoint['level']

            print(f"Loaded existing model with {ts.num_classes} classes...")
            ts.model_locked = True

        self._model_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self.dataset = LiveTrainingDataset(self.services.tracker)

        # GPU-heavy jobs share one thread so they run one at a time
        gpu = ThreadPoolExecutor(max_workers=1)

        self.scheduler.register_sync("init_model", self._init_model, mode="queue", executor=gpu)
        self.scheduler.register_sync("live_train", self._train, max_hz=1, executor=gpu)
        self.scheduler.register_sync("live_predict", self._predict, max_hz=1)
        self.scheduler.register_sync("reset_model", self._reset_model, mode="drop", executor=gpu)
        self.scheduler.register_sync("change_classes", self._change_classes, mode="queue", executor=gpu)
        self.scheduler.register_sync("predict_volumes", self.callbacks.run_predict_volumes, mode="drop", executor=gpu)

        self._train_job = self.scheduler.jobs["live_train"]
        self.scheduler.request("init_model", checkpoint)

    def _new_model(self):
        ts = self.state.train
        return UNet2D(
            num_channels=ts.slab_size,
            num_classes=ts.num_classes,
            architecture=ts.architecture,
            encoder_name=ts.encoder_name,
        )

    def _init_model(self, checkpoint):
        try:
            model = UNet2D.from_checkpoint(checkpoint) if checkpoint else self._new_model()
        except Exception as e:
            self.callbacks.notify(f'Could not build the model: {e}', type='negative')
            raise

        self._set_model(model)

    def _set_model(self, model):
        model.to(self.device)

        # Live prediction runs on a copy of the weights, so it never waits for a training step
        with self._model_lock, self._snapshot_lock:
            self.model = model
            self._snapshot = copy.deepcopy(model).eval()

        self.opt = torch.optim.AdamW(model.parameters(), lr=self.state.train.lr)
        self.scaler = torch.amp.GradScaler(self.device)

    def _save(self):
        with self._model_lock:
            save_checkpoint(self.model, self.state.train.level, self.model_path)

    def predict_volumes(self, zarr_files, project_path, **kwargs):
        if not self.model_path.exists():
            raise FileNotFoundError('There is no trained model yet.')

        with self._model_lock:
            predict2d.predict_all_volumes(zarr_files, self.model, self.state.train.level, project_path / 'predictions', project_path / 'temp', **kwargs)

    def _interrupted(self):
        ts = self.state.train
        return self._closed or self.model is None or ts.predicting or ts.changing_classes or not ts.live_training_enabled or not ts.level_available

    def _report_progress(self, remaining):
        self.callbacks.update_live_train_progress(self._steps_done, self._steps_done + remaining, self._loss)

    def _end_progress(self):
        if self._steps_done:
            self._report_progress(0)
        self._steps_done = 0

    def _train(self):
        ts = self.state.train

        if self._interrupted():
            self._end_progress()
            return

        labeled = {a.class_idx for a in self.services.tracker.annotations()}
        unlabeled = [c for c in range(1, ts.num_classes + 1) if c not in labeled]
        if unlabeled:
            self._end_progress()
            self.callbacks.show_live_train_hint(f'Annotate class {unlabeled[0]} to train')
            return

        if not ts.model_locked:
            ts.model_locked = True
            self.callbacks.set_model_lock()

        for params in self.opt.param_groups:
            params["lr"] = ts.lr

        self.model.train()

        loss_fn = ts.loss_fn

        total_steps = ts.steps_per_epoch
        trained = False

        for step, (x, y, w) in enumerate(prefetch(self.dataset.batches(ts, self.services.slicer.ts_context)), start=1):

            if self._interrupted():
                break

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            w = w.to(self.device, non_blocking=True)

            flips = [dim for dim in (2, 3) if torch.rand(1) < 0.5]
            if flips:
                x, y, w = (t.flip(flips) for t in (x, y, w))

            # Augmentation can shift an annotation out of the crop, so skip batches that lost a class
            labeled_classes = ((y * w).sum(dim=(0, 2, 3)) > 0).sum()
            if labeled_classes < min(ts.num_classes, ts.batch_size):
                continue

            with self._model_lock:
                self.opt.zero_grad(set_to_none=True)

                with torch.autocast(self.device):
                    logits = self.model.logits(x)
                    loss = loss_fn(logits.softmax(1), y, w)

                    if ts.scnp_enabled:
                        penalized = metrics.scnp(logits.float(), y, w, ts.scnp_size)
                        loss = loss + loss_fn(penalized.softmax(1), y, w)

                self.scaler.scale(loss).backward()

                self.scaler.step(self.opt)
                self.scaler.update()

                with self._snapshot_lock:
                    self._snapshot.load_state_dict(self.model.state_dict())

            trained = True
            self._loss = loss.item()
            self._steps_done += 1
            self._report_progress(total_steps - step + self._train_job.pending * total_steps)

        if trained:
            self._save()

        if self._interrupted() or not self._train_job.pending:
            self._end_progress()

        if not self._interrupted():
            self.scheduler.request("live_predict")

    def _reset_model(self):
        ts = self.state.train

        if ts.predicting:
            return

        try:
            model = self._new_model()
        except Exception as e:
            self.callbacks.notify(f'Could not build {ts.architecture} with encoder {ts.encoder_name}: {e}', type='negative')
            raise

        if self.model_path.exists():
            self.model_path.unlink()

        self._set_model(model)

        ts.model_locked = False
        self.callbacks.set_model_lock()

        self.renderer.clear("prediction")
        self.renderer.update()

        if self.device != "cpu":
            torch.accelerator.empty_cache()

    def _change_classes(self, remove=None):
        ts = self.state.train
        keep = [c for c in range(ts.num_classes) if c != remove]
        try:
            if remove is not None:
                # Remap mask labels: the removed class becomes 0 (unannotated) and later classes shift down
                lut = np.zeros(256, np.uint8)
                lut[np.add(keep, 1)] = np.arange(1, len(keep) + 1)
                self.services.history.relabel(Path(self.state.data.project_path) / "masks", lut)
                self.state.annot.remove_color(remove)
                self.renderer.clear("prediction")
                self.renderer.update()
                self.scheduler.request("nav_overlays")

            with self._model_lock:
                self.model.set_classes(keep, int(remove is None))
            ts.num_classes = self.model.num_classes
            if remove is None:
                self.state.annot.color_idx = ts.num_classes - 1
            self._set_model(self.model)
            if self.model_path.exists():
                self._save()
            self.classes_path.write_text(json.dumps(self.state.annot.colors[:ts.num_classes]))
        finally:
            ts.changing_classes = False

        self.callbacks.refresh_button_palette()
        self.scheduler.request("live_train")
        self.scheduler.request("live_predict")

    def _predict(self):
        s = self.state
        slicer = self.services.slicer

        if self.model is None or slicer.images is None or s.train.predicting or s.train.changing_classes or not s.train.level_available or not s.ui.prediction.visible:
            return

        camera = s.camera.copy()

        size = s.train.input_size
        half = size // 2
        slab_size = s.train.slab_size
        d0 = -(slab_size // 2)

        spacing = slicer.voxel_sizes[s.train.level].min()
        image = slicer.get_data(
            camera,
            extent=(d0, d0 + slab_size - 1, -half, half, -half, half),
            out_shape=(slab_size, size, size),
            zoom_override=spacing,
            order=1,
            level=s.train.level,
        )

        image = robust_normalize(image.reshape(slab_size, size, size))
        image = torch.from_numpy(image[None]).to(self.device)

        with self._snapshot_lock, torch.no_grad(), torch.autocast(self.device):
            prediction = (self._snapshot.logits(image).argmax(1)[0] + 1).to(torch.uint8).cpu().numpy()

        # Rescale from training resolution to viewport pixels, cropping to the visible area first
        scale = (s.ui.viewport_shape[0] / s.nav.slice_shape[0]) * spacing / camera.zoom
        crop_y, crop_x = (max(0, (size - int(np.ceil(n / scale)) - 2) // 2) for n in s.ui.viewport_shape)
        prediction = prediction[crop_y:size - crop_y, crop_x:size - crop_x]
        prediction = cv2.resize(prediction, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        self.renderer.update(prediction=prediction, version=camera.version)

    def close(self):
        self._closed = True
