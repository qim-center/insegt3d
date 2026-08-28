import cv2
import copy
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from insegt3d.app.scheduler import JobSpec

from insegt3d.ml.loader2d import build_dataloader
from insegt3d.ml.unet2d import UNet2D
from insegt3d.volume.intensity import robust_normalize


class LiveTrainer:

    def __init__(self, state, services, renderer, scheduler, callbacks):
        self.state = state
        self.services = services
        self.renderer = renderer
        self.scheduler = scheduler
        self.callbacks = callbacks

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ts = self.state.train

        self.num_classes = ts.num_classes

        self.model_path = Path(self.state.data.project_path) / "model.ckpt"

        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, weights_only=False)

            self.model = checkpoint['model']

            # Ensure layers require grad
            for p in self.model.parameters():
                p.requires_grad_(True)

            self.num_classes = checkpoint['num_classes']

            ts.architecture = self.model.architecture
            ts.encoder_name = self.model.encoder_name

            print(f"Loaded existing model with {self.num_classes} classes...")
            ts.model_locked = True
        else:
            self.model = UNet2D(
                num_classes=self.num_classes,
                architecture=ts.architecture,
                encoder_name=ts.encoder_name,
                pretrained=ts.pretrained
            )

        self.model_ema = copy.deepcopy(self.model).eval()
        for p in self.model_ema.parameters():
            p.requires_grad_(False)

        self.model.to(self.device)
        self.model_ema.to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=ts.lr)

        self._model_lock = threading.Lock()

        self._build_dataloader()

        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            # v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
            # v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.02)], p=0.5),
        ])

        self._train_exec = ThreadPoolExecutor(max_workers=1)
        self._pred_exec = ThreadPoolExecutor(max_workers=1)

        self.scheduler.register_sync(
            "live_train",
            fn=self._train,
            spec=JobSpec(
                max_hz=1,
                mode="drop",
                executor=self._train_exec,
                sequential_executor=False,
                idle_gen_guard=True,
            ),
        )
        self.scheduler.register_sync(
            "live_predict",
            fn=self._predict,
            spec=JobSpec(
                max_hz=1,
                mode="latest",
                executor=self._pred_exec,
                sequential_executor=False,
            ),
        )
        self.scheduler.register_sync(
            "reset_model",
            fn=self._reset_model,
            spec=JobSpec(
                mode="drop",
                executor=self._train_exec,
                sequential_executor=False,
            ),
        )

    def _build_dataloader(self):
        ts = self.state.train

        self.dataloader = build_dataloader(
            self.services.tracker,
            input_size=ts.input_size,
            num_classes=self.num_classes,
            batch_size=ts.batch_size,
            steps_per_epoch=ts.steps_per_epoch,
            recency_temp=ts.recency_temp,
            cache_size_mb=ts.cache_size_mb,
        )

    @torch.no_grad()
    def _ema_update(self):
        decay = self.state.train.ema_decay

        with self._model_lock:
            ema_sd = self.model_ema.state_dict()
            sd = self.model.state_dict()

            for k, v_ema in ema_sd.items():
                v = sd[k]
                if torch.is_floating_point(v_ema):
                    v_ema.mul_(decay).add_(v, alpha=1.0 - decay)
                else:
                    v_ema.copy_(v)

    def _train(self):
        ts = self.state.train

        if ts.predicting:
            return

        if not ts.live_training_enabled:
            return

        if not ts.model_locked:
            ts.model_locked = True
            self.callbacks.set_model_lock()

        # Update params with any changes to learning rate
        for params in self.opt.param_groups:
            params["lr"] = ts.lr

        # Rebuild dataloader in case batch size / sampling params changed
        self._build_dataloader()

        self.model.train()

        loss_fn = ts.loss_fn

        scaler = torch.amp.GradScaler(self.device)

        total_steps = len(self.dataloader)

        for _epoch in range(int(ts.epochs)):

            if ts.predicting or not ts.live_training_enabled:
                break

            # Dataset is empty until at least one region has been annotated
            if len(self.dataloader.dataset) == 0:
                continue

            for step, (x, y, w) in enumerate(self.dataloader, start=1):

                if ts.predicting or not ts.live_training_enabled:
                    break

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                w = w.to(self.device, non_blocking=True)

                x = tv_tensors.Image(x)
                y = tv_tensors.Mask(y)
                w = tv_tensors.Mask(w)

                x, y, w = self.transforms(x, y, w)

                # Skip batch if it doesn't have at least one pixel in every class
                has_all_classes = (y.sum(dim=(0, 2, 3)) > 0).all()
                if not has_all_classes:
                    continue

                self.opt.zero_grad(set_to_none=True)

                with torch.autocast(self.device):
                    pred = self.model(x)
                    loss = loss_fn(pred, y, w, axes=[0,2,3])

                scaler.scale(loss).backward()

                scaler.step(self.opt)
                scaler.update()

                self._ema_update()

                self.callbacks.update_live_train_progress(step, total_steps, loss.item())

        torch.save({
            "num_classes": self.num_classes,
            "model": self.model_ema,
        }, self.model_path)

        self.scheduler.request("live_predict")

    def _reset_model(self):
        ts = self.state.train

        if self.model_path.exists():
            self.model_path.unlink()

        self.num_classes = ts.num_classes

        self.model = UNet2D(
            num_classes=self.num_classes,
            architecture=ts.architecture,
            encoder_name=ts.encoder_name,
            pretrained=ts.pretrained
        )

        model_ema = copy.deepcopy(self.model).eval()
        for p in model_ema.parameters():
            p.requires_grad_(False)

        self.model.to(self.device)
        model_ema.to(self.device)

        with self._model_lock:
            self.model_ema = model_ema

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=ts.lr)

        self._build_dataloader()

        ts.model_locked = False
        self.callbacks.set_model_lock()

        self.scheduler.request("live_predict")

    def _predict(self):
        s = self.state

        if s.train.predicting:
            return
        
        rev = s.nav.revision
        camera = s.camera.copy()
        slicer = self.services.slicer

        image = slicer.get_data(camera, zoom_override=1.0, order=1)

        image = robust_normalize(image)
        image = torch.from_numpy(image[None,None,:,:]).to(self.device)

        with self._model_lock, torch.no_grad():
            self.model_ema.eval()
            prediction = self.model_ema(image).argmax(1).squeeze(0).cpu().numpy() + 1

        if rev != s.nav.revision:
            return

        scale = 1 / camera.zoom
        prediction = cv2.resize(prediction, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        self.renderer.update(prediction=prediction)

    def close(self):

        self._train_exec.shutdown(wait=True, cancel_futures=True)
        self._pred_exec.shutdown(wait=True, cancel_futures=True)

        del self.model
        del self.model_ema
        del self.opt

        if self.device == "cuda":
            torch.cuda.empty_cache()
