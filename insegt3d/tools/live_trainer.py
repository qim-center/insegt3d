import cv2
import copy
from pathlib import Path

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from concurrent.futures import ThreadPoolExecutor

from insegt3d.app.scheduler import JobSpec
from insegt3d.tools.base_tool import BaseTool

from insegt3d.ml.loader2d import build_dataloader
from insegt3d.ml.unet2d import UNet2D


class LiveTrainerTool(BaseTool):

    def __init__(self, state, services, renderer, scheduler, callbacks):
        super().__init__(state, services, renderer, scheduler, callbacks)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ts = self.state.train
        tracker = self.services.tracker

        self.num_classes = ts.num_classes

        self.model_path = Path(self.state.data.project_path) / "model.ckpt"

        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, weights_only=False)
            
            self.model = checkpoint['model']

            # Ensure layers require grad
            for p in self.model.parameters():
                p.requires_grad_(True)

            self.num_classes = checkpoint['num_classes']
            print(f"Loaded existing model with {self.num_classes} classes...")
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

        self.training = False

        self.dataloader = build_dataloader(
            tracker,
            input_size=ts.input_size,
            num_classes=self.num_classes,
            batch_size=ts.batch_size,
            steps_per_epoch=ts.steps_per_epoch,
            recency_temp=ts.recency_temp
        )

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
                mode="latest",
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

    async def on_key(self, e):

        if e.action.keydown and not e.action.repeat:
            
            # Toggle prediction overlay
            if e.key == 'd':
                ui = self.state.ui
                ui.prediction.visible = not ui.prediction.visible
                self.callbacks.set_prediction_overlay()
                self.renderer.update()

    @torch.no_grad()
    def _ema_update(self):
        decay = self.state.train.ema_decay

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

        self.training = True

        # Update params with any changes to learning rate
        for params in self.opt.param_groups:
            params["lr"] = ts.lr

        self.model.train()

        loss_fn = ts.loss_fn

        scaler = torch.amp.GradScaler("cuda")

        for _epoch in range(int(ts.epochs)):

            # Dataset returns a length of zero until all classes exist
            if len(self.dataloader.dataset) == 0:
                continue

            for x, y, w in self.dataloader:

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

                with torch.autocast("cuda"):
                    pred = self.model(x)
                    loss = loss_fn(pred, y, w, axes=[0,2,3])
                    print(loss.item())

                scaler.scale(loss).backward()

                scaler.step(self.opt)
                scaler.update()

                self._ema_update()

        torch.save({
            "num_classes": self.num_classes,
            "model": self.model_ema,
        }, self.model_path)

        self.training = False
        self.scheduler.request("live_predict")

    def _predict(self):
        s = self.state
        camera = s.camera
        slicer = self.services.slicer

        image = slicer.get_data(camera, zoom_override=1.0, order=1, rescale=True)

        image = torch.from_numpy(image[None,None,:,:]).to(self.device, dtype=torch.float32) / 255.0

        with torch.no_grad():
            self.model_ema.eval()
            prediction = self.model_ema(image).argmax(1).squeeze(0).cpu().numpy() + 1

        scale = 1 / camera.zoom
        prediction = cv2.resize(prediction, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        self.renderer.update(prediction=prediction)