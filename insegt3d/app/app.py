from pathlib import Path

from nicegui import context, ui

from insegt3d.app.state import AppState, AppServices, DataState
from insegt3d.app.ui.ui import UIBuilder
from insegt3d.app.scheduler import JobScheduler
from insegt3d.app.renderer import ViewportRenderer
from insegt3d.app.callbacks import CallbackManager
from insegt3d.app.input_handler import InputHandler

from insegt3d.tools.navigate import NavigatorTool
from insegt3d.tools.annotate import AnnotatorTool
from insegt3d.tools.flood_fill import FloodFillTool
from insegt3d.tools.mask_fill import MaskFillTool
from insegt3d.tools.regions import RegionTool
from insegt3d.ml.live_trainer import LiveTrainer

class InteractiveSegmentationApp:

    def __init__(self, args, project_path, on_switch, on_shutdown):

        self.state = AppState(data=DataState(Path(project_path)))
        self.state.data.cache_size_mb = int(args.cache_gb * 1024)
        self.services = AppServices(self.state)

        self.scheduler = JobScheduler()

        self.renderer = ViewportRenderer(self.state, self.scheduler)

        self.callbacks = CallbackManager(self.state, self.services, self.renderer, self.scheduler)

        self.live_trainer = LiveTrainer(self.state, self.services, self.renderer, self.scheduler, self.callbacks)
        self.callbacks.trainer = self.live_trainer

        self.tools = [
            NavigatorTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks),
            AnnotatorTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks),
            FloodFillTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks),
            MaskFillTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks),
            RegionTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks),
        ]

        self.input_handler = InputHandler(self.tools)
        self.on_switch = on_switch
        self.on_shutdown = on_shutdown
        self._closed = False

    def open_view(self):
        if self.callbacks.client is not None:
            with self.callbacks.client:
                self.callbacks.client.content.clear()
                ui.label('insegt3d was opened in another tab. Reload this page to use it here.')

        view = UIBuilder(
            state=self.state,
            callbacks=self.callbacks,
            input_handler=self.input_handler,
            scheduler=self.scheduler,
            on_switch=self.on_switch,
            on_shutdown=self.on_shutdown,
        )
        self.callbacks.ui, self.callbacks.client = view, context.client
        self.renderer.viewport, self.renderer.overlay = view.build()
        self.callbacks.refresh_view()

        context.client.on_delete(self._close_view)

    def _close_view(self, client):
        if client is self.callbacks.client:
            self.callbacks.ui = self.callbacks.client = None
            self.renderer.viewport = self.renderer.overlay = None

    async def close(self):
        if self._closed:
            return
        self.services.regions.save()
        self._closed = True
        self.input_handler.enabled = False
        self.callbacks._closing = True
        self.callbacks._predict_cancel_event.set()
        self.live_trainer.close()
        async with self.callbacks._open_lock:
            await self.scheduler.drain()
        self.scheduler.shutdown()
