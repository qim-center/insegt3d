import asyncio

from nicegui import context

from insegt3d.app.state import AppState, AppServices
from insegt3d.app.ui.ui import UIBuilder
from insegt3d.app.scheduler import JobScheduler
from insegt3d.app.renderer import ViewportRenderer
from insegt3d.app.callbacks import CallbackManager
from insegt3d.app.input_handler import InputHandler

from insegt3d.tools import NavigatorTool, AnnotatorTool, FloodFillTool, MaskFillTool
from insegt3d.ml.live_trainer import LiveTrainer

class InteractiveSegmentationApp:

    def __init__(self, args):

        self.state = AppState.from_project(args.project_folder)
        self.services = AppServices(self.state)

        self.state.train.num_classes = args.num_classes

        loop = asyncio.get_running_loop()
        self.scheduler = JobScheduler(loop=loop)

        self.renderer = ViewportRenderer(self.state)

        self.callbacks = CallbackManager(self.state, self.services, self.renderer, self.scheduler)
        self.callbacks.client = context.client

        self.live_trainer = LiveTrainer(self.state, self.services, self.renderer, self.scheduler, self.callbacks)

        annotator = AnnotatorTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks)

        self.tools = [
            NavigatorTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks),
            annotator,
            FloodFillTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks, annotator),
            MaskFillTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks, annotator),
        ]

        self.input_handler = InputHandler(self.state, self.tools)

        self.ui = UIBuilder(
            state=self.state,
            callbacks=self.callbacks,
            input_handler=self.input_handler
        )
        self.callbacks.ui = self.ui
        self.renderer.viewport = self.ui.build()

        context.client.on_delete(self.close)

    def close(self):
        self.scheduler.shutdown()
        self.callbacks.close()
        self.ui.navigator.close()
        for tool in self.tools:
            tool.close()
        self.live_trainer.close()
