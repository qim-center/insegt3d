import asyncio

from interactive_unet.app import AppState, AppServices, UIBuilder, JobScheduler, ViewportRenderer, CallbackManager, InputHandler

from interactive_unet.tools import NavigatorTool, AnnotatorTool, FloodFillTool, LiveTrainerTool

class InteractiveSegmentationApp:
    
    def __init__(self, args):

        self.state = AppState.from_project(args.project_folder)
        self.services = AppServices(self.state)

        if args.num_classes is not None:
            self.state.train.num_classes = args.num_classes

        loop = asyncio.get_event_loop()
        self.scheduler = JobScheduler(loop=loop)

        self.renderer = ViewportRenderer(self.state)

        self.callbacks = CallbackManager(self.state, self.services, self.renderer, self.scheduler)

        self.tools = [
            NavigatorTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks),
            AnnotatorTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks),
            FloodFillTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks),
            LiveTrainerTool(self.state, self.services, self.renderer, self.scheduler, self.callbacks)
            ]
        
        self.input_handler = InputHandler(self.state, self.tools)

        self.ui = UIBuilder(
            state=self.state,
            callbacks=self.callbacks,
            input_handler=self.input_handler
        )
        self.callbacks.ui = self.ui
        self.renderer.viewport = self.ui.build()