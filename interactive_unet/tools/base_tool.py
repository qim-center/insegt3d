class BaseTool:
    """
    Base Tool class.
    """

    def __init__(self, state, services, renderer, scheduler, callbacks):
        self.state = state
        self.services = services
        self.renderer = renderer
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.ignore_pointer = False
        self.ignore_key = False

    async def on_pointer(self, e):
        return None

    async def on_key(self, e):
        return None