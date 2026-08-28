from concurrent.futures import ThreadPoolExecutor

from insegt3d.app.scheduler import JobSpec

class BaseTool:
    """
    Base class for interactive tools: owns the shared app references and the
    scheduler lanes a tool registers for its background work.
    """

    def __init__(self, state, services, renderer, scheduler, callbacks):
        self.state = state
        self.services = services
        self.renderer = renderer
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.ignore_pointer = False
        self.ignore_key = False
        self._executors = []

    def register_latest_job(self, name, fn, max_hz=60):
        
        executor = ThreadPoolExecutor(max_workers=1)
        self._executors.append(executor)
        self.scheduler.register_sync(
            name,
            fn=fn,
            spec=JobSpec(
                max_hz=max_hz,
                mode="latest",
                executor=executor,
                sequential_executor=False,
            ),
        )

    async def on_pointer(self, e):
        return None

    async def on_key(self, e):
        return None

    def close(self):
        for executor in self._executors:
            executor.shutdown(wait=False, cancel_futures=True)
        self._executors.clear()
