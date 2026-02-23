import asyncio
import time
import threading
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Literal
from collections import deque
from functools import partial  # <-- NEW

AsyncFn = Callable[..., Awaitable[None]]
SyncFn = Callable[..., None]
Mode = Literal["latest", "queue", "drop"]


@dataclass
class JobSpec:
    max_hz: Optional[float] = None
    idle_after: Optional[float] = None
    idle_args: tuple[Any, ...] = ()
    idle_kwargs: Optional[dict[str, Any]] = None

    mode: Mode = "latest"
    max_queue: int = 64

    executor: Optional[concurrent.futures.Executor] = None
    sequential_executor: bool = True
    idle_gen_guard: bool = True


class JobScheduler:
    """
    Thread-safe scheduler:
    - request() may be called from worker threads.
    - internal asyncio tasks are always created on the scheduler's event loop thread.
    """

    def __init__(self, *, default_workers: int = 2, loop: Optional[asyncio.AbstractEventLoop] = None):
        # IMPORTANT: this must run on the asyncio thread (or pass loop explicitly)
        if loop is None:
            loop = asyncio.get_running_loop()

        self.loop = loop
        self._loop_thread_id = threading.get_ident()

        self._default_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=default_workers
        )
        self.jobs: dict[str, Job] = {}

    def register_async(self, name: str, fn: AsyncFn, *, spec: JobSpec) -> None:
        self.jobs[name] = Job(fn, spec, loop=self.loop, loop_thread_id=self._loop_thread_id)

    def register_sync(self, name: str, fn: SyncFn, *, spec: JobSpec) -> None:
        exec_ = spec.executor
        if exec_ is None and spec.sequential_executor:
            exec_ = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        if exec_ is None:
            exec_ = self._default_executor

        async def wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            # run_in_executor does NOT accept **kwargs; bind them first
            call = partial(fn, *args, **kwargs)
            await loop.run_in_executor(exec_, call)

        self.jobs[name] = Job(wrapper, spec, loop=self.loop, loop_thread_id=self._loop_thread_id)

    def request(self, name: str, *args, **kwargs) -> None:
        """
        Safe to call from any thread.
        If called off the event loop thread, it forwards the request to the loop thread.
        """
        try:
            job = self.jobs[name]
        except KeyError as e:
            raise KeyError(f"Unknown job '{name}'. Registered: {list(self.jobs)}") from e

        # If we're not on the loop thread, bounce to the loop thread
        if threading.get_ident() != self._loop_thread_id:
            self.loop.call_soon_threadsafe(job.request, *args, **kwargs)
            return

        job.request(*args, **kwargs)

    def unregister(self, name: str) -> None:
        job = self.jobs.pop(name, None)
        if job:
            job.cancel()

    def shutdown(self) -> None:
        for job in self.jobs.values():
            job.cancel()
        self._default_executor.shutdown(wait=False)


class Job:
    def __init__(self, fn: AsyncFn, spec: JobSpec, *, loop: asyncio.AbstractEventLoop, loop_thread_id: int):
        self.fn = fn
        self.loop = loop
        self._loop_thread_id = loop_thread_id

        self.mode = spec.mode
        self.min_interval = (1.0 / spec.max_hz) if spec.max_hz else 0.0

        self.idle_after = spec.idle_after
        self.idle_args = spec.idle_args
        self.idle_kwargs = spec.idle_kwargs or {}
        self.idle_gen_guard = bool(spec.idle_gen_guard)

        self._latest: Optional[tuple[tuple[Any, ...], dict[str, Any]]] = None
        self._queue = deque(maxlen=max(1, int(spec.max_queue)))

        self._drain_task: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None

        self._busy = False
        self._last_start = 0.0
        self._gen = 0  # bumps on every request

    def request(self, *args, **kwargs) -> None:
        """
        MUST be executed on the loop thread.
        (Scheduler.request ensures this by forwarding across threads.)
        """
        self._gen += 1

        if self.mode == "drop" and (self._busy or self._has_pending()):
            return

        if self.mode == "queue":
            self._queue.append((args, kwargs))
        else:  # latest / drop
            self._latest = (args, kwargs)

        self._ensure_drain()
        self._arm_idle(self._gen)

    def cancel(self) -> None:
        # Note: cancel can be called from any thread; forward if needed.
        if threading.get_ident() != self._loop_thread_id:
            self.loop.call_soon_threadsafe(self.cancel)
            return

        if self._drain_task:
            self._drain_task.cancel()
        if self._idle_task:
            self._idle_task.cancel()
        self._latest = None
        self._queue.clear()

    def _has_pending(self) -> bool:
        return bool(self._queue) if self.mode == "queue" else (self._latest is not None)

    def _pop_next(self):
        if self.mode == "queue":
            return self._queue.popleft() if self._queue else None
        item = self._latest
        self._latest = None
        return item

    def _ensure_drain(self) -> None:
        # Always create tasks on the scheduler's loop (not "current thread loop")
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = self.loop.create_task(self._drain())

    async def _drain(self) -> None:
        while True:
            item = self._pop_next()
            if item is None:
                return

            # throttle
            if self.min_interval:
                now = time.time()
                wait = (self._last_start + self.min_interval) - now
                if wait > 0:
                    await asyncio.sleep(wait)

            self._busy = True
            self._last_start = time.time()
            try:
                args, kwargs = item
                await self.fn(*args, **kwargs)
            finally:
                self._busy = False

    def _arm_idle(self, gen: int) -> None:
        if not self.idle_after:
            return

        # cancel previous idle task
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()

        async def idle():
            try:
                await asyncio.sleep(self.idle_after)

                if self.idle_gen_guard and gen != self._gen:
                    return

                # only fire if truly quiet
                if not self._busy and not self._has_pending():
                    await self.fn(*self.idle_args, **self.idle_kwargs)
            except asyncio.CancelledError:
                pass

        # Create idle task on the scheduler loop
        self._idle_task = self.loop.create_task(idle())
