import asyncio
import time
import threading
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Literal
from functools import partial

AsyncFn = Callable[..., Awaitable[None]]
SyncFn = Callable[..., None]
Mode = Literal["latest", "drop"]


@dataclass
class JobSpec:
    max_hz: Optional[float] = None
    idle_after: Optional[float] = None
    idle_args: tuple[Any, ...] = ()
    idle_kwargs: Optional[dict[str, Any]] = None

    mode: Mode = "latest"

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
        executor = spec.executor
        owned_executor = None
        if executor is None and spec.sequential_executor:
            executor = owned_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        if executor is None:
            executor = self._default_executor

        async def wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            # run_in_executor does NOT accept **kwargs; bind them first
            await loop.run_in_executor(executor, partial(fn, *args, **kwargs))

        self.jobs[name] = Job(
            wrapper, spec,
            loop=self.loop, loop_thread_id=self._loop_thread_id,
            owned_executor=owned_executor,
        )

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
            job.shutdown()

    def shutdown(self) -> None:
        for job in self.jobs.values():
            job.shutdown()
        self._default_executor.shutdown(wait=False)


class Job:
    def __init__(
        self, fn: AsyncFn, spec: JobSpec, *,
        loop: asyncio.AbstractEventLoop, loop_thread_id: int,
        owned_executor: Optional[concurrent.futures.Executor] = None,
    ):
        self.fn = fn
        self.loop = loop
        self._loop_thread_id = loop_thread_id

        self._owned_executor = owned_executor

        self.mode = spec.mode
        self.min_interval = (1.0 / spec.max_hz) if spec.max_hz else 0.0

        self.idle_after = spec.idle_after
        self.idle_args = spec.idle_args
        self.idle_kwargs = spec.idle_kwargs or {}
        self.idle_gen_guard = bool(spec.idle_gen_guard)

        self._latest: Optional[tuple[tuple[Any, ...], dict[str, Any]]] = None

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

        self._latest = (args, kwargs)

        self._ensure_drain()
        self._arm_idle(self._gen)

    def cancel(self) -> None:
        if threading.get_ident() != self._loop_thread_id:
            self.loop.call_soon_threadsafe(self.cancel)
            return

        if self._drain_task:
            self._drain_task.cancel()
        if self._idle_task:
            self._idle_task.cancel()
        self._latest = None

    def shutdown(self) -> None:
        self.cancel()
        if self._owned_executor is not None:
            self._owned_executor.shutdown(wait=False, cancel_futures=True)

    def _has_pending(self) -> bool:
        return self._latest is not None

    def _pop_next(self):
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

        # Always create tasks on the scheduler's loop
        self._idle_task = self.loop.create_task(idle())
