import asyncio
import time
import logging
import threading
import concurrent.futures
from collections import deque
from typing import Any, Awaitable, Callable, Optional, Literal
from functools import partial

AsyncFn = Callable[..., Awaitable[None]]
SyncFn = Callable[..., None]
Mode = Literal["latest", "drop", "queue"]

log = logging.getLogger(__name__)


class JobScheduler:
    """Named, rate-limited jobs on the asyncio loop it is created on. request() and call_soon() are thread-safe."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self._loop_thread_id = threading.get_ident()

        self.jobs: dict[str, Job] = {}

    def register_async(self, name: str, fn: AsyncFn, **spec) -> None:
        self.jobs[name] = Job(name, fn, loop=self.loop, loop_thread_id=self._loop_thread_id, **spec)

    def register_sync(self, name: str, fn: SyncFn, executor: Optional[concurrent.futures.Executor] = None, **spec) -> None:
        executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1)

        async def wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(executor, partial(fn, *args, **kwargs))

        self.jobs[name] = Job(
            name, wrapper,
            loop=self.loop, loop_thread_id=self._loop_thread_id,
            executor=executor, **spec,
        )

    def request(self, name: str, *args, **kwargs) -> None:
        job = self.jobs[name]

        if threading.get_ident() != self._loop_thread_id:
            self.loop.call_soon_threadsafe(job.request, *args, **kwargs)
            return

        job.request(*args, **kwargs)

    def call_soon(self, fn: Callable[..., Any], *args, **kwargs) -> None:
        if threading.get_ident() == self._loop_thread_id:
            fn(*args, **kwargs)
        else:
            self.loop.call_soon_threadsafe(partial(fn, *args, **kwargs))

    def shutdown(self) -> None:
        for job in self.jobs.values():
            job.shutdown()


class Job:
    """
    Runs `fn` on the loop, one call at a time and at most `max_hz` times per second.
    mode: 'latest' keeps only the newest pending request, 'drop' ignores requests while busy, 'queue' runs them all.
    idle_after: once requests stop for this many seconds, `fn` runs once more with `idle_kwargs`.
    """
    def __init__(
        self, name: str, fn: AsyncFn, *,
        loop: asyncio.AbstractEventLoop, loop_thread_id: int,
        executor: Optional[concurrent.futures.Executor] = None,
        max_hz: Optional[float] = None, mode: Mode = "latest",
        idle_after: Optional[float] = None, idle_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.fn = fn
        self.loop = loop
        self._loop_thread_id = loop_thread_id

        self._executor = executor

        self.mode = mode
        self.min_interval = (1.0 / max_hz) if max_hz else 0.0

        self.idle_after = idle_after
        self.idle_kwargs = idle_kwargs or {}

        self._pending: deque[tuple[tuple[Any, ...], dict[str, Any]]] = deque()

        self._drain_task: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None

        self._busy = False
        self._closed = False
        self._last_start = 0.0

    def request(self, *args, **kwargs) -> None:
        if self._closed or (self.mode == "drop" and (self._busy or self._pending)):
            return

        if self.mode == "latest":
            self._pending.clear()
        self._pending.append((args, kwargs))

        self._ensure_drain()
        self._arm_idle()

    @property
    def pending(self) -> int:
        return len(self._pending)

    def cancel(self) -> None:
        if threading.get_ident() != self._loop_thread_id:
            self.loop.call_soon_threadsafe(self.cancel)
            return

        if self._drain_task:
            self._drain_task.cancel()
        if self._idle_task:
            self._idle_task.cancel()
        self._pending.clear()

    def shutdown(self) -> None:
        self._closed = True
        self.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _ensure_drain(self) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = self.loop.create_task(self._drain())

    async def _drain(self) -> None:
        while self._pending:
            args, kwargs = self._pending.popleft()

            if self.min_interval:
                now = time.time()
                wait = (self._last_start + self.min_interval) - now
                if wait > 0:
                    await asyncio.sleep(wait)

            self._busy = True
            self._last_start = time.time()
            try:
                await self.fn(*args, **kwargs)
            except Exception:
                log.exception("Job '%s' failed", self.name)
            finally:
                self._busy = False

    def _arm_idle(self) -> None:
        if not self.idle_after:
            return

        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()

        async def idle():
            try:
                await asyncio.sleep(self.idle_after)

                if not self._busy and not self._pending:
                    await self.fn(**self.idle_kwargs)
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("Job '%s' failed", self.name)

        self._idle_task = self.loop.create_task(idle())
