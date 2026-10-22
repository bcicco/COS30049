"""Caps concurrent scoring jobs so a flood gets 503s instead of freezing the server."""

from collections.abc import Callable

import anyio
from fastapi import HTTPException


def busy() -> HTTPException:
    return HTTPException(
        status_code=503, detail="Server busy, try again shortly", headers={"Retry-After": "5"}
    )


class ScoreGate:
    """At most `workers` jobs run at once and `queue` more wait, anything beyond is refused."""

    def __init__(self, workers: int, queue: int, wait_seconds: float) -> None:
        self.slots = anyio.CapacityLimiter(workers)
        self.max_pending = workers + queue
        self.wait_seconds = wait_seconds
        self.pending = 0  # only touched on the event loop thread, so no lock needed

    async def run[T](self, fn: Callable[[str], T], arg: str) -> T:
        """run fn(arg) on a worker thread once a slot frees up, or raise a 503"""
        if self.pending >= self.max_pending:
            raise busy()
        self.pending += 1
        try:
            acquired = False
            with anyio.move_on_after(self.wait_seconds):
                await self.slots.acquire()
                acquired = True
            if not acquired:
                raise busy()
            try:
                # a thread keeps the event loop free for every other request meanwhile
                return await anyio.to_thread.run_sync(fn, arg)
            finally:
                self.slots.release()
        finally:
            self.pending -= 1
