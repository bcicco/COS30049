import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import httpx
import pytest

from aivhuman_api.gate import ScoreGate
from aivhuman_api.main import app
from aivhuman_api.routes.score import get_scorer
from tests.conftest import TEXT, FakeScorer

pytestmark = pytest.mark.anyio
DELAY = 0.5


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_answers_while_scoring() -> None:
    app.dependency_overrides[get_scorer] = lambda: FakeScorer(delay=DELAY)
    async with client() as c, anyio.create_task_group() as tg:

        async def slow() -> None:
            await c.post("/api/score", json={"text": TEXT})

        tg.start_soon(slow)
        await anyio.sleep(0.1)  # scoring is now running
        start = time.perf_counter()
        r = await c.get("/api/health")
        elapsed = time.perf_counter() - start
    assert r.status_code == 200
    assert elapsed < DELAY / 5


async def test_excess_requests_get_503() -> None:
    app.dependency_overrides[get_scorer] = lambda: FakeScorer(delay=DELAY)
    app.state.gate = ScoreGate(workers=1, queue=2, wait_seconds=30)
    codes: list[int] = []

    async with client() as c:

        async def one() -> None:
            codes.append((await c.post("/api/score", json={"text": TEXT})).status_code)

        async with anyio.create_task_group() as tg:
            for _ in range(8):
                tg.start_soon(one)

    # 1 running + 2 queued are served, the other 5 are refused straight away
    assert sorted(codes) == [200] * 3 + [503] * 5


async def test_waiting_too_long_is_503() -> None:
    app.dependency_overrides[get_scorer] = lambda: FakeScorer(delay=DELAY)
    app.state.gate = ScoreGate(workers=1, queue=4, wait_seconds=DELAY / 5)
    codes: list[int] = []

    async with client() as c:

        async def one() -> None:
            codes.append((await c.post("/api/score", json={"text": TEXT})).status_code)

        async with anyio.create_task_group() as tg:
            tg.start_soon(one)
            await anyio.sleep(0.05)
            tg.start_soon(one)

    assert sorted(codes) == [200, 503]
