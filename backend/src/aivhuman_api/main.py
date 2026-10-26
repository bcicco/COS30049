"""App entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aivhuman.mil.infer import Scorer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aivhuman_api.config import settings
from aivhuman_api.gate import ScoreGate
from aivhuman_api.limits import BodySizeLimit
from aivhuman_api.routes import health, score


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # gpt-2 + spacy take a few seconds to load, so once per process
    app.state.scorer = Scorer(
        settings.model_dir / "model.pt",
        settings.model_dir / "evaluation.json",
        settings.model_dir.name,
        settings.device,
    )
    yield


app = FastAPI(title="aivhuman API", lifespan=lifespan)
app.state.gate = ScoreGate(
    settings.score_workers, settings.score_queue, settings.score_wait_seconds
)

# added before CORS so CORS wraps it and 413s still carry CORS headers
app.add_middleware(BodySizeLimit, max_bytes=settings.max_body_bytes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(score.router, prefix="/api")
