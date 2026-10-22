import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from aivhuman.mil.infer import ScoreResult, SentenceScore
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from aivhuman_api.db import Base, get_db, make_engine
from aivhuman_api.gate import ScoreGate
from aivhuman_api.main import app
from aivhuman_api.routes.score import get_scorer

TEXT = "Hi 👋 there. Second one."
PASSWORD = "correct horse battery"


class FakeScorer:
    """Two fixed sentences, no model. delay simulates a slow request."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def score(self, raw: str) -> ScoreResult:
        time.sleep(self.delay)
        if not raw.strip():
            raise ValueError("no text to score")
        return ScoreResult(
            text=raw,
            sentences=[
                SentenceScore(
                    start=0,
                    end=11,
                    n_tokens=5,
                    score=0.9,
                    flagged=False,
                    too_short=True,
                    contributions=[("lm_logprob", 0.4)],
                ),
                SentenceScore(
                    start=12,
                    end=23,
                    n_tokens=4,
                    score=0.3,
                    flagged=False,
                    too_short=True,
                    contributions=[],
                ),
            ],
            document_score=0.6,
            document_flagged=True,
            caveats=["c"],
        )


@pytest.fixture(autouse=True)
def fresh_app(tmp_path: Path) -> Iterator[None]:
    """Own SQLite file, fake scorer and default gate for every test."""
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)

    def test_db() -> Iterator[Session]:
        with sessions() as db:
            yield db

    app.dependency_overrides[get_db] = test_db
    app.dependency_overrides[get_scorer] = FakeScorer
    app.state.gate = ScoreGate(workers=1, queue=4, wait_seconds=30)
    yield
    app.dependency_overrides.clear()
    engine.dispose()


def logged_in(username: str) -> TestClient:
    """A client holding a fresh user's session cookie."""
    client = TestClient(app)
    r = client.post("/api/auth/register", json={"username": username, "password": PASSWORD})
    assert r.status_code == 201
    return client
