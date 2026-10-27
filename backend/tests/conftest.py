import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from aivhuman.mil.infer import ScoreResult, SentenceScore

from aivhuman_api.db import Database
from aivhuman_api.gate import ScoreGate
from aivhuman_api.main import app
from aivhuman_api.routes.score import get_db, get_scorer

TEXT = "Hi 👋 there. Second one."


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
    """Fake scorer, empty database and default gate for every test."""
    db = Database(tmp_path / "checks.db")
    app.dependency_overrides[get_scorer] = FakeScorer
    app.dependency_overrides[get_db] = lambda: db
    app.state.gate = ScoreGate(workers=1, queue=4, wait_seconds=30)
    yield
    app.dependency_overrides.clear()
