import json
from collections.abc import Iterator

from fastapi.testclient import TestClient

from aivhuman_api.config import settings
from aivhuman_api.main import app
from aivhuman_api.schemas import MAX_CHARS
from tests.conftest import logged_in

OVER = settings.max_body_bytes + 1
JSON = {"Content-Type": "application/json"}


def test_declared_length_over_limit_is_413() -> None:
    body = json.dumps({"text": "a" * OVER})
    assert TestClient(app).post("/api/score", content=body, headers=JSON).status_code == 413


def test_streamed_body_over_limit_is_413() -> None:
    # a generator body is sent chunked, with no Content-Length to check up front
    def chunks() -> Iterator[bytes]:
        yield b'{"text": "'
        for _ in range(OVER // 65_536 + 1):
            yield b"a" * 65_536
        yield b'"}'

    client = logged_in("alice")
    assert client.post("/api/score", content=chunks(), headers=JSON).status_code == 413


def test_longest_valid_text_fits() -> None:
    # worst case: every char an astral emoji, JSON escaped to a 12 byte surrogate pair
    body = json.dumps({"text": "🍅" * MAX_CHARS})
    assert len(body) < settings.max_body_bytes
    assert logged_in("alice").post("/api/score", content=body, headers=JSON).status_code == 200
