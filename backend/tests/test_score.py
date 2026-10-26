from fastapi.testclient import TestClient

from aivhuman_api.main import app
from tests.conftest import TEXT

client = TestClient(app)


def test_score_maps_result() -> None:
    result = client.post("/api/score", json={"text": TEXT}).json()
    # sentence text is sliced server side, so the emoji doesn't shift it
    assert [s["text"] for s in result["sentences"]] == ["Hi 👋 there.", "Second one."]
    assert "caveats" not in result
    assert "contributions" not in result["sentences"][0]
    assert {"model", "document_flagged"}.isdisjoint(result)
    assert "n_tokens" not in result["sentences"][0]


def test_whitespace_is_422() -> None:
    assert client.post("/api/score", json={"text": "   "}).status_code == 422


def test_unknown_field_is_422() -> None:
    assert client.post("/api/score", json={"text": "a", "extra": 1}).status_code == 422
