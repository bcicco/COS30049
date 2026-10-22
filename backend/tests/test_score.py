from fastapi.testclient import TestClient

from aivhuman_api.main import app
from tests.conftest import TEXT, logged_in


def test_score_maps_result_and_saves_it() -> None:
    client = logged_in("alice")
    body = client.post("/api/score", json={"text": TEXT}).json()
    result = body["result"]
    assert result["model"] == "p5-adv"
    # sentence text is sliced server side, so the emoji doesn't shift it
    assert [s["text"] for s in result["sentences"]] == ["Hi 👋 there.", "Second one."]
    assert "caveats" not in result
    assert "contributions" not in result["sentences"][0]
    assert client.get(f"/api/checks/{body['id']}").json()["result"] == result


def test_score_needs_login() -> None:
    assert TestClient(app).post("/api/score", json={"text": TEXT}).status_code == 401


def test_whitespace_is_422() -> None:
    assert logged_in("alice").post("/api/score", json={"text": "   "}).status_code == 422


def test_unknown_field_is_422() -> None:
    r = logged_in("alice").post("/api/score", json={"text": "a", "extra": 1})
    assert r.status_code == 422
