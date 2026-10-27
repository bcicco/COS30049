from fastapi.testclient import TestClient

from aivhuman_api.main import app
from aivhuman_api.routes.score import get_scorer

# return the 500 response instead of re-raising into the test
client = TestClient(app, raise_server_exceptions=False)


class BrokenScorer:
    def score(self, _raw: str) -> None:
        raise RuntimeError("secret internal detail")


def test_validation_error_is_one_string() -> None:
    res = client.post("/api/score", json={"text": "a", "extra": 1})
    assert res.status_code == 422
    assert res.json() == {"detail": "extra: Extra inputs are not permitted"}


def test_unexpected_error_is_json_without_internals() -> None:
    app.dependency_overrides[get_scorer] = BrokenScorer
    res = client.post("/api/score", json={"text": "hello"})
    assert res.status_code == 500
    assert res.json() == {"detail": "Something went wrong on the server"}
    assert "secret" not in res.text


def test_unknown_route_is_json() -> None:
    res = client.get("/api/nope")
    assert res.status_code == 404
    assert res.json() == {"detail": "Not Found"}
