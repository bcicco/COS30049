import uuid

from fastapi.testclient import TestClient

from aivhuman_api.main import app

client = TestClient(app)

# classic injections, each must come back as plain stored text
INJECTIONS = [
    "' OR '1'='1",
    "'); DROP TABLE checks; --",
    "x' UNION SELECT 1, 2, 3, 4, 5, 6 --",
]


def headers(session: uuid.UUID) -> dict[str, str]:
    return {"X-Session-Id": str(session)}


def score(text: str, session: uuid.UUID) -> None:
    assert client.post("/api/score", json={"text": text}, headers=headers(session)).is_success


def history(session: uuid.UUID, **params: str) -> list[dict]:
    res = client.get("/api/checks", params=params, headers=headers(session))
    assert res.status_code == 200
    return res.json()


def test_checks_are_saved_newest_first() -> None:
    me = uuid.uuid4()
    score("First one. Here.", me)
    score("Second one. Here.", me)
    rows = history(me)
    assert [r["text"] for r in rows] == ["Second one. Here.", "First one. Here."]
    assert rows[0]["n_sentences"] == 2
    assert rows[0]["document_score"] == 0.6


def test_injections_are_stored_as_plain_text() -> None:
    me = uuid.uuid4()
    for text in INJECTIONS:
        score(text, me)
    # table survived the DROP and every input came back unchanged
    assert [r["text"] for r in history(me)] == INJECTIONS[::-1]


def test_injection_in_search_matches_nothing() -> None:
    me, other = uuid.uuid4(), uuid.uuid4()
    score("my text. Is here.", me)
    score("their text. Is here.", other)
    for q in INJECTIONS:
        assert history(me, q=q) == []
    # a real search still works, and never crosses into another session
    assert [r["text"] for r in history(me, q="text")] == ["my text. Is here."]


def test_like_wildcards_match_literally() -> None:
    me = uuid.uuid4()
    score("plain text. Is here.", me)
    assert history(me, q="%") == []
    assert history(me, q="_") == []


def test_sessions_only_see_their_own_checks() -> None:
    me, other = uuid.uuid4(), uuid.uuid4()
    score("mine. Is here.", me)
    assert history(other) == []


def test_no_session_id_is_not_saved() -> None:
    assert client.post("/api/score", json={"text": "anon. Is here."}).is_success
    assert client.get("/api/checks").status_code == 422


def test_bad_session_id_is_422() -> None:
    res = client.get("/api/checks", headers={"X-Session-Id": "' OR '1'='1"})
    assert res.status_code == 422
    assert res.json()["detail"].startswith("x-session-id:")
