from fastapi.testclient import TestClient

from aivhuman_api.main import app
from tests.conftest import TEXT, logged_in


def test_user_cannot_read_another_users_check() -> None:
    alice, bob = logged_in("alice"), logged_in("bob")
    check_id = alice.post("/api/score", json={"text": TEXT}).json()["id"]

    assert alice.get(f"/api/checks/{check_id}").status_code == 200
    # same response as an id that doesn't exist, so bob learns nothing
    assert bob.get(f"/api/checks/{check_id}").status_code == 404
    assert bob.get(f"/api/checks/{check_id}").json() == bob.get("/api/checks/999").json()


def test_history_only_lists_own_checks() -> None:
    alice, bob = logged_in("alice"), logged_in("bob")
    alice.post("/api/score", json={"text": TEXT})
    bob.post("/api/score", json={"text": TEXT})
    assert len(alice.get("/api/checks").json()) == 1
    assert len(bob.get("/api/checks").json()) == 1


def test_history_needs_login() -> None:
    client = TestClient(app)
    assert client.get("/api/checks").status_code == 401
    assert client.get("/api/checks/1").status_code == 401


def test_non_integer_id_is_422() -> None:
    assert logged_in("alice").get("/api/checks/1 OR 1=1").status_code == 422
