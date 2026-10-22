from fastapi.testclient import TestClient

from aivhuman_api.main import app
from tests.conftest import PASSWORD, logged_in


def creds(username: str, password: str = PASSWORD) -> dict[str, str]:
    return {"username": username, "password": password}


def test_register_logs_in() -> None:
    assert logged_in("alice").get("/api/auth/me").json() == {"username": "alice"}


def test_duplicate_username_is_409() -> None:
    logged_in("alice")
    r = TestClient(app).post("/api/auth/register", json=creds("Alice"))
    assert r.status_code == 409


def test_login() -> None:
    logged_in("alice")
    client = TestClient(app)
    assert client.post("/api/auth/login", json=creds("alice")).status_code == 200
    assert client.get("/api/auth/me").json() == {"username": "alice"}


def test_wrong_password_and_unknown_user_look_the_same() -> None:
    logged_in("alice")
    client = TestClient(app)
    wrong = client.post("/api/auth/login", json=creds("alice", "wrong password"))
    unknown = client.post("/api/auth/login", json=creds("nobody"))
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_logout_ends_session() -> None:
    client = logged_in("alice")
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401


def test_session_cookie_is_httponly() -> None:
    r = TestClient(app).post("/api/auth/register", json=creds("alice"))
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie


def test_bad_username_is_422() -> None:
    r = TestClient(app).post("/api/auth/register", json=creds("a' OR '1'='1"))
    assert r.status_code == 422
