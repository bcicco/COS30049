"""Passwords, login sessions and the current-user dependency."""

import hashlib
import secrets
from datetime import timedelta
from typing import Annotated, Final

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aivhuman_api.config import settings
from aivhuman_api.db import LoginSession, User, get_db, utcnow
from aivhuman_api.schemas import Credentials, Me

COOKIE: Final = "session"
_hasher = PasswordHasher()
# checked against when the username doesn't exist, so a failed login costs one argon2
# verify either way and its timing can't reveal which usernames are real
_DUMMY_HASH: Final = _hasher.hash(secrets.token_urlsafe(16))

DB = Annotated[Session, Depends(get_db)]
router = APIRouter(prefix="/auth")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _password_ok(stored_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(stored_hash, password)
    except VerificationError:
        return False


def _start_session(db: Session, user: User, response: Response) -> None:
    token = secrets.token_urlsafe(32)
    expires = utcnow() + timedelta(days=settings.session_days)
    db.add(LoginSession(token_hash=_token_hash(token), user_id=user.id, expires_at=expires))
    db.commit()
    response.set_cookie(
        COOKIE,
        token,
        max_age=settings.session_days * 86_400,
        httponly=True,  # page scripts can't read it, so an XSS bug can't steal it
        samesite="lax",  # not sent on cross-site POSTs
        secure=settings.cookie_secure,
        path="/api",
    )


def current_user(request: Request, db: DB) -> User:
    """The logged-in user, or 401."""
    token = request.cookies.get(COOKIE)
    if token:
        row = db.get(LoginSession, _token_hash(token))
        if row is not None and row.expires_at > utcnow():
            user = db.get(User, row.user_id)
            if user is not None:
                return user
    raise HTTPException(status_code=401, detail="Not logged in")


CurrentUser = Annotated[User, Depends(current_user)]


@router.post("/register", status_code=201)
def register(body: Credentials, response: Response, db: DB) -> Me:
    user = User(username=body.username.lower(), password_hash=_hasher.hash(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Username taken") from None
    _start_session(db, user, response)
    return Me(username=user.username)


@router.post("/login")
def login(body: Credentials, response: Response, db: DB) -> Me:
    user = db.scalar(select(User).where(User.username == body.username.lower()))
    ok = _password_ok(user.password_hash if user else _DUMMY_HASH, body.password)
    if user is None or not ok:
        raise HTTPException(status_code=401, detail="Wrong username or password")
    _start_session(db, user, response)
    return Me(username=user.username)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, db: DB) -> None:
    token = request.cookies.get(COOKIE)
    if token:
        row = db.get(LoginSession, _token_hash(token))
        if row is not None:
            db.delete(row)
            db.commit()
    response.delete_cookie(COOKIE, path="/api")


@router.get("/me")
def me(user: CurrentUser) -> Me:
    return Me(username=user.username)
