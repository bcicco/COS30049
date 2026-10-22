"""SQLite tables for users, login sessions and saved checks."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import ForeignKey, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from aivhuman_api.config import settings


def utcnow() -> datetime:
    # naive UTC, because SQLite hands datetimes back without a timezone
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class LoginSession(Base):
    __tablename__ = "sessions"

    # sha256 of the cookie token, so a leaked database can't be replayed as logins
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime]


class Check(Base):
    __tablename__ = "checks"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    document_score: Mapped[float]
    preview: Mapped[str] = mapped_column(String(100))
    result_json: Mapped[str] = mapped_column(Text)  # ScoreResponse as JSON


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    # fastapi runs sync dependencies and endpoints on worker threads
    return create_engine(url, connect_args={"check_same_thread": False})


engine = make_engine(settings.database_url)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    with SessionLocal() as db:
        yield db
