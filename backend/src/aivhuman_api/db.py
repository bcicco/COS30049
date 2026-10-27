"""History of past checks in SQLite.

Every query passes user input as a ? parameter, never pasted into the SQL text, so input
like ' OR '1'='1 is stored and matched as plain data (SQL injection defence).
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel

SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    document_score REAL NOT NULL,
    n_sentences INTEGER NOT NULL,
    n_flagged INTEGER NOT NULL
)
"""


class Check(BaseModel):
    """one stored check"""

    id: int
    created_at: datetime
    text: str
    document_score: float
    n_sentences: int
    n_flagged: int


def escape_like(s: str) -> str:
    """so % and _ in a search match themselves, not any characters"""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Database:
    """a connection per call, so worker threads never share one"""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """commit on success, roll back on error, always close"""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def add(
        self, session: UUID, text: str, document_score: float, n_sentences: int, n_flagged: int
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO checks"
                " (session_id, created_at, text, document_score, n_sentences, n_flagged)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(session),
                    datetime.now(UTC).isoformat(),
                    text,
                    document_score,
                    n_sentences,
                    n_flagged,
                ),
            )

    def recent(self, session: UUID, search: str | None, limit: int) -> list[Check]:
        """newest first, only this session's checks"""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, text, document_score, n_sentences, n_flagged"
                " FROM checks WHERE session_id = ? AND text LIKE ? ESCAPE '\\'"
                " ORDER BY id DESC LIMIT ?",
                (str(session), f"%{escape_like(search or '')}%", limit),
            ).fetchall()
        return [Check.model_validate(dict(r)) for r in rows]
