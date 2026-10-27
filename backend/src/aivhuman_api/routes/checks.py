"""History of the checks made in one browser session."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query

from aivhuman_api.db import Check, Database
from aivhuman_api.routes.score import get_db

router = APIRouter()


@router.get("/checks")
def checks(
    db: Annotated[Database, Depends(get_db)],
    x_session_id: Annotated[UUID, Header()],
    q: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[Check]:
    """Newest first. `q` keeps only checks whose text contains it."""
    # plain def, so fastapi runs it on a worker thread and the query never blocks the loop
    return db.recent(x_session_id, q, limit)
