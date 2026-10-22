"""A user's saved checks. Every query is filtered by the logged-in user's id."""

from datetime import UTC

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from aivhuman_api.auth import DB, CurrentUser
from aivhuman_api.db import Check
from aivhuman_api.schemas import CheckOut, CheckSummary, ScoreResponse

router = APIRouter(prefix="/checks")
HISTORY_LIMIT = 50


def to_out(check: Check) -> CheckOut:
    return CheckOut(
        id=check.id,
        created_at=check.created_at.replace(tzinfo=UTC),
        result=ScoreResponse.model_validate_json(check.result_json),
    )


@router.get("")
def list_checks(user: CurrentUser, db: DB) -> list[CheckSummary]:
    rows = db.scalars(
        select(Check)
        .where(Check.user_id == user.id)
        .order_by(Check.created_at.desc())
        .limit(HISTORY_LIMIT)
    )
    return [
        CheckSummary(
            id=c.id,
            created_at=c.created_at.replace(tzinfo=UTC),
            document_score=c.document_score,
            preview=c.preview,
        )
        for c in rows
    ]


@router.get("/{check_id}")
def get_check(check_id: int, user: CurrentUser, db: DB) -> CheckOut:
    # owner is part of the lookup, so another user's id is indistinguishable from a
    # missing one: 404 either way, never 403, which would confirm the id exists
    check = db.scalar(select(Check).where(Check.id == check_id, Check.user_id == user.id))
    if check is None:
        raise HTTPException(status_code=404, detail="Check not found")
    return to_out(check)
