"""Score a text sentence by sentence and save it to the user's history."""

from typing import Annotated

import anyio
from aivhuman.mil.infer import Scorer, ScoreResult
from fastapi import APIRouter, Depends, HTTPException, Request

from aivhuman_api.auth import DB, CurrentUser
from aivhuman_api.config import settings
from aivhuman_api.db import Check
from aivhuman_api.gate import ScoreGate
from aivhuman_api.routes.checks import to_out
from aivhuman_api.schemas import CheckOut, ScoreRequest, ScoreResponse, SentenceScore

router = APIRouter()
PREVIEW_CHARS = 100


def get_scorer(request: Request) -> Scorer:
    scorer: Scorer = request.app.state.scorer
    return scorer


def get_gate(request: Request) -> ScoreGate:
    gate: ScoreGate = request.app.state.gate
    return gate


def to_response(result: ScoreResult, model: str) -> ScoreResponse:
    """ml result -> api schema, adds sentence text so the frontend never slices"""
    sentences = [
        SentenceScore(
            text=result.text[s.start : s.end],
            start=s.start,
            end=s.end,
            n_tokens=s.n_tokens,
            score=s.score,
            flagged=s.flagged,
            too_short=s.too_short,
        )
        for s in result.sentences
    ]
    return ScoreResponse(
        model=model,
        text=result.text,
        sentences=sentences,
        document_score=result.document_score,
        document_flagged=result.document_flagged,
    )


@router.post("/score")
async def score(
    body: ScoreRequest,
    user: CurrentUser,
    db: DB,
    scorer: Annotated[Scorer, Depends(get_scorer)],
    gate: Annotated[ScoreGate, Depends(get_gate)],
) -> CheckOut:
    # async so the event loop keeps answering other requests, the model itself runs on
    # a worker thread behind the gate
    try:
        result = await gate.run(scorer.score, body.text)
    except ValueError as e:  # e.g. whitespace only, nothing to segment
        raise HTTPException(status_code=422, detail=str(e)) from e
    response = to_response(result, settings.model_dir.name)

    def save() -> Check:
        check = Check(
            user_id=user.id,
            document_score=response.document_score,
            preview=response.text[:PREVIEW_CHARS],
            result_json=response.model_dump_json(),
        )
        db.add(check)
        db.commit()
        return check

    return to_out(await anyio.to_thread.run_sync(save))
