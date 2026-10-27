"""Score a text sentence by sentence."""

from typing import Annotated
from uuid import UUID

import anyio
from aivhuman.mil.infer import Scorer, ScoreResult
from fastapi import APIRouter, Depends, Header, HTTPException, Request

from aivhuman_api.db import Database
from aivhuman_api.gate import ScoreGate
from aivhuman_api.schemas import ScoreRequest, ScoreResponse, SentenceScore

router = APIRouter()


def get_scorer(request: Request) -> Scorer:
    scorer: Scorer = request.app.state.scorer
    return scorer


def get_gate(request: Request) -> ScoreGate:
    gate: ScoreGate = request.app.state.gate
    return gate


def get_db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


def to_response(result: ScoreResult) -> ScoreResponse:
    """ml result -> api schema, adds sentence text so the frontend never slices"""
    sentences = [
        SentenceScore(
            text=result.text[s.start : s.end],
            start=s.start,
            end=s.end,
            score=s.score,
            flagged=s.flagged,
            too_short=s.too_short,
        )
        for s in result.sentences
    ]
    return ScoreResponse(
        text=result.text,
        sentences=sentences,
        document_score=result.document_score,
    )


@router.post("/score")
async def score(
    body: ScoreRequest,
    scorer: Annotated[Scorer, Depends(get_scorer)],
    gate: Annotated[ScoreGate, Depends(get_gate)],
    db: Annotated[Database, Depends(get_db)],
    x_session_id: Annotated[UUID | None, Header()] = None,
) -> ScoreResponse:
    # async so the event loop keeps answering other requests, the model itself runs on
    # a worker thread behind the gate
    try:
        result = await gate.run(scorer.score, body.text)
    except ValueError as e:  # e.g. whitespace only, nothing to segment
        raise HTTPException(status_code=422, detail=str(e)) from e
    # no session id (e.g. plain curl) means nothing to file it under, so it isn't saved
    if x_session_id is not None:
        flagged = sum(s.flagged for s in result.sentences)
        await anyio.to_thread.run_sync(
            db.add, x_session_id, result.text, result.document_score, len(result.sentences), flagged
        )
    return to_response(result)
