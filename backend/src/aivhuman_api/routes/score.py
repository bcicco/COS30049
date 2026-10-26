"""Score a text sentence by sentence."""

from typing import Annotated

from aivhuman.mil.infer import Scorer, ScoreResult
from fastapi import APIRouter, Depends, HTTPException, Request

from aivhuman_api.config import settings
from aivhuman_api.gate import ScoreGate
from aivhuman_api.schemas import ScoreRequest, ScoreResponse, SentenceScore

router = APIRouter()


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
    scorer: Annotated[Scorer, Depends(get_scorer)],
    gate: Annotated[ScoreGate, Depends(get_gate)],
) -> ScoreResponse:
    # async so the event loop keeps answering other requests, the model itself runs on
    # a worker thread behind the gate
    try:
        result = await gate.run(scorer.score, body.text)
    except ValueError as e:  # e.g. whitespace only, nothing to segment
        raise HTTPException(status_code=422, detail=str(e)) from e
    return to_response(result, settings.model_dir.name)
