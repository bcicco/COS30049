"""Request and response bodies for the scoring API."""

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

# Mirrors aivhuman.mil.infer.MAX_CHARS.
MAX_CHARS: Final = 100_000


class ScoreRequest(BaseModel):
    """Text to score."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_CHARS)


class SentenceScore(BaseModel):
    """One scored sentence."""

    # start/end are code point offsets into the NFC text. JS strings index by UTF-16,
    # so the frontend should render `text` rather than slicing by offset.
    text: str
    start: int
    end: int
    n_tokens: int
    score: float = Field(ge=0, le=1)  # calibrated p(machine)
    flagged: bool
    too_short: bool  # under the shortest length bucket, never flagged


class ScoreResponse(BaseModel):
    """Everything the frontend needs to render one result."""

    model: str  # run name, e.g. "p5-adv"
    text: str  # NFC-normalised input
    sentences: list[SentenceScore]
    document_score: float = Field(ge=0, le=1)
    document_flagged: bool
