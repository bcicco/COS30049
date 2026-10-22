import pytest
from pydantic import ValidationError

from aivhuman_api.schemas import MAX_CHARS, ScoreRequest


def test_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest(text="")


def test_rejects_long_text() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest(text="a" * (MAX_CHARS + 1))
