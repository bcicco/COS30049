"""Sentence segmentation into character offsets."""

# ***************** IMPORTANT ******************
# clean = False is MANDATORY (prevent rewrite)
# char_span = True is MANDATORY (return offsets, not strings)


from __future__ import annotations

import re
import warnings
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PrivateAttr,
    field_validator,
)

# get weird warnings thrown with this module, ignore them
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd
    from pysbd.languages import LANGUAGE_CODES

__all__ = ["SegmentStats", "Segmenter", "n_words"]

_WORD_RE = re.compile(r"\S+")


def n_words(s: str) -> int:
    """Whitespace-delimited token count. Tokenizer-independent sanity check."""
    return len(_WORD_RE.findall(s))


class SegmentStats(BaseModel):
    """Segmentation health, published in the Phase 1 report."""

    model_config = ConfigDict(validate_assignment=False, extra="forbid")

    docs: NonNegativeInt = 0
    spans: NonNegativeInt = 0
    repaired: NonNegativeInt = 0
    failed: NonNegativeInt = 0
    nonws_gap_chars: NonNegativeInt = 0
    single_span_docs: NonNegativeInt = 0
    empty_docs: NonNegativeInt = 0
    trimmed_spans: NonNegativeInt = 0
    dropped_empty_spans: NonNegativeInt = 0
    recovered_gap_spans: NonNegativeInt = 0
    unsegmentable_gaps: NonNegativeInt = 0

    @property
    def spans_per_doc(self) -> float:
        return round(self.spans / self.docs, 3) if self.docs else 0.0

    @property
    def anchor_failure_rate(self) -> float:
        """Share of spans whose PySBD offsets could not be verified."""

        # quality, not correctness.
        return round(self.failed / self.spans, 6) if self.spans else 0.0

    @property
    def is_healthy(self) -> bool:
        """Whether the run met its correctness bar."""
        # health check
        return self.nonws_gap_chars == 0

    def as_dict(self) -> dict[str, Any]:
        """Report-facing dict of the raw counters for postmortem and sanity checks."""
        return self.model_dump()


class Segmenter(BaseModel):
    """Wraps PySBD and asserts what PySBD does not.

    Not thread-safe and expensive to construct, so build one per process (see
    the ``initializer=`` argument used by the ingest worker pool).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    language: str = "en"
    stats: SegmentStats = Field(default_factory=SegmentStats)

    # private bc nothing. to val, its machinery
    _seg: Any = PrivateAttr(default=None)

    @field_validator("language")
    @classmethod
    def _known_language(cls, value: str) -> str:
        """Reject an unsupported language up front."""
        if value not in LANGUAGE_CODES:
            raise ValueError(
                f"pysbd has no segmenter for {value!r}; "
                f"supported: {', '.join(sorted(LANGUAGE_CODES))}"
            )
        return value

    def model_post_init(self, _context: Any, /) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            self._seg = pysbd.Segmenter(
                language=self.language, clean=False, char_span=True
            )

    def segment(self, text: str) -> list[tuple[int, int]]:
        """Return ordered, non-overlapping ``(start, end)`` pairs."""
        self.stats.docs += 1
        if not text.strip():
            self.stats.empty_docs += 1
            return []

        spans = self._enforce_monotonic(self._collect(text, 0, len(text)))
        spans = self._fill_gaps(text, spans)
        # Coverage is a post-condition, not an emergent property of the passes
        # IMPORTANT: NOT SURE HOW WELL _FILL_GAPS WORKS

        # above. _fill_gaps recovers what it can *as sentences*; this guarantees
        # nothing is left behind regardless of how PySBD behaved.
        spans = self._close_gaps(text, spans)
        self._account_gaps(text, spans)

        self.stats.spans += len(spans)
        if len(spans) == 1:
            self.stats.single_span_docs += 1
        return spans

    def _collect(self, text: str, offset: int, end_limit: int) -> list[tuple[int, int]]:
        """Run PySBD over ``text[offset:end_limit]`` and return absolute spans."""
        chunk = text[offset:end_limit]
        raw = self._seg.segment(chunk)  # type: ignore[attr-defined]
        spans: list[tuple[int, int]] = []
        cursor = 0
        for ts in raw:
            start, stop = self._anchor(chunk, ts, cursor)
            if start is None or stop is None:
                continue
            cursor = stop
            trimmed = _trim(chunk, start, stop)
            if trimmed is None:
                self.stats.dropped_empty_spans += 1
                continue
            if trimmed != (start, stop):
                self.stats.trimmed_spans += 1
            spans.append((trimmed[0] + offset, trimmed[1] + offset))
        return spans

    def _fill_gaps(
        self, text: str, spans: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Recover text PySBD omitted entirely. Essay incoming, apologies....

        pysbd 0.3.4 does not always cover its input. On an arXiv abstract
        containing ``@xmath0`` placeholders, unicode symbols (a solar-mass sign
        and a true minus sign) and an inline citation, it returned six spans
        whose offsets jump
        558 -> 857 and 1026 -> 1142, dropping 413 characters. Every span it
        returned round-tripped perfectly, so neither the offset check nor the
        anchor-repair path sees this.....only coverage accounting does.

        Dropped text is not cosmetic. Those characters would be excluded from
        every sentence score, so the interface would render a region of the
        document as unscored with no indication why, and the document score
        would be computed over less text than the user submitted.

        Recovery re-runs PySBD on the gap in isolation, which usually succeeds
        because the context that confused it is gone. If it still comes back
        short, the gap is emitted as a single span: an imperfectly-bounded
        sentence is a far smaller problem than a silently unscored one.
        """
        filled: list[tuple[int, int]] = []
        cursor = 0
        for start, end in [*spans, (len(text), len(text))]:
            if text[cursor:start].strip():
                # Recovered spans are clipped to the gap, so monotonicity holds
                # by construction and no second enforcement pass is needed --
                # one was previously dropping the very spans added here.
                recovered = [
                    (a, b)
                    for a, b in self._collect(text, cursor, start)
                    if cursor <= a < b <= start
                ]
                if recovered:
                    self.stats.recovered_gap_spans += len(recovered)
                    filled.extend(recovered)
            if start < end:
                filled.append((start, end))
            cursor = max(cursor, end)
        return filled

    def _close_gaps(
        self, text: str, spans: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Normalise to a guaranteed partition: ordered, disjoint, fully covering."""

        # The single top man #1 MVP authoritative pass. The only one whose output the rest of
        # the pipeline trusts. KEEP IN MIND --> upstream = best-effort

        closed: list[tuple[int, int]] = []
        cursor = 0
        for start, end in [*sorted(spans), (len(text), len(text))]:
            if text[cursor : max(cursor, start)].strip():
                trimmed = _trim(text, cursor, start)
                if trimmed is not None:
                    self.stats.unsegmentable_gaps += 1
                    closed.append(trimmed)
                    cursor = trimmed[1]
            clipped = _trim(text, max(start, cursor), end)
            if clipped is None:
                continue
            closed.append(clipped)
            cursor = clipped[1]
        return closed

    def _anchor(
        self, text: str, ts: object, cursor: int
    ) -> tuple[int, int] | tuple[None, None]:
        """Verify PySBD's offsets, re-anchoring them if they do not round-trip."""
        sent: str = ts.sent  # type: ignore[attr-defined]
        start: int = ts.start  # type: ignore[attr-defined]
        end: int = ts.end  # type: ignore[attr-defined]

        if text[start:end] == sent:
            return start, end

        # Re-anchor with a cursor that only ever advances, so repeated sentences
        # map to successive occurrences rather than all to the first.
        found = text.find(sent, cursor)
        if found >= 0:
            self.stats.repaired += 1
            return found, found + len(sent)

        self.stats.failed += 1
        return None, None

    def _enforce_monotonic(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Drop any span that overlaps its predecessor. Should never fire."""
        out: list[tuple[int, int]] = []
        prev_end = 0
        for start, end in spans:
            if start < prev_end:
                self.stats.failed += 1
                continue
            out.append((start, end))
            prev_end = end
        return out

    def _account_gaps(self, text: str, spans: list[tuple[int, int]]) -> None:
        prev_end = 0
        for start, _end in spans:
            self.stats.nonws_gap_chars += len(text[prev_end:start].strip())
            prev_end = _end
        self.stats.nonws_gap_chars += len(text[prev_end:].strip())


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink a span past surrounding whitespace. PySBD keeps the trailing space."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None
