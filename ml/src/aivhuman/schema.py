"""The one normalised schema every corpus converts into."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Final, Self

import orjson
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    model_serializer,
    model_validator,
)

__all__ = [
    "LABEL_HUMAN",
    "LABEL_MACHINE",
    "LABEL_NAMES",
    "SOURCES",
    "SPLIT_ROLES",
    "TRAINABLE_ROLES",
    "Doc",
    "SchemaError",
    "SentenceSpan",
    "doc_from_json",
    "doc_to_json",
    "iter_jsonl",
    "label_name",
    "validate_doc",
]

LABEL_HUMAN: Final = 0
LABEL_MACHINE: Final = 1
LABEL_NAMES: Final = {LABEL_HUMAN: "human", LABEL_MACHINE: "machine"}

SOURCES: Final = frozenset({"raid", "mage", "seqxgpt"})


SPLIT_ROLES: Final = frozenset(
    {
        "train_pool",  # RAID train.csv, attack == "none"
        "xcorpus_test",  # MAGE train/valid/test
        "xcorpus_ood_test",  # MAGE test_ood_set_gpt
        "xcorpus_para_test",  # MAGE test_ood_set_gpt_para
        "calib_pool",  # SeqXGPT-Bench
        "calib_ood_pool",  # SeqXGPT OOD sentence-level
    }
)

TRAINABLE_ROLES: Final = frozenset({"train_pool"})

Label = Annotated[int, Field(ge=0, le=1)]
Offset = Annotated[int, Field(ge=0)]


class SchemaError(ValueError):
    """A document violates an invariant the rest of the pipeline relies on"""


def label_name(label: int) -> str:
    return LABEL_NAMES[label]


# ****** NOTE ******
# bag = DOC, items = SENTENCES
class SentenceSpan(BaseModel):
    """One sentence, as character offsets into the parent attribute (Doc.text)"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: Offset

    end: Offset
    """**Exclusive** character offset. ``text[start:end]`` is the sentence."""

    n_tokens: Annotated[int, Field(ge=0)]
    """Not confirmed, likely to be ModernBERT-base tokens"""

    n_words: Annotated[int, Field(ge=0)]
    """Whitespace-delimited runs. A cheap tokenizer-independent sanity check."""

    label: Label | None = None
    """Sentence provenance where it is *known* (likely to be SeqXGPT only for sentence calib.) """
    # ***** IMPORTANT *******
    # The model is multiple-instance precisely because sentence labels are unavailable at training time
    # Dont leak them in here

    machine_char_frac: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """Fraction of this span's characters on the machine side of the boundary."""

    straddles_boundary: bool = False
    """True when this span contains both human and machine characters. """

    # ***** IMPORTANT *******
    # For SeqGXPT, the boundary will be a sentence boundary, so straddle means:
    # segmenter disagrees with their segmenter. Will need to be excluded from evaluation metrics.

    @model_validator(mode="after")
    def _check_width(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"span ({self.start}, {self.end}) is empty or inverted")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialise, omitting the optional keys at their defaults."""
        # *** NOTE ***
        # Deterministic, and it keeps RAID's ~500k x ~20 spans to four keys apiece
        # instead of seven.
        out: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "n_tokens": self.n_tokens,
            "n_words": self.n_words,
        }
        if self.label is not None:
            out["label"] = self.label
        if self.machine_char_frac is not None:
            out["machine_char_frac"] = self.machine_char_frac
        if self.straddles_boundary:
            out["straddles_boundary"] = True
        return out

    @model_serializer
    def _serialise(self) -> dict[str, Any]:
        return self.to_dict()


class Doc(BaseModel):
    """A document with its sentence spans, normalised across all three corpora."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(min_length=1)
    """ {source}:{local_id} ---- Globally unique across all three corpora."""

    text: str
    """NFC-normalised document text. Verbatim in every other respect."""

    label: Label
    """Canonical polarity: [see final: datatypes] data:`LABEL_HUMAN` or :data:`LABEL_MACHINE`."""

    source: str
    """One of :data:`SOURCES`."""

    domain: str | None
    """Genre/corpus of origin. None for SeqXGPT (records no domain)."""

    generator: str | None
    """Model that produced the text. None for human documents."""

    group_id: str = Field(min_length=1)
    """Leak-free grouping unit. No group_id may span two Phase 2 splits."""

    split_role: str
    """One of :data:`SPLIT_ROLES`."""

    sentences: list[SentenceSpan]
    """Ordered, non-overlapping spans covering every non-whitespace character."""

    label_raw: str = Field(min_length=1)
    """The *original* fine-grained label, verbatim."""

    # NOTE RAID's model, MAGE's src, SeqXGPT's generator name

    meta: dict[str, Any] = Field(default_factory=dict)
    """Per-source provenance, can be dropped"""

    @field_serializer("sentences")
    def _serialise_sentences(self, spans: list[SentenceSpan]) -> list[dict[str, Any]]:
        return [s.to_dict() for s in spans]

    @model_validator(mode="after")
    def _check_enums_and_ids(self) -> Self:
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}")
        if self.split_role not in SPLIT_ROLES:
            raise ValueError(f"unknown split_role {self.split_role!r}")
        if not self.doc_id.startswith(f"{self.source}:"):
            raise ValueError(f"doc_id {self.doc_id!r} lacks the {self.source!r} prefix")
        if not self.group_id.startswith(f"{self.source}:"):
            raise ValueError(f"group_id {self.group_id!r} lacks the source prefix")
        if (
            self.label == LABEL_HUMAN
            and self.generator is not None
            and self.source != "mage"
        ):
            # MAGE's paraphrase testbed labels paraphrased human text as machine,
            # so it is the only case where a human doc has a generator name.
            raise ValueError(f"human doc names generator {self.generator!r}")
        return self

    @model_validator(mode="after")
    def _check_text_is_nfc(self) -> Self:
        if not unicodedata.is_normalized("NFC", self.text):
            raise ValueError("text is not NFC-normalised")
        return self

    @model_validator(mode="after")
    def _check_span_coverage(self) -> Self:
        """Spans must partition the document's non-whitespace content."""
        # ****** NOTE ****** performs:
        # Ordering
        # Disjointness
        # Full coverage of non-whitespace characters
        n = len(self.text)
        prev_end = 0
        for i, s in enumerate(self.sentences):
            if s.end > n:
                raise ValueError(f"span {i} ends at {s.end}, past text length {n}")
            if s.start < prev_end:
                raise ValueError(
                    f"span {i} starts {s.start} before previous end {prev_end}"
                )
            gap = self.text[prev_end : s.start]
            if gap.strip():
                raise ValueError(f"non-whitespace gap before span {i}: {gap!r}")
            prev_end = s.end

        tail = self.text[prev_end:]
        if tail.strip():
            raise ValueError(f"non-whitespace tail after last span: {tail!r}")
        return self

    @model_validator(mode="after")
    def _check_sentence_labels(self) -> Self:
        """SeqXGPT's labels must be complete; no other source may carry any."""

        # ***************** NOTE ********************
        # Only SeqXGPT has sentence-level labels
        # All for calibration
        # Cannot have sentence label in training corpus

        labelled = [s.label is not None for s in self.sentences]
        if self.source == "seqxgpt":
            if not all(labelled):
                raise ValueError("seqxgpt span without a label")
        elif any(labelled):
            raise ValueError(f"{self.source} must not carry sentence labels")
        return self


def doc_to_json(doc: Doc) -> bytes:
    """Serialise to a single JSONL line (no trailing newline)."""
    return orjson.dumps(doc.model_dump())


def doc_from_json(line: bytes | str) -> Doc:
    """Parse a JSONL line back into a fully validated :class:`Doc`."""
    try:
        return Doc.model_validate(orjson.loads(line))
    except ValidationError as exc:
        raise SchemaError(str(exc)) from exc
    except orjson.JSONDecodeError as exc:
        raise SchemaError(f"malformed JSON: {exc}") from exc


def validate_doc(doc: Doc) -> None:
    """Re-validate an existing document, used in testing"""
    try:
        Doc.model_validate(doc.model_dump())
    except ValidationError as exc:
        raise SchemaError(f"{doc.doc_id}: {exc}") from exc


def iter_jsonl(path: str | Path) -> Iterator[Doc]:
    """Yield validated :class:`Doc` objects from a JSONL file."""
    with Path(path).open("rb") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield doc_from_json(line)
            except SchemaError as exc:
                raise SchemaError(f"{path}:{lineno}: {exc}") from exc
