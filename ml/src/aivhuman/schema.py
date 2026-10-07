"""Normalised data format for all sources"""

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
    model_serializer,
    model_validator,
)

LABEL_HUMAN: Final = 0
LABEL_MACHINE: Final = 1
LABEL_NAMES: Final = {LABEL_HUMAN: "human", LABEL_MACHINE: "machine"}

SOURCES: Final = frozenset({"raid", "mage", "seqxgpt", "daigt"})

# note ood = out of domain
SPLIT_ROLES: Final = frozenset(
    {
        "train_pool",  # RAID train.csv, attack == "none"
        "xcorpus_test",  # MAGE train/valid/test
        "xcorpus_ood_test",  # MAGE test_ood_set_gpt
        "xcorpus_para_test",  # MAGE test_ood_set_gpt_para
        "calib_pool",  # SeqXGPT-Bench
        "calib_ood_pool",  # SeqXGPT OOD sentence-level
        "xcorpus_essay_test",  # DAIGT v2
    }
)

TRAINABLE_ROLES: Final = frozenset({"train_pool"})

Label = Annotated[int, Field(ge=0, le=1)]

# sentences stored as character offsets into the parent Doc.text, which is NFC-normalised, Offset
# used to denote this
Offset = Annotated[int, Field(ge=0)]


class SchemaError(ValueError):
    """A document violates an invariant the rest of the pipeline relies on"""


def label_name(label: int) -> str:
    return LABEL_NAMES[label]


# ****** NOTE ******
# bag = DOC, items = SENTENCES
class SentenceSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: Offset

    end: Offset  # text[start:end] is the sentence

    n_tokens: Annotated[int, Field(ge=0)]  # not confirmed, prob ModernBERT-base tokens

    n_words: Annotated[int, Field(ge=0)]  # whitespace split, cheap sanity check

    label: Label | None = None  # only where known (SeqXGPT only for sentence calib.)
    # ***** IMPORTANT *******
    # The model is multiple-instance precisely because sentence labels are
    # unavailable at training time we need to make sure not to leak them here

    machine_char_frac: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    # frac of chars on the machine side of the boundary, flags bizzare parsing

    straddles_boundary: bool = False  # has both human + machine chars, see above ^^

    # ***** IMPORTANT *******
    # For SeqGXPT, the boundary will be a sentence boundary, so straddle means:
    # segmenter disagrees with their segmenter. Will need to be excluded from evaluation metrics.

    @model_validator(mode="after")
    def _check_width(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"span ({self.start}, {self.end}) is empty or inverted")
        return self

    @model_serializer
    def _serialise(self) -> dict[str, Any]:
        # skip optional keys when theyre at default
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


class Doc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(min_length=1)  # {source}:{local_id}
    text: str  # NFC, otherwise verbatim
    label: Label
    source: str
    domain: str | None  # None for SeqXGPT, it has no domain
    generator: str | None
    group_id: str = Field(min_length=1)  # a group can never be in two splits
    split_role: str
    sentences: list[SentenceSpan]
    label_raw: str = Field(min_length=1)
    # NOTE RAID's model, MAGE's src, SeqXGPT's generator name

    meta: dict[str, Any] = Field(default_factory=dict)  # per-source stuff, can be dropped

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
        if self.label == LABEL_HUMAN and self.generator is not None and self.source != "mage":
            # MAGE para testbed labels paraphrased human text as machine, only
            # case where a human doc has a generator
            raise ValueError(f"human doc names generator {self.generator!r}")
        return self

    @model_validator(mode="after")
    def _check_text_is_nfc(self) -> Self:
        if not unicodedata.is_normalized("NFC", self.text):
            raise ValueError("text is not NFC-normalised")
        return self

    @model_validator(mode="after")
    def _check_span_coverage(self) -> Self:
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
                raise ValueError(f"span {i} starts {s.start} before previous end {prev_end}")
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
    # no trailing newline
    return orjson.dumps(doc.model_dump())


def doc_from_json(line: bytes | str) -> Doc:
    try:
        return Doc.model_validate(orjson.loads(line))
    except ValidationError as exc:
        raise SchemaError(str(exc)) from exc
    except orjson.JSONDecodeError as exc:
        raise SchemaError(f"malformed JSON: {exc}") from exc


def validate_doc(doc: Doc) -> None:
    # used in testing
    try:
        Doc.model_validate(doc.model_dump())
    except ValidationError as exc:
        raise SchemaError(f"{doc.doc_id}: {exc}") from exc


def iter_jsonl(path: str | Path) -> Iterator[Doc]:
    with Path(path).open("rb") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield doc_from_json(line)
            except SchemaError as exc:
                raise SchemaError(f"{path}:{lineno}: {exc}") from exc
