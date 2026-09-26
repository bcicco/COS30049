"""Re-read the ingested JSONL from disk and re-assert every invariant."""

# ----------------------- REASONING FOR THIS EXISTING----------------------

# Loading a class Doc already validates most of the schema with model_validation decorators, which is the
# design.  This adds what construction cannot see:

# - text[start:end] really is the sentence, trimmed, for every span.
# - n_words still matches the text it describes.
# - doc_id is unique across *all three* sources, not just within one.
# - the line count matches the docs in the sidecar, so a file that lost its
#  tail is caught rather than quietly read short.


from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import orjson
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from aivhuman.schema import (
    LABEL_HUMAN,
    SOURCES,
    Doc,
    SchemaError,
    iter_jsonl,
)
from aivhuman.text.normalize import is_nfc
from aivhuman.text.segment import n_words

MAX_PROBLEMS: Final = 50


class VerifyReport(BaseModel):
    """What one JSONL file turned out to contain."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    source: str = ""
    docs: NonNegativeInt = 0
    spans: NonNegativeInt = 0
    human_docs: NonNegativeInt = 0
    machine_docs: NonNegativeInt = 0
    labelled_span_docs: NonNegativeInt = 0
    problems: list[str] = Field(default_factory=list)
    n_problems: NonNegativeInt = 0
    sidecar_docs: int | None = None

    @property
    def ok(self) -> bool:
        return self.n_problems == 0

    def note(self, message: str) -> None:
        self.n_problems += 1
        if len(self.problems) < MAX_PROBLEMS:
            self.problems.append(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source": self.source,
            "docs": self.docs,
            "spans": self.spans,
            "human_docs": self.human_docs,
            "machine_docs": self.machine_docs,
            "labelled_span_docs": self.labelled_span_docs,
            "sidecar_docs": self.sidecar_docs,
            "ok": self.ok,
            "n_problems": self.n_problems,
            "problems": self.problems,
        }


def verify_file(path: Path, *, seen_doc_ids: set[str] | None = None) -> VerifyReport:
    """Re-read one JSONL and check every document in it"""
    report = VerifyReport(path=path)
    ids = seen_doc_ids if seen_doc_ids is not None else set()

    try:
        for doc in iter_jsonl(path):
            report.docs += 1
            report.spans += len(doc.sentences)
            _check_doc(doc, report)
            if doc.doc_id in ids:
                report.note(f"duplicate doc_id {doc.doc_id!r}")
            ids.add(doc.doc_id)
            if report.source and doc.source != report.source:
                report.note(
                    f"{doc.doc_id}: source {doc.source!r} in a {report.source!r} file"
                )
            report.source = report.source or doc.source
            if doc.label == LABEL_HUMAN:
                report.human_docs += 1
            else:
                report.machine_docs += 1
            if any(s.label is not None for s in doc.sentences):
                report.labelled_span_docs += 1
    except SchemaError as exc:
        # A malformed line is fatal for the file: line numbers after it cannot
        # be trusted
        report.note(f"unreadable: {exc}")
        return report

    _check_sidecar(path, report)
    return report


def verify_all(directory: Path) -> list[VerifyReport]:
    """Verify every `*.jsonl` in `directory`, sharing the `doc_id` set."""
    seen: set[str] = set()
    return [
        verify_file(path, seen_doc_ids=seen)
        for path in sorted(directory.glob("*.jsonl"))
    ]


def iter_problems(reports: list[VerifyReport]) -> Iterator[str]:
    for report in reports:
        for problem in report.problems:
            yield f"{report.path.name}: {problem}"


def _check_doc(doc: Doc, report: VerifyReport) -> None:
    """Everything about one document that loading it did not already prove."""
    if doc.source not in SOURCES:
        report.note(f"{doc.doc_id}: unknown source {doc.source!r}")
    if not is_nfc(doc.text):
        report.note(f"{doc.doc_id}: text is not NFC")
    if not doc.text.strip():
        report.note(f"{doc.doc_id}: empty text")
    if not doc.sentences:
        report.note(f"{doc.doc_id}: no spans, so nothing for MIL to pool")

    for i, span in enumerate(doc.sentences):
        sentence = doc.text[span.start : span.end]
        if not sentence:
            report.note(f"{doc.doc_id}: span {i} is empty in the text")
            continue
        if sentence != sentence.strip():
            report.note(f"{doc.doc_id}: span {i} is not trimmed: {sentence[:40]!r}")
        actual = n_words(sentence)
        if span.n_words != actual:
            report.note(
                f"{doc.doc_id}: span {i} claims {span.n_words} words, text has {actual}"
            )

    # SeqXGPT is the only source with sentence provenance; a label anywhere else
    # is a training leak, and a missing one there breaks ground truth.
    labelled = sum(s.label is not None for s in doc.sentences)
    if doc.source == "seqxgpt" and labelled != len(doc.sentences):
        report.note(
            f"{doc.doc_id}: {len(doc.sentences) - labelled} spans without a label"
        )
    if doc.source != "seqxgpt" and labelled:
        report.note(
            f"{doc.doc_id}: {labelled} sentence labels in a {doc.source} document"
        )


def _check_sidecar(path: Path, report: VerifyReport) -> None:
    """Compare the file against the sidecar the ingest wrote beside it."""
    sidecar = path.with_name(f"{path.stem}.stats.json")
    if not sidecar.exists():
        report.note(
            f"no sidecar at {sidecar.name}; provenance for this file is unknown"
        )
        return

    payload = orjson.loads(sidecar.read_bytes())
    report.sidecar_docs = int(payload.get("docs", -1))
    if report.sidecar_docs != report.docs:
        report.note(f"sidecar says {report.sidecar_docs} docs, file has {report.docs}")
    if not payload.get("is_green", False):
        report.note("sidecar reports the ingest was not green")
