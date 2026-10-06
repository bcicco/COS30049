"""DAIGT v2: student essays against many LLMs, the essay-domain generalisation set."""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE, Doc, SentenceSpan
from aivhuman.text.normalize import nfc, stable_hash, text_key
from aivhuman.text.segment import Segmenter, n_words
from aivhuman.text.style import detect_style
from aivhuman.text.tokens import count_tokens

COLUMNS: Final = ["text", "label", "prompt_name", "source", "RDizzl3_seven"]
SPLIT_ROLE: Final = "xcorpus_essay_test"

# Same reason as mage.py: long rows, and `sys.maxsize` overflows on Windows.
_FIELD_SIZE_LIMIT: Final = 2**31 - 1


class RawRow(BaseModel):
    """One CSV row. `source` is the generator for AI rows and the human corpus otherwise."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    row_index: NonNegativeInt
    text: str
    label: str
    prompt_name: str
    source: str
    rdizzl3_seven: bool


class DaigtStats(BaseModel):
    """Structural measurements to keep track of integrity"""

    model_config = ConfigDict(validate_assignment=False, extra="forbid")

    rows: NonNegativeInt = 0
    docs: NonNegativeInt = 0
    empty_text: NonNegativeInt = 0
    human_rows: NonNegativeInt = 0
    machine_rows: NonNegativeInt = 0
    bad_labels: NonNegativeInt = 0
    by_domain: dict[str, int] = Field(default_factory=dict)
    by_generator: dict[str, int] = Field(default_factory=dict)
    styles: dict[str, int] = Field(default_factory=dict)

    @property
    def integrity_ok(self) -> bool:
        return self.bad_labels == 0 and self.human_rows > 0 and self.machine_rows > 0

    @property
    def is_green(self) -> bool:
        return self.integrity_ok and self.docs + self.empty_text == self.rows

    def as_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        for key in ("by_domain", "by_generator", "styles"):
            d[key] = dict(sorted(d[key].items()))
        return d


def label(row: RawRow) -> int:
    """Same polarity as ours: 1 is AI."""
    return {"0": LABEL_HUMAN, "1": LABEL_MACHINE}[row.label]


def generator(row: RawRow) -> str | None:
    return None if label(row) == LABEL_HUMAN else row.source


def iter_rows(path: Path) -> Iterator[RawRow]:
    """Stream the CSV in file order."""
    csv.field_size_limit(_FIELD_SIZE_LIMIT)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path.name}: expected columns {COLUMNS}, got {reader.fieldnames}")
        for i, row in enumerate(reader):
            yield RawRow(
                row_index=i,
                text=row["text"],
                label=row["label"],
                prompt_name=row["prompt_name"],
                source=row["source"],
                rdizzl3_seven=row["RDizzl3_seven"] == "True",
            )


def scan(path: Path) -> DaigtStats:
    """Populate counters that do not require segmenting the text."""
    st = DaigtStats()
    for row in iter_rows(path):
        account(row, st)
    return st


def account(row: RawRow, st: DaigtStats) -> None:
    """Every counter that reads the row rather than the segmented text."""
    st.rows += 1
    if row.label not in ("0", "1"):
        st.bad_labels += 1
        return
    if label(row) == LABEL_HUMAN:
        st.human_rows += 1
    else:
        st.machine_rows += 1
    gen = generator(row) or "human"
    st.by_domain[row.prompt_name] = st.by_domain.get(row.prompt_name, 0) + 1
    st.by_generator[gen] = st.by_generator.get(gen, 0) + 1
    if not nfc(row.text).strip():
        st.empty_text += 1


def to_doc(row: RawRow, seg: Segmenter) -> Doc | None:
    """Build one document. Pure, so it can run in a worker process."""
    text = nfc(row.text)
    if not text.strip():
        return None

    spans = seg.segment(text)
    _doc_tokens, per_span = count_tokens(text, spans)
    style = detect_style(text)

    return Doc(
        doc_id=f"daigt:{row.row_index:07d}",
        text=text,
        label=label(row),
        source="daigt",
        domain=row.prompt_name,
        generator=generator(row),
        # No source document field, so each essay is its own group, as in MAGE.
        group_id=f"daigt:{stable_hash(text_key(text))}",
        split_role=SPLIT_ROLE,
        sentences=[
            SentenceSpan(
                start=start,
                end=end,
                n_tokens=per_span[i],
                n_words=n_words(text[start:end]),
            )
            for i, (start, end) in enumerate(spans)
        ],
        label_raw=row.label,
        meta={
            "row_index": row.row_index,
            "raw_source": row.source,
            "rdizzl3_seven": row.rdizzl3_seven,
            "n_chars": len(text),
            "detok_style": style.style,
            "uppercase_ratio": style.uppercase_ratio,
            "spaced_punct_ratio": style.spaced_punct_ratio,
        },
    )
