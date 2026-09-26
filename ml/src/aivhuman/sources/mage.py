"""MAGE: the cross-corpus generalisation set."""

import csv
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from aivhuman.acquire import MAGE_ENCODING, MAGE_FILES
from aivhuman.group import mage_group_id
from aivhuman.labels import mage_label, parse_src
from aivhuman.schema import LABEL_HUMAN, Doc, SentenceSpan
from aivhuman.text.normalize import nfc
from aivhuman.text.segment import Segmenter, n_words
from aivhuman.text.style import detect_style
from aivhuman.text.tokens import count_tokens

COLUMNS: Final = ["text", "label", "src"]

# *** NOTE ****
# Rows exceed the 128 KB default. Not `sys.maxsize`: on Windows that raises
# OverflowError, might be a problem on diff. OS
_FIELD_SIZE_LIMIT: Final = 2**31 - 1


# ood == out of domain, i.e. not in the training set
SPLIT_ROLE: Final = {
    "train": "xcorpus_test",
    "valid": "xcorpus_test",
    "test": "xcorpus_test",
    "ood_gpt": "xcorpus_ood_test",
    "ood_gpt_para": "xcorpus_para_test",
}

MAX_UNPARSED_EXAMPLES: Final = 20

# Breakdown key for rows whose src did not parse
UNPARSED: Final = "<unparsed>"


class RawRow(BaseModel):
    """One CSV row"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    row_index: NonNegativeInt
    text: str
    label: str

    src: str


class MageStats(BaseModel):
    """Structural measurements to keep track of integrity"""

    model_config = ConfigDict(validate_assignment=False, extra="forbid")

    rows: NonNegativeInt = 0
    docs: NonNegativeInt = 0
    empty_text: NonNegativeInt = 0
    human_rows: NonNegativeInt = 0
    machine_rows: NonNegativeInt = 0
    para_rows: NonNegativeInt = 0
    para_human_rows: NonNegativeInt = 0
    unparsed_src: NonNegativeInt = 0
    label_generator_disagreements: NonNegativeInt = 0
    unparsed_examples: list[str] = Field(default_factory=list)
    by_split: dict[str, int] = Field(default_factory=dict)
    by_domain: dict[str, int] = Field(default_factory=dict)
    by_generator: dict[str, int] = Field(default_factory=dict)
    styles: dict[str, int] = Field(default_factory=dict)

    @property
    def human_frac(self) -> float:
        labelled = self.human_rows + self.machine_rows
        return self.human_rows / labelled if labelled else 0.0

    @property
    def integrity_ok(self) -> bool:
        return (
            self.unparsed_src == 0
            and self.label_generator_disagreements == 0
            and self.human_rows > 0
            and self.machine_rows > 0
        )

    @property
    def is_green(self) -> bool:
        return self.integrity_ok and self.docs + self.empty_text == self.rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "docs": self.docs,
            "empty_text": self.empty_text,
            "human_rows": self.human_rows,
            "machine_rows": self.machine_rows,
            "human_frac": round(self.human_frac, 4),
            "para_rows": self.para_rows,
            "para_human_rows": self.para_human_rows,
            "unparsed_src": self.unparsed_src,
            "unparsed_examples": sorted(self.unparsed_examples),
            "label_generator_disagreements": self.label_generator_disagreements,
            "by_split": dict(sorted(self.by_split.items())),
            "by_domain": dict(sorted(self.by_domain.items())),
            "by_generator": dict(sorted(self.by_generator.items())),
            "styles": dict(sorted(self.styles.items())),
        }


def load_rows(path: Path, split: str) -> Iterator[RawRow]:
    """Stream one MAGE CSV."""
    csv.field_size_limit(_FIELD_SIZE_LIMIT)
    with path.open(encoding=MAGE_ENCODING, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path.name}: expected columns {COLUMNS}, got {reader.fieldnames}")
        for i, row in enumerate(reader):
            yield RawRow(
                split=split,
                row_index=i,
                text=row["text"],
                label=row["label"],
                src=row["src"],
            )


def iter_rows(directory: Path, splits: Sequence[str] | None = None) -> Iterator[RawRow]:
    """Stream every requested split in file order."""
    for split in splits if splits is not None else list(MAGE_FILES):
        yield from load_rows(directory / MAGE_FILES[split], split)


def build_docs(
    directory: Path,
    *,
    splits: Sequence[str] | None = None,
    segmenter: Segmenter | None = None,
    stats: MageStats | None = None,
) -> Iterator[Doc]:
    """Normalise MAGE's CSVs doc objects, in file order."""
    seg = segmenter or Segmenter()
    st = stats if stats is not None else MageStats()

    for row in iter_rows(directory, splits):
        account(row, st)
        doc = to_doc(row, seg)
        if doc is not None:
            count_doc(doc, st)
            yield doc


def scan(
    directory: Path,
    *,
    splits: Sequence[str] | None = None,
    stats: MageStats | None = None,
) -> MageStats:
    """Populate counters that do not require segmenting the text."""
    st = stats if stats is not None else MageStats()
    for row in iter_rows(directory, splits):
        account(row, st)
    return st


def count_doc(doc: Doc, st: MageStats) -> None:
    """Record the counters that only a built document can supply."""
    st.docs += 1
    style = str(doc.meta["detok_style"])
    st.styles[style] = st.styles.get(style, 0) + 1


def account(row: RawRow, st: MageStats) -> None:
    """Every counter that reads the row rather than the segmented text."""
    st.rows += 1
    st.by_split[row.split] = st.by_split.get(row.split, 0) + 1

    label = mage_label(row.label)
    if label == LABEL_HUMAN:
        st.human_rows += 1
    else:
        st.machine_rows += 1

    parsed = parse_src(row.src)
    domain, generator = (parsed.domain, parsed.generator) if parsed.ok else (None, None)
    if not parsed.ok:
        # Counted, not raised, and the row is still emitted
        st.unparsed_src += 1
        if (
            # for debug, we can probably remove if wanna
            len(st.unparsed_examples) < MAX_UNPARSED_EXAMPLES
            and row.src not in st.unparsed_examples
        ):
            st.unparsed_examples.append(row.src)

    if parsed.is_paraphrased:
        st.para_rows += 1
        if generator == "human":
            st.para_human_rows += 1
    # was very confused with polarity for debug, probably dont need this anymore
    elif generator is not None and (generator == "human") != (label == LABEL_HUMAN):
        st.label_generator_disagreements += 1

    domain_key = domain if domain is not None else UNPARSED
    generator_key = generator if generator is not None else UNPARSED
    st.by_domain[domain_key] = st.by_domain.get(domain_key, 0) + 1
    st.by_generator[generator_key] = st.by_generator.get(generator_key, 0) + 1

    if not nfc(row.text).strip():
        # A document with no sentences carries no instances, so MIL has nothing
        # to pool. Dropped rather than emitted as an empty bag.
        st.empty_text += 1


def to_doc(row: RawRow, seg: Segmenter) -> Doc | None:
    """Build one document. Pure, so it can run in a worker process."""
    label = mage_label(row.label)
    parsed = parse_src(row.src)
    domain, generator = (parsed.domain, parsed.generator) if parsed.ok else (None, None)

    text = nfc(row.text)
    if not text.strip():
        return None

    spans = seg.segment(text)
    _doc_tokens, per_span = count_tokens(text, spans)
    style = detect_style(text)

    return Doc(
        doc_id=f"mage:{row.split}:{row.row_index:07d}",
        text=text,
        label=label,
        source="mage",
        domain=domain,
        generator=generator,
        group_id=mage_group_id(text),
        split_role=SPLIT_ROLE[row.split],
        sentences=[
            SentenceSpan(
                start=start,
                end=end,
                n_tokens=per_span[i],
                n_words=n_words(text[start:end]),
            )
            for i, (start, end) in enumerate(spans)
        ],
        label_raw=row.src,
        meta={
            "split": row.split,
            "row_index": row.row_index,
            "src": row.src,
            "strategy": parsed.strategy,
            "is_paraphrased": parsed.is_paraphrased,
            "n_chars": len(text),
            "detok_style": style.style,
            "uppercase_ratio": style.uppercase_ratio,
            "spaced_punct_ratio": style.spaced_punct_ratio,
        },
    )
