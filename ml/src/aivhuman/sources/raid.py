"""RAID: the only corpus this project trains on."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from aivhuman.group import raid_group_id
from aivhuman.labels import RAID_DOMAINS, raid_label
from aivhuman.schema import LABEL_HUMAN, Doc, SentenceSpan
from aivhuman.sources.raid_parquet import CLEAN_ATTACK, open_clean
from aivhuman.text.normalize import nfc
from aivhuman.text.segment import Segmenter, n_words
from aivhuman.text.style import detect_style
from aivhuman.text.tokens import count_tokens

# Every clean RAID row lands in one pool (all train)
SPLIT_ROLE: Final = "train_pool"

# Rows per parquet batch
BATCH_SIZE: Final = 2048

MAX_EXAMPLES: Final = 20


class RawRow(BaseModel):
    """One row of the derived clean parquet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    source_id: str
    adv_source_id: str
    model: str
    """RAID's generator column. `"human"` here *is* the label."""

    decoding: str
    repetition_penalty: str
    attack: str
    domain: str
    title: str
    generation: str
    prompt_sha: str
    prompt_len_chars: int


class RaidStats(BaseModel):
    """Structural and integrity measurements"""

    model_config = ConfigDict(validate_assignment=False, extra="forbid")

    rows: NonNegativeInt = 0
    docs: NonNegativeInt = 0
    empty_text: NonNegativeInt = 0
    human_rows: NonNegativeInt = 0
    machine_rows: NonNegativeInt = 0
    unknown_domains: dict[str, int] = Field(default_factory=dict)
    attacks_seen: dict[str, int] = Field(default_factory=dict)
    by_model: dict[str, int] = Field(default_factory=dict)
    by_domain: dict[str, int] = Field(default_factory=dict)
    by_decoding: dict[str, int] = Field(default_factory=dict)
    styles: dict[str, int] = Field(default_factory=dict)

    group_domain: dict[str, str] = Field(default_factory=dict)
    """First domain seen per `source_id`, to detect groups spanning domains."""

    multi_domain_groups: list[str] = Field(default_factory=list)
    groups_with_human: set[str] = Field(default_factory=set)

    @property
    def n_groups(self) -> int:
        return len(self.group_domain)

    @property
    def groups_without_human(self) -> int:
        """Groups holding machine rows but no human original."""
        # -------------------- IMPORTANT, SLIGHTLY CONFUSING ----------------
        # Not a corruption!! -- it means raid ood on that domain has no negatives,
        # so a per-domain TPR-at-fixed-FPR number cannot be computed there.

        return self.n_groups - len(self.groups_with_human)

    @property
    def machine_per_human(self) -> float:
        return self.machine_rows / self.human_rows if self.human_rows else 0.0

    @property
    def integrity_ok(self) -> bool:
        """Everything checkable without touching the text."""
        return (
            not self.unknown_domains
            and set(self.attacks_seen) <= {CLEAN_ATTACK}
            and not self.multi_domain_groups
            and self.human_rows > 0
            and self.machine_rows > 0
        )

    @property
    def is_green(self) -> bool:
        """Whether a full func build_docs pass may be used."""
        return self.integrity_ok and self.docs + self.empty_text == self.rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "docs": self.docs,
            "empty_text": self.empty_text,
            "human_rows": self.human_rows,
            "machine_rows": self.machine_rows,
            "machine_per_human": round(self.machine_per_human, 2),
            "n_groups": self.n_groups,
            "groups_without_human": self.groups_without_human,
            "multi_domain_groups": sorted(self.multi_domain_groups),
            "unknown_domains": dict(sorted(self.unknown_domains.items())),
            "attacks_seen": dict(sorted(self.attacks_seen.items())),
            "by_model": dict(sorted(self.by_model.items())),
            "by_domain": dict(sorted(self.by_domain.items())),
            "by_decoding": dict(sorted(self.by_decoding.items())),
            "styles": dict(sorted(self.styles.items())),
        }


def load_rows(path: Path, *, batch_size: int = BATCH_SIZE) -> Iterator[RawRow]:
    """Stream the derived clean parquet in file order."""
    handle = open_clean(path)
    for batch in handle.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield RawRow(**row)


def build_docs(
    path: Path,
    *,
    batch_size: int = BATCH_SIZE,
    segmenter: Segmenter | None = None,
    stats: RaidStats | None = None,
) -> Iterator[Doc]:
    """Normalise the clean RAID parquet into class Doc objects, in file order."""
    seg = segmenter or Segmenter()
    st = stats if stats is not None else RaidStats()

    for row in load_rows(path, batch_size=batch_size):
        account(row, st)
        doc = to_doc(row, seg)
        if doc is not None:
            count_doc(doc, st)
            yield doc


def scan(path: Path, *, batch_size: int = BATCH_SIZE, stats: RaidStats | None = None) -> RaidStats:
    """Populate every counter that does not require the text"""

    st = stats if stats is not None else RaidStats()
    for row in load_rows(path, batch_size=batch_size):
        account(row, st)
    return st


def count_doc(doc: Doc, st: RaidStats) -> None:
    """Record the counters only a built document can supply."""

    # --------------------NOTE-----------------------------------------
    # Separated from _account because _to_doc` runs in a worker
    # process where a  stats object is a copy that gets thrown away.

    st.docs += 1
    style = str(doc.meta["detok_style"])
    st.styles[style] = st.styles.get(style, 0) + 1


def to_doc(row: RawRow, seg: Segmenter) -> Doc | None:
    """Build one document. Pure, so it can run in a worker process."""
    label = raid_label(row.model)

    text = nfc(row.generation)
    if not text.strip():
        return None

    spans = seg.segment(text)
    _doc_tokens, per_span = count_tokens(text, spans)
    style = detect_style(text)

    return Doc(
        doc_id=f"raid:{row.id}",
        text=text,
        label=label,
        source="raid",
        domain=row.domain,
        generator=None if label == LABEL_HUMAN else row.model,
        group_id=raid_group_id(row.source_id),
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
        label_raw=row.model,
        meta={
            "source_id": row.source_id,
            "adv_source_id": row.adv_source_id,
            "attack": row.attack,
            "decoding": row.decoding,
            "repetition_penalty": row.repetition_penalty,
            "title": row.title,
            "prompt_sha": row.prompt_sha,
            "prompt_len_chars": row.prompt_len_chars,
            "n_chars": len(text),
            "detok_style": style.style,
            "uppercase_ratio": style.uppercase_ratio,
            "spaced_punct_ratio": style.spaced_punct_ratio,
        },
    )


def account(row: RawRow, st: RaidStats) -> int:
    """Every counter that reads metadata rather than text. Returns the label."""
    st.rows += 1
    label = raid_label(row.model)

    st.by_model[row.model] = st.by_model.get(row.model, 0) + 1
    st.by_domain[row.domain] = st.by_domain.get(row.domain, 0) + 1
    st.by_decoding[row.decoding] = st.by_decoding.get(row.decoding, 0) + 1
    st.attacks_seen[row.attack] = st.attacks_seen.get(row.attack, 0) + 1
    if row.domain not in RAID_DOMAINS:
        # extra.csv
        st.unknown_domains[row.domain] = st.unknown_domains.get(row.domain, 0) + 1

    if label == LABEL_HUMAN:
        st.human_rows += 1
        st.groups_with_human.add(row.source_id)
    else:
        st.machine_rows += 1

    seen_domain = st.group_domain.setdefault(row.source_id, row.domain)
    if seen_domain != row.domain and len(st.multi_domain_groups) < MAX_EXAMPLES:
        # A source_id spanning two domains would mean the grouping key is not
        # the document identity we think it is
        st.multi_domain_groups.append(row.source_id)

    if not row.generation.strip():
        st.empty_text += 1

    return label
