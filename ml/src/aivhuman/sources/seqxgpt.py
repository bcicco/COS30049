"""SeqXGPT: the only source with sentence-level provenance."""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from aivhuman.group import recover_seqxgpt_groups
from aivhuman.labels import seqxgpt_doc_label
from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE, Doc, SentenceSpan
from aivhuman.text.normalize import CompositionStraddlesBoundary, nfc, nfc_split
from aivhuman.text.segment import Segmenter, n_words
from aivhuman.text.style import detect_style
from aivhuman.text.tokens import count_tokens

MACHINE_CHAR_THRESHOLD: Final = 0.5

# Maps a file stem to the generator it holds, as a cross-check on the per-record
# label field.  *** IS NOT a replacement for it. ***
FILE_GENERATOR: Final = {
    "en_gpt2_lines": "gpt2",
    "en_gpt3_lines": "gpt3re",
    "en_gptj_lines": "gptj",
    "en_gptneo_lines": "gptneo",
    "en_llama_lines": "llama",
    "en_human_lines": "human",
    "gpt2_lines": "gpt2",
    "gpt3_lines": "gpt3re",
    "gptj_lines": "gptj",
    "gptneo_lines": "gptneo",
    "llama_lines": "llama",
    "human_lines": "human",
}


class RawRecord(BaseModel):
    """One line of a SeqXGPT JSONL, with its provenance attached."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_stem: str
    row_index: NonNegativeInt
    text: str
    prompt_len: NonNegativeInt | None
    label_raw: str


class SeqXGPTStats(BaseModel):
    """Structural measurements published in the Phase 1 report."""

    model_config = ConfigDict(extra="forbid")

    records: NonNegativeInt = 0
    quarantined: NonNegativeInt = 0
    label_file_mismatches: NonNegativeInt = 0
    straddle_spans: NonNegativeInt = 0
    total_spans: NonNegativeInt = 0
    nfc_retracted: NonNegativeInt = 0
    nfc_shortened: NonNegativeInt = 0
    boundary_snap_dists: list[int] = Field(default_factory=list)
    styles: dict[str, int] = Field(default_factory=dict)

    @property
    def straddle_rate(self) -> float:
        return self.straddle_spans / self.total_spans if self.total_spans else 0.0

    def as_dict(self) -> dict[str, Any]:
        snaps = sorted(self.boundary_snap_dists)

        def pct(p: float) -> int:
            return snaps[min(len(snaps) - 1, int(p * len(snaps)))] if snaps else 0

        return {
            "records": self.records,
            "quarantined": self.quarantined,
            "label_file_mismatches": self.label_file_mismatches,
            "total_spans": self.total_spans,
            "straddle_spans": self.straddle_spans,
            "straddle_rate": round(self.straddle_rate, 4),
            "nfc_retracted": self.nfc_retracted,
            "nfc_shortened": self.nfc_shortened,
            "boundary_snap_exact_frac": (
                round(sum(1 for d in snaps if d == 0) / len(snaps), 4) if snaps else 0.0
            ),
            "boundary_snap_p50": pct(0.50),
            "boundary_snap_p90": pct(0.90),
            "boundary_snap_max": snaps[-1] if snaps else 0,
            "styles": dict(sorted(self.styles.items())),
        }


def load_records(directory: Path) -> list[RawRecord]:
    """Read every JSONL in `directory` in sorted filename order."""
    out: list[RawRecord] = []
    for path in sorted(directory.glob("*.jsonl")):
        for i, line in enumerate(path.open(encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                RawRecord(
                    file_stem=path.stem,
                    row_index=i,
                    text=d["text"],
                    prompt_len=d.get("prompt_len"),
                    label_raw=d["label"],
                )
            )
    return out


def assign_sentence_labels(
    spans: Sequence[tuple[int, int]], boundary: int
) -> list[tuple[int, float, bool]]:
    """Label each span by how much of it sits past the human/machine boundary.

    Returns [(label, machine_char_frac, straddles), ...]."""

    # ------------------------------- NOTE ------------------------------------
    # I don't love the way I did this. Done mmy best to explain my reasoning below,
    # but is convoluted an a bit confusing. Still useful but should revisit at some point

    # -------------------------------- EXPLANATION -----------------------------

    # The majority-character rule is a choice, and straddles exists so it does
    # not have to be a silent. SeqXGPT's boundary is a *sentence* boundary by
    # construction, so a straddling span means our segmenter disagrees with the one
    # upstream used ----> a segmentation artifact, not genuinely mixed authorship.
    # Phase 7 can therefore exclude straddlers from strict precision and recall and
    # say how many it excluded, rather than folding disagreements into the score.

    out: list[tuple[int, float, bool]] = []
    for start, end in spans:
        width = end - start
        overlap = max(0, end - max(start, boundary))
        frac = overlap / width if width else 0.0
        # messssyyyyy
        label = LABEL_MACHINE if frac >= MACHINE_CHAR_THRESHOLD else LABEL_HUMAN
        out.append((label, frac, 0.0 < frac < 1.0))
    return out


def _boundary_snap_dist(spans: Sequence[tuple[int, int]], boundary: int) -> int:
    """Distance from the boundary to the nearest sentence edge."""
    if not spans:
        return 0
    edges = [spans[0][0]] + [e for _s, e in spans]
    return min(abs(boundary - edge) for edge in edges)


def build_docs(
    directory: Path,
    split_role: str,
    *,
    segmenter: Segmenter | None = None,
    stats: SeqXGPTStats | None = None,
) -> Iterator[Doc]:
    """Normalise a SeqXGPT directory into class Doc objects."""
    seg = segmenter or Segmenter()
    st = stats if stats is not None else SeqXGPTStats()

    records = load_records(directory)
    st.records = len(records)

    assignment = recover_seqxgpt_groups([(r.file_stem, r.text, r.prompt_len) for r in records])

    for rec, group_id in zip(records, assignment.group_ids, strict=True):
        expected = FILE_GENERATOR.get(rec.file_stem)
        if expected is not None and rec.label_raw != expected:
            st.label_file_mismatches += 1

        if rec.prompt_len is None:
            # en_human_lines: wholly human, no boundary to carry.
            text = nfc(rec.text)
            boundary = len(text)
            retract = delta = 0
        else:
            try:
                split = nfc_split(rec.text, rec.prompt_len)
            except (CompositionStraddlesBoundary, ValueError):
                # Quarantined rather than guessed: a boundary we cannot carry
                # through NFC would mislabel every sentence near the transition.
                st.quarantined += 1
                continue
            text, boundary = split.text, split.cut
            retract, delta = split.retract, split.nfc_delta

        if retract:
            st.nfc_retracted += 1
        if delta:
            st.nfc_shortened += 1

        spans = seg.segment(text)
        labels = assign_sentence_labels(spans, boundary)
        _doc_tokens, per_span = count_tokens(text, spans)
        style = detect_style(text)
        st.styles[style.style] = st.styles.get(style.style, 0) + 1
        st.total_spans += len(spans)
        st.straddle_spans += sum(1 for _l, _f, straddle in labels if straddle)
        if rec.prompt_len is not None:
            st.boundary_snap_dists.append(_boundary_snap_dist(spans, boundary))

        sentences = [
            SentenceSpan(
                start=start,
                end=end,
                n_tokens=per_span[i],
                n_words=n_words(text[start:end]),
                label=labels[i][0],
                machine_char_frac=round(labels[i][1], 4),
                straddles_boundary=labels[i][2],
            )
            for i, (start, end) in enumerate(spans)
        ]

        doc_label = seqxgpt_doc_label(rec.label_raw, boundary, len(text))
        yield Doc(
            doc_id=f"seqxgpt:{rec.file_stem}:{rec.row_index:06d}",
            text=text,
            label=doc_label,
            source="seqxgpt",
            domain=None,
            generator=None if doc_label == LABEL_HUMAN else rec.label_raw,
            group_id=group_id,
            split_role=split_role,
            sentences=sentences,
            label_raw=rec.label_raw,
            meta={
                "file_stem": rec.file_stem,
                "row_index": rec.row_index,
                "prompt_len_orig": rec.prompt_len,
                "boundary_nfc": boundary,
                "nfc_retract": retract,
                "nfc_delta": delta,
                "n_chars": len(text),
                "detok_style": style.style,
                "uppercase_ratio": style.uppercase_ratio,
                "spaced_punct_ratio": style.spaced_punct_ratio,
            },
        )
