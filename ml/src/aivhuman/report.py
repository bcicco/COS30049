"""The dataloading / normalising report: one pass over the JSONL, written as CSV."""

import csv
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import orjson
from pydantic import BaseModel, ConfigDict, Field

from aivhuman.schema import LABEL_HUMAN, iter_jsonl
from aivhuman.text.normalize import stable_hash, text_key

LENGTH_BUCKETS: Final = ((0, 15), (15, 30), (30, 60), (60, 1 << 30))

# ModernBERT's context, and the chunking threshold
TOKEN_LIMITS: Final = (4096, 8192)


class CorpusSummary(BaseModel):
    """Everything the report needs about one corpus, from one pass over it."""

    model_config = ConfigDict(validate_assignment=False, extra="forbid")

    source: str = ""
    docs: int = 0
    spans: int = 0
    human_docs: int = 0
    machine_docs: int = 0
    by_domain: dict[str, int] = Field(default_factory=dict)
    by_generator: dict[str, int] = Field(default_factory=dict)
    by_split_role: dict[str, int] = Field(default_factory=dict)
    by_style: dict[str, int] = Field(default_factory=dict)
    style_by_label: dict[str, dict[str, int]] = Field(default_factory=dict)
    groups: int = 0
    span_length_buckets: dict[str, int] = Field(default_factory=dict)
    docs_over_token_limit: dict[str, int] = Field(default_factory=dict)
    spans_per_doc: list[int] = Field(default_factory=list)
    doc_tokens: list[int] = Field(default_factory=list)
    straddling_spans: int = 0
    labelled_spans: int = 0
    machine_spans: int = 0

    @property
    def machine_per_human(self) -> float:
        return self.machine_docs / self.human_docs if self.human_docs else 0.0

    @property
    def human_frac(self) -> float:
        return self.human_docs / self.docs if self.docs else 0.0

    @property
    def straddle_rate(self) -> float:
        return self.straddling_spans / self.labelled_spans if self.labelled_spans else 0.0

    def quantiles(self, values: list[int]) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)

        def pick(p: float) -> float:
            return float(ordered[min(len(ordered) - 1, int(p * len(ordered)))])

        return {
            "min": float(ordered[0]),
            "p50": pick(0.50),
            "p90": pick(0.90),
            "p99": pick(0.99),
            "max": float(ordered[-1]),
            "mean": round(statistics.fmean(ordered), 2),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "docs": self.docs,
            "spans": self.spans,
            "groups": self.groups,
            "human_docs": self.human_docs,
            "machine_docs": self.machine_docs,
            "human_frac": round(self.human_frac, 4),
            "machine_per_human": round(self.machine_per_human, 2),
            "by_split_role": dict(sorted(self.by_split_role.items())),
            "by_domain": dict(sorted(self.by_domain.items())),
            "by_generator": dict(sorted(self.by_generator.items())),
            "by_style": dict(sorted(self.by_style.items())),
            "style_by_label": {
                k: dict(sorted(v.items())) for k, v in sorted(self.style_by_label.items())
            },
            "span_length_buckets": dict(self.span_length_buckets),
            "docs_over_token_limit": dict(sorted(self.docs_over_token_limit.items())),
            "doc_tokens": self.quantiles(self.doc_tokens),
            "spans_per_doc": self.quantiles(self.spans_per_doc),
            "labelled_spans": self.labelled_spans,
            "machine_spans": self.machine_spans,
            "straddling_spans": self.straddling_spans,
            "straddle_rate": round(self.straddle_rate, 4),
        }


def _bucket_name(low: int, high: int) -> str:
    return f"{low}-{high}" if high < (1 << 30) else f"{low}+"


def summarise(
    path: Path, *, content_keys: dict[str, tuple[str, int]] | None = None
) -> CorpusSummary:
    """One pass over one JSONL."""
    summary = CorpusSummary()
    groups: set[str] = set()
    buckets = {_bucket_name(lo, hi): 0 for lo, hi in LENGTH_BUCKETS}
    over = {str(limit): 0 for limit in TOKEN_LIMITS}

    for doc in iter_jsonl(path):
        summary.source = summary.source or doc.source
        summary.docs += 1
        summary.spans += len(doc.sentences)
        groups.add(doc.group_id)
        if doc.label == LABEL_HUMAN:
            summary.human_docs += 1
        else:
            summary.machine_docs += 1

        domain = doc.domain or "(none)"
        generator = doc.generator or "(human)"
        style = str(doc.meta.get("detok_style", "(unknown)"))
        summary.by_domain[domain] = summary.by_domain.get(domain, 0) + 1
        summary.by_generator[generator] = summary.by_generator.get(generator, 0) + 1
        summary.by_split_role[doc.split_role] = summary.by_split_role.get(doc.split_role, 0) + 1
        summary.by_style[style] = summary.by_style.get(style, 0) + 1
        label_name = "human" if doc.label == LABEL_HUMAN else "machine"
        summary.style_by_label.setdefault(style, {})
        summary.style_by_label[style][label_name] = (
            summary.style_by_label[style].get(label_name, 0) + 1
        )
        doc_tokens = 0
        for span in doc.sentences:
            doc_tokens += span.n_tokens
            for lo, hi in LENGTH_BUCKETS:
                if lo <= span.n_tokens < hi:
                    buckets[_bucket_name(lo, hi)] += 1
                    break
            if span.label is not None:
                summary.labelled_spans += 1
                if span.label != LABEL_HUMAN:
                    summary.machine_spans += 1
            if span.straddles_boundary:
                summary.straddling_spans += 1

        summary.doc_tokens.append(doc_tokens)
        summary.spans_per_doc.append(len(doc.sentences))
        for limit in TOKEN_LIMITS:
            if doc_tokens > limit:
                over[str(limit)] += 1

        if content_keys is not None:
            content_keys.setdefault(stable_hash(text_key(doc.text)), (doc.doc_id, doc.label))

    summary.groups = len(groups)
    summary.span_length_buckets = buckets
    summary.docs_over_token_limit = over
    return summary


# SHAPE: (section, source, metric, value).
Row = tuple[str, str, str, Any]

METRICS_HEADER: Final = ("section", "source", "metric", "value")
FINDINGS_HEADER: Final = ("kind", "subject", "expected", "measured", "consequence")

# Where measurement contradicted expected
FINDINGS: Final = (
    (
        "contradiction",
        "RAID class balance",
        "unstated",
        "34:1 machine:human, 13,371 human documents",
        "83% accuracy floor is below the 97.1% base rate; report TPR at fixed FPR",
    ),
    (
        "contradiction",
        "MAGE domains",
        "10",
        "14 (cnn, dialogsum, imdb, pubmed are extra)",
        "the four extras appear only in the OOD testbeds",
    ),
    (
        "contradiction",
        "MAGE polarity",
        "unstated",
        "inverted: 1 is human",
        "flipped labels",
    ),
    (
        "contradiction",
        "MAGE paraphrase set",
        "all paraphrased",
        "1,600 paraphrased plus 762 originals",
        'filter on meta["is_paraphrased"]',
    ),
    (
        "contradiction",
        "RAID test split",
        "usable",
        "unlabeled leaderboard split",
        "train/dev and raid-ood both come out of train.csv",
    ),
    (
        "contradiction",
        "SeqXGPT base ids",
        "assumed present",
        "absent, recovered at 99.4%",
        "grouping is reconstructed from human prefixes",
    ),
    (
        "contradiction",
        "pysbd",
        "assumed total",
        "drops text, and hangs on 3 RAID documents",
        "coverage is enforced; hangs are bounded by a task timeout",
    ),
    (
        "contradiction",
        "SeqXGPT structure",
        "mixed text",
        "exactly one human-to-machine transition per document",
        "CRF and Span-IoU results do not transfer to alternately edited text",
    ),
    (
        "gap",
        "non-native English writing",
        "",
        "not identified in any corpus",
        "false positive rate on that group cannot be measured",
    ),
    (
        "gap",
        "generator recency",
        "",
        "all corpora built with 2023-24 generators",
        "results are a lower bound on difficulty against current models",
    ),
    (
        "gap",
        "document token counts",
        "",
        "sum over spans, excluding inter-sentence whitespace",
        "slight undercount; safe for context-limit checks",
    ),
)

# duplicated from label.py, not ideal but ive been doing this for too long to care
POLARITY_RULES: Final = {
    "raid": ('model == "human"', "no label column exists"),
    "mage": ('label == "1"', "inverted: 1 is human"),
    "seqxgpt": (
        'label == "human" or prompt covers the text',
        "per-record generator name",
    ),
}

_SEGMENT_COUNTERS: Final = (
    "spans",
    "nonws_gap_chars",
    "failed",
    "repaired",
    "recovered_gap_spans",
    "unsegmentable_gaps",
    "single_span_docs",
    "numeric_ref_fallbacks",
)


def metric_rows(
    summaries: list[CorpusSummary],
    sidecars: dict[str, dict[str, Any]],
    overlap: dict[str, Any] | None = None,
    verify: list[dict[str, Any]] | None = None,
) -> list[Row]:
    """Flatten every reported number into long-format rows."""
    rows: list[Row] = []

    def add(section: str, source: str, values: dict[str, Any]) -> None:
        rows.extend((section, source, k, v) for k, v in values.items())

    for s in summaries:
        side = sidecars.get(s.source, {})
        add(
            "corpus",
            s.source,
            {
                "docs": s.docs,
                "spans": s.spans,
                "groups": s.groups,
                "human_docs": s.human_docs,
                "machine_docs": s.machine_docs,
                "human_frac": round(s.human_frac, 4),
                "machine_per_human": round(s.machine_per_human, 2),
                "majority_baseline_acc": (round(s.machine_docs / s.docs, 4) if s.docs else 0.0),
                "is_green": side.get("is_green", ""),
            },
        )
        if s.source in POLARITY_RULES:
            rule, note = POLARITY_RULES[s.source]
            add("polarity", s.source, {"human_when": rule, "note": note})
        add("split_role", s.source, dict(sorted(s.by_split_role.items())))
        add("domain", s.source, dict(sorted(s.by_domain.items())))
        add("generator", s.source, dict(sorted(s.by_generator.items())))
        add("detok_style", s.source, dict(sorted(s.by_style.items())))
        add("doc_tokens", s.source, s.quantiles(s.doc_tokens))
        add("spans_per_doc", s.source, s.quantiles(s.spans_per_doc))
        add("docs_over_tokens", s.source, s.docs_over_token_limit)
        add("span_length_bucket", s.source, s.span_length_buckets)
        if s.labelled_spans:
            add(
                "sentence_labels",
                s.source,
                {
                    "labelled_spans": s.labelled_spans,
                    "machine_spans": s.machine_spans,
                    "straddling_spans": s.straddling_spans,
                    "straddle_rate": round(s.straddle_rate, 4),
                },
            )
        seg = side.get("segment_stats", {})
        if seg:
            add("segmentation", s.source, {k: seg.get(k, 0) for k in _SEGMENT_COUNTERS})
            add(
                "segmentation",
                s.source,
                {
                    "timed_out_batches": side.get("timed_out_batches", 0),
                    "forced_fallback_docs": ";".join(side.get("forced_fallback_docs") or []),
                },
            )

    if overlap is not None:
        human_both = overlap.get("shared_keys_human_both_sides", {})
        for pair, count in sorted(overlap.get("shared_keys", {}).items()):
            human = human_both.get(pair, 0)
            add(
                "overlap",
                pair,
                {
                    "shared_texts": count,
                    "human_both_sides": human,
                    "other": count - human,
                },
            )
        for source, count in sorted(overlap.get("internal_duplicate_docs", {}).items()):
            add("internal_duplicates", source, {"docs": count})
        add("overlap", "all", {"is_clean": bool(overlap.get("is_clean"))})

    for v in verify or []:
        source = str(v.get("source") or Path(str(v.get("path"))).name)
        add(
            "verify",
            source,
            {
                "docs": v.get("docs", 0),
                "spans": v.get("spans", 0),
                "ok": bool(v.get("ok")),
                "n_problems": v.get("n_problems", 0),
            },
        )
    return rows


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def build(
    processed_dir: Path,
    reports_dir: Path,
    *,
    overlap: dict[str, Any] | None = None,
    verify: list[dict[str, Any]] | None = None,
) -> Path:
    """Summarise every JSONL and write the metrics CSV, findings CSV and JSON."""
    summaries: list[CorpusSummary] = []
    sidecars: dict[str, dict[str, Any]] = {}
    keys: dict[str, dict[str, tuple[str, int]]] = {}

    for path in sorted(processed_dir.glob("*.jsonl")):
        collected: dict[str, tuple[str, int]] = {}
        summary = summarise(path, content_keys=collected)
        summaries.append(summary)
        keys[summary.source] = collected
        sidecar = path.with_name(f"{path.stem}.stats.json")
        if sidecar.exists():
            sidecars[summary.source] = orjson.loads(sidecar.read_bytes())

    if overlap is None:
        overlap = _overlap_from_keys(keys, {s.source: s.docs for s in summaries})

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "phase1_data.json").write_bytes(
        orjson.dumps(
            {
                "corpora": [s.as_dict() for s in summaries],
                "overlap": overlap,
                "verify": verify,
            },
            option=orjson.OPT_INDENT_2,
        )
    )
    _write_csv(reports_dir / "phase1_findings.csv", FINDINGS_HEADER, FINDINGS)
    path = reports_dir / "phase1_metrics.csv"
    _write_csv(path, METRICS_HEADER, metric_rows(summaries, sidecars, overlap, verify))
    return path


def _overlap_from_keys(
    keys: dict[str, dict[str, tuple[str, int]]], docs: dict[str, int]
) -> dict[str, Any]:
    """Pair up the content keys collected during the summary pass.

    `shared_keys_human_both_sides` is the distinction that decides what to do
    about an overlap. Two corpora drawing the same public human documents is
    expected and mostly harmless; a machine generation appearing in a test set
    is a leak of the model's own training text.
    """
    shared: dict[str, int] = {}
    human_only: dict[str, int] = {}
    examples: dict[str, list[list[str]]] = {}
    names = sorted(keys)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            pair = f"{left}|{right}"
            common = sorted(keys[left].keys() & keys[right].keys())
            shared[pair] = len(common)
            human_only[pair] = sum(
                1
                for k in common
                if keys[left][k][1] == LABEL_HUMAN and keys[right][k][1] == LABEL_HUMAN
            )
            if common:
                examples[pair] = [[keys[left][k][0], keys[right][k][0]] for k in common[:10]]
    return {
        "shared_keys": shared,
        "shared_keys_human_both_sides": human_only,
        "unique_keys": {k: len(v) for k, v in keys.items()},
        # Derivable rather than counted: a key is recorded once per corpus, so
        # whatever a corpus has beyond its unique keys is an internal repeat.
        "internal_duplicate_docs": {k: docs.get(k, 0) - len(v) for k, v in keys.items()},
        "is_clean": not any(shared.values()),
        "examples": examples,
    }
