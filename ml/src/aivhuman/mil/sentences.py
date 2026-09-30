"""Sentence-level precision, recall and span overlap of calibrated sentence scores."""

from pathlib import Path
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import average_precision_score, precision_recall_curve

from aivhuman.evaluate import bootstrap_ci
from aivhuman.mil.calibrate import Spans, balanced_weights, bucket_names, bucket_of

OPERATING_FPRS: Final = (0.01, 0.02, 0.05)
"""Per-sentence false-positive rates whose thresholds are fitted on the calibration spans."""
EVEN_THRESHOLD: Final = 0.5

BUCKET_COLOURS: Final = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
"""Ordinal blue ramp, short to long."""
OVERALL_COLOUR: Final = "#52514e"


class PRCell(BaseModel):
    """Precision and recall of one bucket at one threshold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bucket: str
    spans: int
    machine_share: float
    precision: float
    precision_even: float
    """Precision at an even prior: TPR / (TPR + FPR)."""
    recall: float
    fpr: float
    f1: float
    precision_ci: tuple[float, float] | None = None
    recall_ci: tuple[float, float] | None = None


class OperatingPoint(BaseModel):
    """Every sentence-level metric at one threshold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    threshold: float
    cells: list[PRCell]
    iou_docs: int
    """Documents with at least one machine sentence."""
    iou_mean: float
    iou_median: float
    regions_mean: float
    """Mean number of flagged runs among machine-containing documents with any flag."""
    human_docs: int
    false_highlight_rate: float
    """Share of wholly human documents with at least one flagged sentence."""


class SentenceReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    spans: int
    docs: int
    average_precision: dict[str, float]
    average_precision_even: dict[str, float]
    points: list[OperatingPoint]


def threshold_at_fpr(labels: np.ndarray, probs: np.ndarray, fpr: float) -> float:
    """Lowest threshold at which at most `fpr` of human spans score at or above it."""
    human = np.sort(probs[labels == 0])[::-1]
    k = int(np.floor(fpr * len(human)))
    if k >= len(human):
        return float(human[-1])
    return float(np.nextafter(human[k], np.inf))


def _rates(y: np.ndarray, flag: np.ndarray) -> tuple[float, float, float, float]:
    """Precision, even-prior precision, recall and FPR."""
    tp = float((flag & (y == 1)).sum())
    fp = float((flag & (y == 0)).sum())
    recall = tp / max(int((y == 1).sum()), 1)
    fpr = fp / max(int((y == 0).sum()), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    even = recall / (recall + fpr) if recall + fpr else 0.0
    return precision, even, recall, fpr


def pr_cell(y: np.ndarray, probs: np.ndarray, threshold: float, bucket: str) -> PRCell:
    precision, even, recall, fpr = _rates(y, probs >= threshold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRCell(
        bucket=bucket,
        spans=len(y),
        machine_share=float(y.mean()),
        precision=precision,
        precision_even=even,
        recall=recall,
        fpr=fpr,
        f1=f1,
    )


def doc_overlap(spans: Spans, flag: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per document: token-weighted IoU of flagged and machine spans, flagged run count, and
    whether it is wholly human. IoU is NaN for wholly human documents."""
    _, inv = np.unique(spans.doc_ids, return_inverse=True)
    y = spans.labels == 1
    w = spans.n_tokens.astype(np.float64)
    inter = np.bincount(inv, w * (flag & y))
    union = np.bincount(inv, w * (flag | y))
    human = np.bincount(inv, y) == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(human, np.nan, inter / union)
    same_doc = np.r_[False, inv[1:] == inv[:-1]]
    adjacent = same_doc & (np.r_[-2, spans.span_idx[:-1]] == spans.span_idx - 1)
    starts = flag & ~(adjacent & np.r_[False, flag[:-1]])
    return iou, np.bincount(inv, starts), human


def _masks(spans: Spans, edges: list[int]) -> list[tuple[str, np.ndarray]]:
    """Each length bucket with both classes present, then all spans."""
    buckets = bucket_of(spans.n_tokens, edges)
    out = []
    for k, b in enumerate(bucket_names(edges)):
        m = buckets == k
        if len(np.unique(spans.labels[m])) == 2:
            out.append((b, m))
    return [*out, ("all", np.ones(len(buckets), dtype=bool))]


def operating_point(
    spans: Spans, probs: np.ndarray, threshold: float, name: str, edges: list[int]
) -> OperatingPoint:
    """Overall and per-bucket P/R (overall with group-bootstrap intervals), IoU and the
    false-highlight rate at one threshold."""
    cells = [pr_cell(spans.labels[m], probs[m], threshold, b) for b, m in _masks(spans, edges)]
    p_ci, r_ci = bootstrap_ci(
        spans.labels,
        probs,
        spans.groups,
        [lambda y, s: _rates(y, s >= threshold)[0], lambda y, s: _rates(y, s >= threshold)[2]],
    )
    cells[-1] = cells[-1].model_copy(update={"precision_ci": p_ci, "recall_ci": r_ci})

    flag = probs >= threshold
    iou, regions, human = doc_overlap(spans, flag)
    flagged_machine = ~human & (regions > 0)
    return OperatingPoint(
        name=name,
        threshold=threshold,
        cells=cells,
        iou_docs=int((~human).sum()),
        iou_mean=float(np.nanmean(iou)),
        iou_median=float(np.nanmedian(iou)),
        regions_mean=float(regions[flagged_machine].mean()) if flagged_machine.any() else 0.0,
        human_docs=int(human.sum()),
        false_highlight_rate=float((regions[human] > 0).mean()) if human.any() else 0.0,
    )


def evaluate(
    spans: Spans, probs: np.ndarray, thresholds: dict[str, float], edges: list[int]
) -> SentenceReport:
    ap, ap_even = {}, {}
    for b, m in _masks(spans, edges):
        y, p = spans.labels[m], probs[m]
        ap[b] = float(average_precision_score(y, p))
        ap_even[b] = float(average_precision_score(y, p, sample_weight=balanced_weights(y)))
    return SentenceReport(
        split=spans.name,
        spans=len(spans.labels),
        docs=len(np.unique(spans.doc_ids)),
        average_precision=ap,
        average_precision_even=ap_even,
        points=[operating_point(spans, probs, t, n, edges) for n, t in thresholds.items()],
    )


def plot_pr(
    spans: Spans, probs: np.ndarray, report: SentenceReport, edges: list[int], path: Path
) -> None:
    """Even-prior precision against recall per length bucket, with the operating points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colours = dict(zip(bucket_names(edges), BUCKET_COLOURS, strict=True))
    for b, m in _masks(spans, edges):
        colour, lw = (OVERALL_COLOUR, 2.5) if b == "all" else (colours[b], 2.0)
        y = spans.labels[m]
        precision, recall, _ = precision_recall_curve(
            y, probs[m], sample_weight=balanced_weights(y)
        )
        ap = report.average_precision_even[b]
        label = f"{b} tokens (AP {ap:.3f})" if b != "all" else f"all spans (AP {ap:.3f})"
        ax.plot(recall, precision, color=colour, lw=lw, label=label, drawstyle="steps-post")
    for point in report.points:
        cell = point.cells[-1]
        ax.plot(
            cell.recall,
            cell.precision_even,
            "o",
            ms=8,
            color=OVERALL_COLOUR,
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=5,
        )
        ax.annotate(
            point.name,
            (cell.recall, cell.precision_even),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#2b2b29",
        )
    ax.axhline(0.5, color="#b5b4ad", lw=1, ls="--")
    ax.set(
        xlim=(0, 1),
        ylim=(0.4, 1.0),
        xlabel="recall (machine sentences)",
        ylabel="precision at an even prior",
        title=f"Sentence precision-recall, {report.split}",
    )
    ax.grid(color="#e4e3dd", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
