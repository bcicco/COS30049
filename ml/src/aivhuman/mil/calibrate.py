"""Isotonic calibration of sentence logits per length bucket, and its reliability report.

Calibrated scores assume an even prior: each bucket is fitted with class-balancing weights,
so a score is evidence for machine authorship and a bucket's own base rate does not move it.
"""

from collections.abc import Callable, Collection, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Final

import numpy as np
import orjson
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from aivhuman.evaluate import bootstrap_ci
from aivhuman.mil.data import Standardizer, load_bags
from aivhuman.mil.model import MILModel
from aivhuman.mil.train import score

EDGES: Final = (15, 30, 60)
"""Token-count bucket edges, left-closed: `<15, 15-30, 30-60, 60+`."""
SHORT_CAP: Final = 0.75
"""Scores in the shortest bucket are clipped to `[1 - SHORT_CAP, SHORT_CAP]`."""
BINS: Final = 10
ECE_TARGET: Final = 0.05
MIN_STYLE_SPANS: Final = 200
"""Smallest bucket-by-style cell the report scores."""


def bucket_of(n_tokens: np.ndarray, edges: Sequence[int] = EDGES) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), n_tokens, side="right")


def bucket_names(edges: Sequence[int] = EDGES) -> list[str]:
    inner = [f"{a}-{b}" for a, b in pairwise(edges)]
    return [f"<{edges[0]}", *inner, f"{edges[-1]}+"]


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    """Weights giving each class half the total mass."""
    n1 = max(int(labels.sum()), 1)
    n0 = max(len(labels) - n1, 1)
    return np.where(labels == 1, 0.5 / n1, 0.5 / n0)


def shift_prior(probs: np.ndarray, prior: float) -> np.ndarray:
    """Move even-prior probabilities to a machine base rate of `prior`."""
    num = probs * prior
    return num / (num + (1 - probs) * (1 - prior))


class Calibrator(BaseModel):
    """Piecewise-linear isotonic map from sentence logit to probability, one per bucket."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    edges: list[int]
    short_cap: float
    x: list[list[float]]
    y: list[list[float]]
    n_fit: list[int]

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        n_tokens: np.ndarray,
        labels: np.ndarray,
        edges: Sequence[int] = EDGES,
        short_cap: float = SHORT_CAP,
    ) -> "Calibrator":
        buckets = bucket_of(n_tokens, edges)
        xs, ys, ns = [], [], []
        for k in range(len(edges) + 1):
            m = buckets == k
            if len(np.unique(labels[m])) != 2:
                raise ValueError(f"bucket {k} needs both classes to fit")
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(logits[m], labels[m], sample_weight=balanced_weights(labels[m]))
            xs.append(iso.X_thresholds_.tolist())
            ys.append(iso.y_thresholds_.tolist())
            ns.append(int(m.sum()))
        return cls(edges=list(edges), short_cap=short_cap, x=xs, y=ys, n_fit=ns)

    def apply(self, logits: np.ndarray, n_tokens: np.ndarray, cap: bool = True) -> np.ndarray:
        buckets = bucket_of(n_tokens, self.edges)
        probs = np.empty(len(logits), dtype=np.float64)
        for k, (x, y) in enumerate(zip(self.x, self.y, strict=True)):
            m = buckets == k
            probs[m] = np.interp(logits[m], x, y)
        if cap:
            short = buckets == 0
            probs[short] = np.clip(probs[short], 1 - self.short_cap, self.short_cap)
        return probs

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(self.model_dump()))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        return cls.model_validate(orjson.loads(path.read_bytes()))


def reliability(
    labels: np.ndarray, probs: np.ndarray, balanced: bool = True, bins: int = BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per equal-width bin: mean predicted, observed machine rate, and share of the weight.
    Empty bins are NaN with zero weight."""
    w = balanced_weights(labels) if balanced else np.full(len(labels), 1.0 / len(labels))
    idx = np.minimum((probs * bins).astype(int), bins - 1)
    mass = np.bincount(idx, weights=w, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred = np.bincount(idx, weights=w * probs, minlength=bins) / mass
        obs = np.bincount(idx, weights=w * labels, minlength=bins) / mass
    return pred, obs, mass / mass.sum()


def ece(labels: np.ndarray, probs: np.ndarray, balanced: bool = True, bins: int = BINS) -> float:
    """Expected calibration error over equal-width bins, class-balanced by default."""
    pred, obs, share = reliability(labels, probs, balanced, bins)
    ok = share > 0
    return float(np.sum(share[ok] * np.abs(pred[ok] - obs[ok])))


class Spans(BaseModel):
    """Labelled, non-straddling sentences of one split with their raw logits."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    labels: np.ndarray
    logits: np.ndarray
    n_tokens: np.ndarray
    styles: np.ndarray
    groups: np.ndarray
    doc_ids: np.ndarray
    span_idx: np.ndarray
    """Position of each span in its document."""


def load_spans(
    name: str,
    path: Path,
    model: MILModel,
    std: Standardizer,
    group_of: Callable[[str], str],
    keep: Collection[str] | None = None,
) -> Spans:
    bags = load_bags(path, std.names, keep).standardised(std)
    logits = score(model, bags).sentence_logits
    filters = [("doc_id", "in", list(keep))] if keep is not None else None
    meta = pq.read_table(path, columns=["doc_id", "len_tokens", "detok_style"], filters=filters)
    doc_ids = np.repeat(np.asarray(bags.doc_ids, dtype=object), bags.sizes)
    if not (meta["doc_id"].to_numpy(zero_copy_only=False) == doc_ids).all():
        raise ValueError(f"{path}: span metadata is not aligned with the bags")
    rows, labels = bags.sentence_labels()
    span_idx = np.arange(len(doc_ids)) - np.repeat(bags.offsets[:-1], bags.sizes)
    return Spans(
        name=name,
        labels=labels,
        logits=logits[rows],
        n_tokens=meta["len_tokens"].to_numpy(zero_copy_only=False)[rows],
        styles=meta["detok_style"].to_numpy(zero_copy_only=False)[rows],
        groups=np.array([group_of(d) for d in doc_ids[rows]]),
        doc_ids=doc_ids[rows],
        span_idx=span_idx[rows],
    )


class CellMetrics(BaseModel):
    """Calibration of one split, bucket and style."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    bucket: str
    style: str = "all"
    spans: int
    machine_share: float
    auroc_raw: float
    auroc: float
    """On calibrated scores; below `auroc_raw` only through ties the isotonic fit creates."""
    ece: float
    ece_ci: tuple[float, float] | None = None
    ece_uncapped: float
    ece_at_prior: float
    """Unweighted ECE after shifting scores to this cell's own machine share."""
    p01: float
    p99: float
    reliability: list[list[float | None]]
    """Per bin: mean predicted, observed rate, weight share."""


def _cell(
    cal: Calibrator, spans: Spans, m: np.ndarray, bucket: str, style: str, ci: bool
) -> CellMetrics | None:
    y = spans.labels[m]
    if len(y) < MIN_STYLE_SPANS or y.min() == y.max():
        return None
    logits, n = spans.logits[m], spans.n_tokens[m]
    probs = cal.apply(logits, n)
    prior = float(y.mean())
    interval = None
    if ci:
        (interval,) = bootstrap_ci(y, probs, spans.groups[m], [ece])
    pred, obs, share = reliability(y, probs)
    return CellMetrics(
        split=spans.name,
        bucket=bucket,
        style=style,
        spans=len(y),
        machine_share=prior,
        auroc_raw=float(roc_auc_score(y, logits)),
        auroc=float(roc_auc_score(y, probs)),
        ece=ece(y, probs),
        ece_ci=interval,
        ece_uncapped=ece(y, cal.apply(logits, n, cap=False)),
        ece_at_prior=ece(y, shift_prior(probs, prior), balanced=False),
        p01=float(np.percentile(probs, 1)),
        p99=float(np.percentile(probs, 99)),
        reliability=[[None if np.isnan(v) else float(v) for v in r] for r in (pred, obs, share)],
    )


def evaluate(cal: Calibrator, spans: Spans, by_style: bool = False) -> list[CellMetrics]:
    """Per-bucket and overall cells, with group-bootstrap intervals; optionally per style."""
    buckets = bucket_of(spans.n_tokens, cal.edges)
    names = bucket_names(cal.edges)
    masks = [(names[k], buckets == k) for k in range(len(names))]
    masks.append(("all", np.ones(len(buckets), dtype=bool)))
    out = []
    for bucket, m in masks:
        if by_style:
            for style in sorted(set(spans.styles[m])):
                cell = _cell(cal, spans, m & (spans.styles == style), bucket, style, ci=False)
                if cell is not None:
                    out.append(cell)
        else:
            cell = _cell(cal, spans, m, bucket, "all", ci=True)
            if cell is not None:
                out.append(cell)
    return out


def report(
    cal: Calibrator,
    cells: list[CellMetrics],
    style_cells: list[CellMetrics],
    cap_moved: tuple[int, int],
    out_dir: Path,
) -> Path:
    """Write `calibration.json` and the reliability diagrams."""
    moved, short = cap_moved
    payload = {
        "calibrator": {"edges": cal.edges, "short_cap": cal.short_cap, "n_fit": cal.n_fit},
        "bins": BINS,
        "ece_target": ECE_TARGET,
        "cells": [c.model_dump() for c in cells],
        "style_cells": [c.model_dump() for c in style_cells],
        "min_style_spans": MIN_STYLE_SPANS,
        "cap_moved": {"moved": moved, "short_spans": short},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibration.json"
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    _plot([c for c in cells if c.bucket != "all"], "split", out_dir / "reliability.png")
    _plot([c for c in style_cells if c.bucket != "all"], "style", out_dir / "reliability_style.png")
    return path


def _plot(cells: list[CellMetrics], series: str, path: Path) -> None:
    """One panel per bucket: reliability curve per series, diagonal, and the first series'
    weight per bin as faint bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not cells:
        return
    buckets = list(dict.fromkeys(c.bucket for c in cells))
    fig, axes = plt.subplots(2, (len(buckets) + 1) // 2, figsize=(10, 9), squeeze=False)
    centres = (np.arange(BINS) + 0.5) / BINS
    for ax, bucket in zip(axes.flat, buckets, strict=False):
        ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
        first = True
        for c in (c for c in cells if c.bucket == bucket):
            pred, obs, share = (np.array(r, dtype=float) for r in c.reliability)
            ok = ~np.isnan(pred)
            label = getattr(c, series)
            ax.plot(pred[ok], obs[ok], marker="o", ms=3, label=f"{label} (ECE {c.ece:.3f})")
            if first:
                ax.bar(centres, share, width=1 / BINS, alpha=0.12, color="C0")
                first = False
        ax.set(title=f"{bucket} tokens", xlim=(0, 1), ylim=(0, 1), xlabel="predicted")
        ax.set_ylabel("observed machine rate (balanced)")
        ax.legend(fontsize=7, loc="upper left")
    for ax in list(axes.flat)[len(buckets) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
