"""Evaluation harness: split loading, prediction files, and document-level metrics."""

# A model is evaluated by writing one {split}.parquet of (doc_id, score) per split, with
# higher scores meaning machine. Metrics are computed from those files and the JSONL labels
# alone, so every model is measured by the same code.

import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import roc_auc_score, roc_curve

from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE, LABEL_NAMES

SPLIT_SOURCE: Final = {
    "train": "raid",
    "dev": "raid",
    "raid-ood": "raid",
    "mage-x": "mage",
    "mage-para": "mage",
    "seqxgpt-calib": "seqxgpt",
    "seqxgpt-test": "seqxgpt",
    "train-adv": "raid-attacks",
    "dev-adv": "raid-attacks",
    "raid-ood-adv": "raid-attacks",
}
ATTACK_MANIFESTS: Final = "attacks"
"""Subdirectory of the manifests holding the adversarial splits, which share groups with the
clean split each was derived from."""

EVAL_SPLITS: Final = ("dev", "raid-ood", "mage-x", "mage-para")

FPR_TARGETS: Final = (0.01, 0.001)

PAUC_MAX_FPR: Final = 0.1
N_BOOTSTRAP: Final = 200

COMMENTARY: Final = re.compile(r"\b(?:paraphras|rephras)\w*", re.IGNORECASE)
# The paraphraser talking about its task instead of doing it ("I cannot paraphrase ..."),
# use this to drop.


class EvalDoc(BaseModel):
    """The fields of a Doc that training and evaluation need."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    text: str
    label: int
    group_id: str
    domain: str | None
    breakdown: str
    """Generator name, `human`, with `_para` appended for paraphrased text."""


class SplitMetrics(BaseModel):
    """Document-level metrics for one model on one split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    split: str
    n_human: int
    n_machine: int
    auroc: float
    pauc_10pct: float
    """Standardised partial AUROC over FPR in [0, 0.1]; 0.5 is chance."""
    pauc_ci: tuple[float, float]
    tpr_at_1pct_fpr: float
    tpr_at_1pct_ci: tuple[float, float]
    """95% bootstrap interval, resampling whole groups."""
    tpr_at_01pct_fpr: float
    tpr_at_01pct_ci: tuple[float, float] | None = None
    balanced_acc: float
    """At a fixed 0.5 threshold on the model's probability."""
    threshold_1pct: float
    threshold_01pct: float | None = None
    by_generator: dict[str, float]
    """TPR per generator at `threshold_1pct`; FPR for `human`."""
    by_generator_01pct: dict[str, float] | None = None
    """The same at `threshold_01pct`."""
    by_domain: dict[str, float]
    """Flag rate per `{domain}/{class}` at `threshold_1pct`: TPR for machine, FPR for human."""


# ---------------------------------------- DATA LOADERS ----------------------------------------


def load_manifest(manifests_dir: Path, split: str) -> dict[str, str]:
    """doc_id -> group_id for one split."""
    payload: dict[str, str] = orjson.loads(manifest_path(manifests_dir, split).read_bytes())
    return payload


def manifest_path(manifests_dir: Path, split: str) -> Path:
    sub = ATTACK_MANIFESTS if SPLIT_SOURCE[split] == "raid-attacks" else ""
    return manifests_dir / sub / f"{split}.json"


def breakdown_name(d: dict[str, Any]) -> str:
    """Generator (or human) of a JSONL record, marked`_para if paraphrased and suffixed
    attack if adversarial."""
    name = str(d["generator"] or "human")
    if d["meta"].get("is_paraphrased"):
        name = f"{name}_para"
    attack = d["meta"].get("attack", "none")
    return name if attack == "none" else f"{name}@{attack}"


def load_splits(
    processed_dir: Path, manifests_dir: Path, splits: Iterable[str]
) -> dict[str, list[EvalDoc]]:
    """Load several splits, reading each source JSONL once."""
    wanted = list(splits)
    out: dict[str, list[EvalDoc]] = {s: [] for s in wanted}
    for source in dict.fromkeys(SPLIT_SOURCE[s] for s in wanted):
        owner = {
            doc_id: split
            for split in wanted
            if SPLIT_SOURCE[split] == source
            for doc_id in load_manifest(manifests_dir, split)
        }
        for doc in _read_docs(processed_dir / f"{source}.jsonl", owner):
            out[owner[doc.doc_id]].append(doc)
    return out


def _read_docs(path: Path, keep: dict[str, str]) -> Iterator[EvalDoc]:
    # reads phase 1 norm. corpora
    with path.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = orjson.loads(line)
            if d["doc_id"] not in keep:
                continue
            yield EvalDoc(
                doc_id=d["doc_id"],
                text=d["text"],
                label=d["label"],
                group_id=d["group_id"],
                domain=d["domain"],
                breakdown=breakdown_name(d),
            )


# ------------------------------ EVALUATION -------------------------------


def drop_commentary(docs: Sequence[EvalDoc]) -> tuple[list[EvalDoc], int]:
    """Paraphrased docs in which the paraphraser comments on its task removed, and their count."""
    kept = [d for d in docs if not (d.breakdown.endswith("_para") and COMMENTARY.search(d.text))]
    return kept, len(docs) - len(kept)


def sample_per_group(
    docs: Sequence[EvalDoc], machine_per_group: int, rng: random.Random
) -> list[int]:
    """Indices of every human doc plus up to machine_per_group machine docs per group."""
    machine: dict[str, list[int]] = defaultdict(list)
    keep = []
    for i, d in enumerate(docs):
        if d.label == LABEL_HUMAN:
            keep.append(i)
        else:
            machine[d.group_id].append(i)
    for group in sorted(machine):
        members = machine[group]
        keep.extend(rng.sample(members, min(machine_per_group, len(members))))
    return keep


def write_predictions(path: Path, doc_ids: list[str], scores: np.ndarray) -> None:
    """Write (doc_id, score) for one split."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"doc_id": doc_ids, "score": np.asarray(scores, dtype=np.float64)})
    pq.write_table(table, path)


def read_predictions(path: Path) -> dict[str, float]:
    table = pq.read_table(path)
    return dict(zip(table["doc_id"].to_pylist(), table["score"].to_pylist(), strict=True))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr_target: float) -> tuple[float, float]:
    """Highest TPR with FPR at or below the target, and the threshold that achieves it."""
    if len(np.unique(labels)) != 2:
        raise ValueError("TPR at FPR needs both classes present")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    ok = fpr <= fpr_target
    i = int(np.flatnonzero(ok)[np.argmax(tpr[ok])])
    return float(tpr[i]), float(thresholds[i])


def partial_auroc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = PAUC_MAX_FPR) -> float:
    """Standardised (McClish) AUROC over FPR in `[0, max_fpr]`: 0.5 is chance, 1 is perfect."""
    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    stats: Sequence[Callable[[np.ndarray, np.ndarray], float]],
    n: int = N_BOOTSTRAP,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """95% percentile interval for each statistic, resampling whole groups with replacement."""
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rng = np.random.default_rng(seed)
    values: list[list[float]] = [[] for _ in stats]
    for _ in range(n):
        g = rng.integers(0, len(counts), len(counts))
        lens = counts[g]
        offsets = np.repeat(starts[g] - np.r_[0, np.cumsum(lens)[:-1]], lens)
        idx = order[offsets + np.arange(lens.sum())]
        y, s = labels[idx], scores[idx]
        if y.min() == y.max():
            continue
        for out, stat in zip(values, stats, strict=True):
            out.append(stat(y, s))
    return [(float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for v in values]


def compute_metrics(
    model: str, split: str, docs: list[EvalDoc], predictions: dict[str, float]
) -> SplitMetrics:
    """Metrics for one split. Every document in the split must have a prediction."""
    missing = [d.doc_id for d in docs if d.doc_id not in predictions]
    if missing:
        raise ValueError(f"{model}/{split}: {len(missing)} docs without a prediction")
    labels = np.array([d.label == LABEL_MACHINE for d in docs], dtype=np.int8)
    scores = np.array([predictions[d.doc_id] for d in docs], dtype=np.float64)

    tpr1, thr1 = tpr_at_fpr(labels, scores, FPR_TARGETS[0])
    tpr01, thr01 = tpr_at_fpr(labels, scores, FPR_TARGETS[1])
    groups = np.array([d.group_id for d in docs])
    tpr_ci, tpr01_ci, pauc_ci = bootstrap_ci(
        labels,
        scores,
        groups,
        [
            lambda y, s: tpr_at_fpr(y, s, FPR_TARGETS[0])[0],
            lambda y, s: tpr_at_fpr(y, s, FPR_TARGETS[1])[0],
            partial_auroc,
        ],
    )
    pred = scores >= 0.5
    balanced = 0.5 * (pred[labels == 1].mean() + (~pred[labels == 0]).mean())

    names = np.array([d.breakdown for d in docs])
    by_generator, by_generator_01 = (
        {str(name): float((scores[names == name] >= t).mean()) for name in sorted(set(names))}
        for t in (thr1, thr01)
    )
    cells = np.array([f"{d.domain or 'none'}/{LABEL_NAMES[d.label]}" for d in docs])
    by_domain = {
        str(cell): float((scores[cells == cell] >= thr1).mean()) for cell in sorted(set(cells))
    }
    return SplitMetrics(
        model=model,
        split=split,
        n_human=int((labels == 0).sum()),
        n_machine=int(labels.sum()),
        auroc=float(roc_auc_score(labels, scores)),
        pauc_10pct=partial_auroc(labels, scores),
        pauc_ci=pauc_ci,
        tpr_at_1pct_fpr=tpr1,
        tpr_at_1pct_ci=tpr_ci,
        tpr_at_01pct_fpr=tpr01,
        tpr_at_01pct_ci=tpr01_ci,
        balanced_acc=float(balanced),
        threshold_1pct=thr1,
        threshold_01pct=thr01,
        by_generator=by_generator,
        by_generator_01pct=by_generator_01,
        by_domain=by_domain,
    )


def evaluate_model(
    model: str,
    predictions_dir: Path,
    splits: dict[str, list[EvalDoc]],
) -> list[SplitMetrics]:
    """Metrics for every split that has a prediction file."""
    out = []
    for split, docs in splits.items():
        path = predictions_dir / model / f"{split}.parquet"
        if path.exists():
            out.append(compute_metrics(model, split, docs, read_predictions(path)))
    return out


def write_report(metrics: list[SplitMetrics], directory: Path, stem: str = "baselines") -> Path:
    """Write the full results as JSON."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.json"
    path.write_bytes(orjson.dumps([m.model_dump() for m in metrics], option=orjson.OPT_INDENT_2))
    return path
