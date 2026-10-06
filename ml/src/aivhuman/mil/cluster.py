"""Unsupervised structure within one class."""

# is k means on the standardised sentence features (machine only)
# label hidden (but used afterward)  i.e where human sentences land
# described from where features deviate

from collections import Counter
from pathlib import Path
from typing import Final

import numpy as np
import orjson
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from aivhuman.mil.data import Standardizer
from aivhuman.mil.model import MILModel

K_RANGE: Final = range(3, 9)
FIT_SAMPLE: Final = 200_000
SILHOUETTE_SAMPLE: Final = 20_000
TOP_FEATURES: Final = 3
EXAMPLES: Final = 3
SEED: Final = 0


class Cluster(BaseModel):
    """One cluster: size, composition checks, feature deviations and example sentences."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    size: int
    share: float
    """Share of machine sentences in this cluster."""
    human_share: float
    """Share of human sentences that project onto this centroid; compare with `share`."""
    mean_prob: float
    """Mean sigmoid of the MIL sentence logit, i.e. how machine-like the detector finds it."""
    generators: dict[str, float]
    domains: dict[str, float]
    deviations: dict[str, float]
    """Cluster mean minus machine-class mean, in train standard deviations, all features."""
    top_features: list[str]
    examples: list[str]


class ClusterReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    n_machine: int
    n_human: int
    k: int
    silhouette: dict[int, float]
    clusters: list[Cluster]


def choose_k(z: np.ndarray, rng: np.random.Generator) -> tuple[int, dict[int, float]]:
    """Fit each k on a sample and pick the best silhouette"""
    fit = z[rng.choice(len(z), min(FIT_SAMPLE, len(z)), replace=False)]
    scores: dict[int, float] = {}
    for k in K_RANGE:
        labels = KMeans(k, n_init=4, random_state=SEED).fit_predict(fit)
        idx = rng.choice(len(fit), min(SILHOUETTE_SAMPLE, len(fit)), replace=False)
        scores[k] = float(silhouette_score(fit[idx], labels[idx]))
    best = max(scores, key=lambda k: (round(scores[k], 3), -k))
    return best, scores


def _shares(values: list[str], top: int = 3) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values())
    return {name: round(n / total, 3) for name, n in counts.most_common(top)}


def _sentence_probs(model: MILModel, z: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model.sentence_logits(torch.as_tensor(z, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def _texts(processed: Path, keys: list[tuple[str, int]]) -> dict[tuple[str, int], str]:
    """Pull sentence text for (doc_id, span_idx) keys from the source JSONL, one pass per corpus."""
    wanted: dict[str, set[int]] = {}
    for doc_id, idx in keys:
        wanted.setdefault(doc_id, set()).add(idx)
    out: dict[tuple[str, int], str] = {}
    for source in {d.split(":")[0] for d in wanted}:
        with (processed / f"{source}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                start = (
                    line.index('"doc_id":"') + 10
                )  # cheap prefix check before parsing the row
                doc_id = line[start : line.index('"', start)]
                if doc_id not in wanted:
                    continue
                doc = orjson.loads(line)
                for idx in wanted[doc_id]:
                    span = doc["sentences"][idx]
                    out[(doc_id, idx)] = doc["text"][span["start"] : span["end"]]
    return out


def cluster(
    features: Path,
    processed: Path,
    model: MILModel,
    std: Standardizer,
    split: str,
    k: int | None,
) -> ClusterReport:
    cols = ["doc_id", "span_idx", "label", "domain", "breakdown", *std.names]
    table = pq.read_table(
        features / f"{split}.parquet", columns=list(dict.fromkeys(cols))
    )
    machine = table.filter(pc.equal(table["label"], 1))
    human = table.filter(pc.equal(table["label"], 0))
    zm = std.transform(
        np.column_stack([machine[n].to_numpy(zero_copy_only=False) for n in std.names])
    )
    zh = std.transform(
        np.column_stack([human[n].to_numpy(zero_copy_only=False) for n in std.names])
    )

    rng = np.random.default_rng(SEED)
    scores: dict[int, float] = {}
    if k is None:
        k, scores = choose_k(zm, rng)
    km = KMeans(k, n_init=4, random_state=SEED).fit(zm)
    assign = km.labels_
    human_assign = km.predict(zh)
    probs = _sentence_probs(model, zm)
    class_mean = zm.mean(axis=0)

    # EXAMPLE COLLECTION (nearest to centroid)
    dist = np.linalg.norm(zm - km.cluster_centers_[assign], axis=1)
    doc_ids = machine["doc_id"].to_pylist()
    span_idx = machine["span_idx"].to_pylist()
    nearest = {
        c: m[np.argsort(dist[m])[: EXAMPLES * 10]].tolist()
        for c in range(k)
        for m in [np.flatnonzero(assign == c)]
    }
    texts = _texts(
        processed,
        [(doc_ids[i], span_idx[i]) for rows in nearest.values() for i in rows],
    )
    picks: dict[int, list[int]] = {}
    for c, rows in nearest.items():
        seen: set[str] = set()  # set to keep duplcates at bay
        picks[c] = []
        for i in rows:  # recipe lines repeat across documents
            t = texts[(doc_ids[i], span_idx[i])].strip().lstrip("-*• ").lower()
            if t not in seen:
                seen.add(t)
                picks[c].append(i)
            if len(picks[c]) == EXAMPLES:
                break

    generators = machine["breakdown"].to_pylist()
    domains = machine["domain"].to_pylist()
    clusters = []
    for c in range(k):
        members = assign == c
        dev = zm[members].mean(axis=0) - class_mean
        order = np.argsort(-np.abs(dev))
        clusters.append(
            Cluster(
                index=c,
                size=int(members.sum()),
                share=round(float(members.mean()), 3),
                human_share=round(float(np.mean(human_assign == c)), 3),
                mean_prob=round(float(probs[members].mean()), 3),
                generators=_shares([generators[i] for i in np.flatnonzero(members)]),
                domains=_shares([str(domains[i]) for i in np.flatnonzero(members)]),
                deviations={
                    std.names[j]: round(float(dev[j]), 2) for j in range(len(std.names))
                },
                top_features=[std.names[j] for j in order[:TOP_FEATURES]],
                examples=[texts[(doc_ids[i], span_idx[i])] for i in picks[c]],
            )
        )
    clusters.sort(key=lambda c: -c.size)
    return ClusterReport(
        split=split,
        n_machine=len(zm),
        n_human=len(zh),
        k=k,
        silhouette=scores,
        clusters=clusters,
    )
