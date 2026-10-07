"""check features for corpus artefacts + length confounding"""

from pathlib import Path
from typing import Final

import numpy as np
import orjson
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from aivhuman.features import FEATURE_NAMES

# a flag means "look at this", not auto drop. thresholds are heuristic
SMD_FLAG: Final = 0.5  # same class, diff corpus -> feature is partly measuring the corpus
LENGTH_FLAG: Final = 0.5  # |spearman| w/ length -> feature is mostly sentence length
TRANSFER_FLAG: Final = 0.15  # max drop in AUROC edge over 0.5, dev -> mage
SAMPLE: Final = 300_000  # sentences per split, enough for stable stats
SEED: Final = 0


class FeatureVet(BaseModel):
    """vetting result for one feature"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    smd: dict[str, float]  # keys like "human raid-mage"
    length_rho: float  # spearman vs len_tokens, RAID human spans
    auroc_dev: float
    auroc_mage: float
    flags: list[str]


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """standardised mean difference (cohen's d, pooled sd), NaNs ignored"""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    df: pd.DataFrame = pq.read_table(path, columns=columns).to_pandas()
    if len(df) > SAMPLE:
        df = df.sample(SAMPLE, random_state=SEED)
    return df


def _doc_auroc(path: Path, name: str) -> float:
    # single feature detector: doc score = mean of the feature over its sentences
    df = pq.read_table(path, columns=["doc_id", "label", name]).to_pandas()
    doc = df.groupby("doc_id").agg(label=("label", "first"), value=(name, "mean")).dropna()
    return float(roc_auc_score(doc["label"], doc["value"]))


def vet(features_dir: Path) -> list[FeatureVet]:
    """corpus artefact, length and transfer checks for every feature"""
    base = ["label", "span_label", *FEATURE_NAMES]
    raid = _read(features_dir / "train.parquet", base)
    mage = _read(features_dir / "mage-x.parquet", base)
    seqx = _read(features_dir / "seqxgpt-calib.parquet", base)
    seqx["label"] = seqx["span_label"]  # seqxgpt has sentence labels, use them

    out = []
    for name in FEATURE_NAMES:
        # compare corpora within a class, so any gap is source/preprocessing not authorship
        diffs: dict[str, float] = {}
        for cls, cname in ((0, "human"), (1, "machine")):
            r, m, s = (df.loc[df["label"] == cls, name].to_numpy() for df in (raid, mage, seqx))
            diffs[f"{cname} raid-mage"] = smd(r, m)
            diffs[f"{cname} raid-seqxgpt"] = smd(r, s)
        # length confound, on RAID human sentences only so authorship doesnt leak in
        rho = 1.0
        if name != "len_tokens":
            human = raid[raid["label"] == 0][[name, "len_tokens"]].dropna()
            rho = float(spearmanr(human[name], human["len_tokens"]).statistic)
        a_dev = _doc_auroc(features_dir / "dev.parquet", name)
        a_mage = _doc_auroc(features_dir / "mage-x.parquet", name)

        flags = [k for k, v in diffs.items() if abs(v) > SMD_FLAG]
        if name != "len_tokens" and abs(rho) > LENGTH_FLAG:
            flags.append("length")
        # transfer: signal flips direction on mage, or loses > TRANSFER_FLAG of its edge
        edge_dev, edge_mage = a_dev - 0.5, a_mage - 0.5
        if edge_dev * edge_mage < 0 or abs(edge_dev) - abs(edge_mage) > TRANSFER_FLAG:
            flags.append("transfer")
        out.append(
            FeatureVet(
                name=name,
                smd=diffs,
                length_rho=rho,
                auroc_dev=a_dev,
                auroc_mage=a_mage,
                flags=flags,
            )
        )
    return out


def write_report(results: list[FeatureVet], path: Path) -> Path:
    """write thresholds + per feature results as json"""
    report = {
        "thresholds": {
            "smd": SMD_FLAG,
            "length_rho": LENGTH_FLAG,
            "transfer": TRANSFER_FLAG,
        },
        "features": [r.model_dump() for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    return path
