from pathlib import Path

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from aivhuman.evaluate import write_predictions
from aivhuman.mil.data import Standardizer, load_bags
from aivhuman.mil.model import MILModel
from aivhuman.mil.train import RunResult, score


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_split(
    model: MILModel, std: Standardizer, features: Path, out: Path, split: str
) -> None:
    bags = load_bags(features, std.names).standardised(std)
    s = score(model, bags)
    write_predictions(out / f"{split}.parquet", bags.doc_ids, _sigmoid(s.doc_logits))
    write_predictions(out / f"{split}.coverage.parquet", bags.doc_ids, s.coverage)
    sizes = bags.sizes
    pq.write_table(
        pa.table(
            {
                "doc_id": np.repeat(np.array(bags.doc_ids, dtype=object), sizes),
                "span_idx": np.concatenate([np.arange(k, dtype=np.int32) for k in sizes]),
                "logit": s.sentence_logits,
            }
        ),
        out / f"{split}.spans.parquet",
    )


def explain(
    model: MILModel, std: Standardizer, x: np.ndarray, top: int = 3
) -> list[list[tuple[str, float]]]:
    # x is raw, gets standardised here
    with torch.no_grad():
        contrib = model.contributions(torch.from_numpy(std.transform(x))).numpy()
    order = np.argsort(-np.abs(contrib), axis=1)[:, :top]
    return [
        [(std.names[j], float(row[j])) for j in idx]
        for row, idx in zip(contrib, order, strict=True)
    ]


def weights_report(model: MILModel, std: Standardizer, result: RunResult, path: Path) -> Path:
    """contribution per feature on a small z grid, biggest effect first"""
    grid = (-2.0, -1.0, 1.0, 2.0)
    n = len(std.names)
    with torch.no_grad():
        terms = np.stack(
            [model.contributions(torch.eye(n) * z).diagonal().numpy() for z in grid], axis=1
        )
    slopes = model.slopes().numpy()
    effect = np.abs(terms).max(axis=1)
    report = {
        "config": result.config.model_dump(),
        "dev_pauc": result.dev_pauc,
        "dev_tpr": result.dev_tpr,
        "best_epoch": result.best_epoch,
        "bias": float(model.bias),
        "grid": list(grid),
        "features": [
            {"name": std.names[j], "slope": float(slopes[j]), "terms": terms[j].tolist()}
            for j in np.argsort(-effect)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    return path
