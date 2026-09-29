"""Scoring splits, per-sentence explanations, and the reports built on the learned weights."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from aivhuman.evaluate import write_predictions
from aivhuman.mil.data import Bags, Standardizer, load_bags
from aivhuman.mil.model import MILModel
from aivhuman.mil.train import RunResult, score


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_split(
    model: MILModel, std: Standardizer, features: Path, out: Path, split: str
) -> None:
    """Write doc scores, coverage, and raw sentence logits for one split.

    Span rows carry the final logit, the head's emission, and their difference: the
    neighbour effect of the CRF, zero without one.
    """
    bags = load_bags(features, std.names).standardised(std)
    s = score(model, bags)
    with torch.no_grad():
        emission = model.sentence_logits(torch.from_numpy(bags.x)).numpy().astype(np.float64)
    write_predictions(out / f"{split}.parquet", bags.doc_ids, _sigmoid(s.doc_logits))
    write_predictions(out / f"{split}.coverage.parquet", bags.doc_ids, s.coverage)
    sizes = bags.sizes
    pq.write_table(
        pa.table(
            {
                "doc_id": np.repeat(np.array(bags.doc_ids, dtype=object), sizes),
                "span_idx": np.concatenate([np.arange(k, dtype=np.int32) for k in sizes]),
                "logit": s.sentence_logits,
                "emission": emission,
                "neighbour": s.sentence_logits - emission,
            }
        ),
        out / f"{split}.spans.parquet",
    )


def explain(
    model: MILModel, std: Standardizer, x: np.ndarray, top: int = 3
) -> list[list[tuple[str, float]]]:
    """The `top` largest-magnitude feature contributions per sentence, signed, for raw `x`."""
    with torch.no_grad():
        contrib = model.contributions(torch.from_numpy(std.transform(x))).numpy()
    order = np.argsort(-np.abs(contrib), axis=1)[:, :top]
    return [
        [(std.names[j], float(row[j])) for j in idx]
        for row, idx in zip(contrib, order, strict=True)
    ]


def weights_report(model: MILModel, std: Standardizer, result: RunResult, path: Path) -> Path:
    """Each feature's contribution at fixed standardised values, largest effect first.

    For the linear head the columns are `w * z`; for the GAM they trace each spline.
    """
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
        "crf_stickiness": float(model.stickiness.item()) if model.cfg.crf else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    return path


def faithfulness_report(model: MILModel, bags: Bags, names: Sequence[str], path: Path) -> Path:
    """Compare MIL slopes with a sentence-supervised logistic regression on standardised bags
    that were not used for model selection."""
    keep, y = bags.sentence_labels()
    x = bags.x[keep]
    sup = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(x, y)
    w_sup = sup.coef_.ravel()
    w_mil = model.slopes().numpy()
    active = np.abs(w_mil) > 1e-3
    agree = np.sign(w_sup[active]) == np.sign(w_mil[active])
    rho = float(spearmanr(np.abs(w_sup), np.abs(w_mil)).statistic)
    mil_auroc = float(roc_auc_score(y, score(model, bags).sentence_logits[keep]))
    sup_auroc = float(roc_auc_score(y, sup.decision_function(x)))
    # Collinear features (log-prob and log-rank) make multivariate signs unstable, so each
    # feature's own direction is reported alongside.
    uni = np.array([roc_auc_score(y, x[:, j]) for j in range(x.shape[1])])

    report = {
        "n_spans": int(keep.sum()),
        "n_dropped": int((~keep).sum()),
        "n_active": int(active.sum()),
        "sign_agreement": float(agree.mean()),
        "spearman_rho": rho,
        "univariate_sign_agreement": float(((uni[active] > 0.5) == (w_mil[active] > 0)).mean()),
        "mil_auroc": mil_auroc,
        "supervised_auroc": sup_auroc,
        "features": [
            {
                "name": names[j],
                "mil_slope": float(w_mil[j]),
                "supervised_w": float(w_sup[j]),
                "univariate_auroc": float(uni[j]),
                "agrees": bool(np.sign(w_sup[j]) == np.sign(w_mil[j])) if active[j] else None,
                "agrees_univariate": bool((uni[j] > 0.5) == (w_mil[j] > 0)) if active[j] else None,
            }
            for j in np.argsort(-np.abs(w_mil))
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    return path
