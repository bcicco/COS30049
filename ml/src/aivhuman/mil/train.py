import copy
import random
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from aivhuman.evaluate import partial_auroc, tpr_at_fpr
from aivhuman.mil.data import Bags, Standardizer, batches
from aivhuman.mil.model import MILConfig, MILModel


class Scores(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    doc_logits: np.ndarray
    coverage: np.ndarray
    sentence_logits: np.ndarray  # flat, same order as span rows


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config: MILConfig
    selection: float  # (dev pauc + sent auroc) / 2
    dev_pauc: float
    sentence_auroc: float  # on the seqxgpt-calib val slice
    dev_tpr: float  # @1% fpr
    best_epoch: int
    seconds: float
    history: list[float] = []


@torch.no_grad()
def score(model: MILModel, bags: Bags, batch_size: int = 256) -> Scores:
    model.eval()
    doc = np.empty(len(bags), dtype=np.float64)
    cov = np.empty(len(bags), dtype=np.float64)
    sent = np.empty(len(bags.x), dtype=np.float64)
    for idx, x, mask in batches(bags, range(len(bags)), batch_size):
        d, c, s = model(x, mask)
        doc[idx], cov[idx] = d.numpy(), c.numpy()
        for row, i in enumerate(idx):
            lo, hi = bags.offsets[i], bags.offsets[i + 1]
            sent[lo:hi] = s[row, : hi - lo].numpy()
    return Scores(doc_logits=doc, coverage=cov, sentence_logits=sent)


def _sentence_loss(bags: Bags, idx: list[int], logits: torch.Tensor) -> torch.Tensor:
    # only spans w/ known labels, 0 if none in the batch
    target = torch.full(logits.shape, -1.0)
    for row, i in enumerate(idx):
        lo, hi = bags.offsets[i], bags.offsets[i + 1]
        target[row, : hi - lo] = torch.from_numpy(np.nan_to_num(bags.span_labels[lo:hi], nan=-1.0))
    known = target >= 0
    if not known.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[known], target[known])


def sentence_auroc(model: MILModel, bags: Bags) -> float:
    keep, y = bags.sentence_labels()
    return float(roc_auc_score(y, score(model, bags).sentence_logits[keep]))


def fit(cfg: MILConfig, train: Bags, dev: Bags, sentence_val: Bags) -> tuple[MILModel, RunResult]:
    """train, keep the best epoch by (doc pauc + sentence auroc) / 2"""
    # doc score alone let the sentence scores drift to chance.
    # pauc bc tpr@1% on ~950 human dev docs is like 9 docs, way too noisy
    start = time.time()
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    model = MILModel(train.x.shape[1], cfg)
    model.set_knots(torch.from_numpy(train.x))
    optimiser = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr, weight_decay=0.0)
    labels = torch.from_numpy(train.labels)
    n_machine = float(train.labels.sum())
    pos_weight = torch.tensor((len(train) - n_machine) / n_machine)
    doc_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best: dict[str, Any] = {"selection": -1.0, "epoch": -1}
    history: list[float] = []
    for epoch in range(cfg.epochs):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        for idx, x, mask in batches(train, order, cfg.batch_size):
            y = labels[idx]
            d, _, s = model(x, mask)
            loss = doc_loss(d, y)
            if cfg.sentence_weight:
                loss = loss + cfg.sentence_weight * _sentence_loss(train, idx, s)
            loss = loss + cfg.l1 * model.l1_penalty()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        dev_scores = score(model, dev).doc_logits
        pauc = partial_auroc(dev.labels, dev_scores)
        sent = sentence_auroc(model, sentence_val)
        selection = (pauc + sent) / 2
        history.append(selection)
        if selection > best["selection"]:
            best = {
                "selection": selection,
                "pauc": pauc,
                "sent": sent,
                "tpr": tpr_at_fpr(dev.labels, dev_scores, 0.01)[0],
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
            }
        elif epoch - best["epoch"] >= cfg.patience:
            break
    model.load_state_dict(best["state"])
    return model, RunResult(
        config=cfg,
        selection=best["selection"],
        dev_pauc=best["pauc"],
        sentence_auroc=best["sent"],
        dev_tpr=best["tpr"],
        best_epoch=best["epoch"],
        seconds=time.time() - start,
        history=history,
    )


KEPT: Final = MILConfig(head="gam", pooling="lse", tau=2.0, l1=1e-4)  # winner of the sweep


def sweep_configs(sentence_weights: tuple[float, ...] = ()) -> list[MILConfig]:
    # with sentence weights only rerun the 3 best shapes
    if sentence_weights:
        shapes: list[dict[str, Any]] = [
            {"head": "linear", "pooling": "lse", "tau": 5.0, "l1": 1e-3},
            {"head": "gam", "pooling": "lse", "tau": 2.0, "l1": 1e-4},
            {"head": "gam", "pooling": "lse", "tau": 5.0, "l1": 1e-4},
        ]
        return [MILConfig(**s, sentence_weight=w) for w in sentence_weights for s in shapes]
    pools: list[dict[str, Any]] = [
        {"pooling": "lse", "tau": 2.0},
        {"pooling": "lse", "tau": 5.0},
        {"pooling": "topk", "k": 1},
        {"pooling": "topk", "k": 3},
    ]
    out = [MILConfig(head="linear", pooling="lse", tau=5.0, l1=1e-3)]
    for l1 in (0.0, 1e-4):
        out += [MILConfig(head="gam", l1=l1, **p) for p in pools]
    return out


def save(path: Path, model: MILModel, std: Standardizer, result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "config": model.cfg.model_dump(),
            "standardizer": std.model_dump(),
            "result": result.model_dump(),
        },
        path,
    )


def load(path: Path) -> tuple[MILModel, Standardizer]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    std = Standardizer.model_validate(payload["standardizer"])
    # old ckpts still have crf / coverage loss keys
    cfg = {k: v for k, v in payload["config"].items() if k in MILConfig.model_fields}
    model = MILModel(len(std.names), MILConfig.model_validate(cfg))
    model.load_state_dict(payload["state"])
    return model, std
