"""Robustness: per-attack comparisons on RAID and the paraphrase gap on MAGE, per feature group."""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import orjson
import torch
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import roc_auc_score

from aivhuman.evaluate import COMMENTARY, load_manifest, partial_auroc, tpr_at_fpr
from aivhuman.features import LENGTH_FEATURES, LEXICAL_FEATURES, LM_FEATURES, SYNTAX_FEATURES
from aivhuman.features.vet import smd
from aivhuman.mil.data import Bags, Standardizer, load_bags
from aivhuman.mil.model import MILModel
from aivhuman.mil.train import KEPT, fit, score
from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE
from aivhuman.sources.raid_attacks import ADV_PARENT, ALL_ATTACKS, TRAIN_ATTACKS

FEATURE_GROUPS: Final = {
    "reference-LM": LM_FEATURES,
    "lexical": LEXICAL_FEATURES,
    "syntactic": SYNTAX_FEATURES,
    "length": LENGTH_FEATURES,
}


class Contrast(BaseModel):
    """Machine and human documents to separate, drawn from one or two feature files."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    machine: dict[str, list[str]]
    """Feature file stem -> doc_ids."""
    human: dict[str, list[str]]


class Scored(BaseModel):
    """Doc scores and doc-mean feature-group contributions for one contrast."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    labels: np.ndarray
    doc_logits: np.ndarray
    groups: np.ndarray
    """[n_docs, n_groups] mean contribution per sentence of each feature group."""


def _metrics(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    return float(roc_auc_score(y, s)), partial_auroc(y, s), tpr_at_fpr(y, s, 0.01)[0]


def _group_columns(names: Sequence[str]) -> dict[str, list[int]]:
    return {g: [names.index(f) for f in fs if f in names] for g, fs in FEATURE_GROUPS.items()}


def _score(model: MILModel, std: Standardizer, bags: Bags) -> tuple[np.ndarray, np.ndarray]:
    """Doc logits and doc-mean group contributions for standardised bags."""
    doc = score(model, bags).doc_logits
    with torch.no_grad():
        contrib = model.contributions(torch.from_numpy(bags.x.astype(np.float32))).numpy()
    cols = _group_columns(std.names)
    per_span = np.column_stack([contrib[:, c].sum(1) for c in cols.values()])
    sums = np.add.reduceat(per_span, bags.offsets[:-1], axis=0)
    return doc, sums / bags.sizes[:, None]


class Scorer:
    """Scores contrasts with one model, caching standardised bags per feature file."""

    def __init__(self, model: MILModel, std: Standardizer, features: Path) -> None:
        self.model, self.std, self.features = model, std, features
        self._bags: dict[tuple[str, frozenset[str]], Bags] = {}

    def bags(self, stem: str, ids: Sequence[str]) -> Bags:
        key = (stem, frozenset(ids))
        if key not in self._bags:
            raw = load_bags(self.features / f"{stem}.parquet", self.std.names, keep=set(ids))
            self._bags[key] = raw.standardised(self.std)
        return self._bags[key]

    def contrast(self, c: Contrast) -> Scored:
        labels, logits, groups = [], [], []
        for side, label in ((c.machine, LABEL_MACHINE), (c.human, LABEL_HUMAN)):
            for stem, ids in side.items():
                doc, grp = _score(self.model, self.std, self.bags(stem, ids))
                labels.append(np.full(len(doc), label))
                logits.append(doc)
                groups.append(grp)
        return Scored(
            labels=np.concatenate(labels),
            doc_logits=np.concatenate(logits),
            groups=np.vstack(groups),
        )


def raid_contrasts(processed_dir: Path, manifests_dir: Path) -> dict[str, list[Contrast]]:
    """Per adversarial evaluation split: the sampled clean documents, then each attack on them."""
    parent_of: dict[str, tuple[str, str]] = {}
    labels: dict[str, int] = {}
    with (processed_dir / "raid-attacks.jsonl").open("rb") as fh:
        for line in fh:
            d = orjson.loads(line)
            parent_of[d["doc_id"]] = (f"raid:{d['meta']['adv_source_id']}", d["meta"]["attack"])
            labels[d["doc_id"]] = d["label"]
    out: dict[str, list[Contrast]] = {}
    for adv, parent in ADV_PARENT.items():
        if adv == "train-adv":
            continue
        docs = load_manifest(manifests_dir, adv)
        clean_m = sorted({parent_of[d][0] for d in docs if labels[d] == LABEL_MACHINE})
        clean_h = sorted({parent_of[d][0] for d in docs if labels[d] == LABEL_HUMAN})
        rows = [Contrast(name="clean", machine={parent: clean_m}, human={parent: clean_h})]
        for attack in ALL_ATTACKS:
            m = sorted(d for d in docs if parent_of[d][1] == attack and labels[d] == LABEL_MACHINE)
            h = sorted(d for d in docs if parent_of[d][1] == attack and labels[d] == LABEL_HUMAN)
            rows.append(Contrast(name=attack, machine={adv: m}, human={adv: h}))
        out[adv] = rows
    return out


def mage_contrasts(processed_dir: Path, manifests_dir: Path) -> tuple[list[Contrast], int]:
    """MAGE's GPT-4 documents, clean and paraphrased, and paraphrased human text, each against
    the unparaphrased human controls. Also returns the commentary rows excluded."""
    para = load_manifest(manifests_dir, "mage-para")
    xcorpus = load_manifest(manifests_dir, "mage-x")
    ids: dict[str, list[str]] = {"gpt4": [], "gpt4_para": [], "human_para": [], "human": []}
    commentary = 0
    with (processed_dir / "mage.jsonl").open("rb") as fh:
        for line in fh:
            d = orjson.loads(line)
            gen = d["generator"] or "human"
            if d["doc_id"] in xcorpus and gen == "gpt4":
                ids["gpt4"].append(d["doc_id"])
            elif d["doc_id"] in para:
                if not d["meta"].get("is_paraphrased"):
                    ids["human"].append(d["doc_id"])
                elif COMMENTARY.search(d["text"]):
                    commentary += 1
                else:
                    ids[f"{gen}_para"].append(d["doc_id"])
    human = {"mage-para": ids["human"]}
    return [
        Contrast(name="gpt4 (clean)", machine={"mage-x": ids["gpt4"]}, human=human),
        Contrast(name="gpt4 paraphrased", machine={"mage-para": ids["gpt4_para"]}, human=human),
        Contrast(name="human paraphrased", machine={"mage-para": ids["human_para"]}, human=human),
    ], commentary


def group_only_models(
    train: Bags, dev: Bags, sentence_val: Bags, names: Sequence[str]
) -> dict[str, tuple[MILModel, list[int]]]:
    """The kept model's shape fitted on each feature group alone."""
    out = {}
    for group, cols in _group_columns(names).items():
        sub = [b.model_copy(update={"x": b.x[:, cols]}) for b in (train, dev, sentence_val)]
        model, result = fit(KEPT, *sub)
        print(f"  {group}-only: select {result.selection:.4f}", flush=True)
        out[group] = (model, cols)
    return out


def group_only_auroc(
    scorer: Scorer, models: dict[str, tuple[MILModel, list[int]]], c: Contrast
) -> dict[str, float]:
    """Doc AUROC of each group-only model on one contrast."""
    out = {}
    for group, (model, cols) in models.items():
        y, s = [], []
        for side, label in ((c.machine, LABEL_MACHINE), (c.human, LABEL_HUMAN)):
            for stem, ids in side.items():
                bags = scorer.bags(stem, ids)
                logits = score(model, bags.model_copy(update={"x": bags.x[:, cols]})).doc_logits
                y.append(np.full(len(logits), label))
                s.append(logits)
        out[group] = float(roc_auc_score(np.concatenate(y), np.concatenate(s)))
    return out


def attack_shift(scorer: Scorer, contrasts: Sequence[Contrast]) -> dict[str, list[float]]:
    """Per attack, each feature's standardised mean difference from clean on human spans only,
    so a shift reflects the attack rather than a change in class balance."""
    clean = next(c for c in contrasts if c.name == "clean")
    x0 = np.vstack([scorer.bags(stem, ids).x for stem, ids in clean.human.items()])
    out = {}
    for c in contrasts:
        if c.name != "clean":
            x1 = np.vstack([scorer.bags(stem, ids).x for stem, ids in c.human.items()])
            out[c.name] = [smd(x1[:, j], x0[:, j]) for j in range(x0.shape[1])]
    return out


def _metric_dict(scorer: Scorer, contrast: Contrast) -> dict[str, float]:
    s = scorer.contrast(contrast)
    auroc, pauc, tpr = _metrics(s.labels, s.doc_logits)
    return {"auroc": auroc, "pauc_10pct": pauc, "tpr_at_1pct_fpr": tpr}


def report(
    runs: dict[str, Scorer],
    raid: dict[str, list[Contrast]],
    mage: list[Contrast],
    commentary: int,
    group_models: dict[str, tuple[MILModel, list[int]]],
    path: Path,
) -> Path:
    """Write the robustness report as JSON. Group analyses use the first run.

    Metrics are per run, with the threshold set within each contrast's own human documents.
    """
    names = list(runs)
    raid_out = {
        adv: {
            "parent": ADV_PARENT[adv],
            "attacks": [
                {
                    "name": c.name,
                    "trained_on": c.name in TRAIN_ATTACKS or c.name == "clean",
                    "n_human": sum(len(v) for v in c.human.values()),
                    "n_machine": sum(len(v) for v in c.machine.values()),
                    "metrics": {run: _metric_dict(s, c) for run, s in runs.items()},
                }
                for c in contrasts
            ],
        }
        for adv, contrasts in raid.items()
    }
    mage_out = {
        "n_human_controls": sum(len(v) for v in mage[0].human.values()),
        "commentary_excluded": commentary,
        "contrasts": [
            {
                "name": c.name,
                "n_machine": sum(len(v) for v in c.machine.values()),
                "metrics": {run: _metric_dict(s, c) for run, s in runs.items()},
            }
            for c in mage
        ],
    }

    first = next(iter(runs.values()))
    ood = raid.get("raid-ood-adv", [])
    by_name = {c.name: c for c in ood}
    raid_rows = [by_name[n] for n in ("clean", "paraphrase", "synonym") if n in by_name]
    contrib_mage = [first.contrast(c) for c in mage[:2]]
    contrib_raid = [first.contrast(c) for c in raid_rows]
    only_mage = [group_only_auroc(first, group_models, c) for c in mage[:2]]
    only_raid = [group_only_auroc(first, group_models, c) for c in raid_rows]
    groups_out = []
    for j, g in enumerate(FEATURE_GROUPS):
        cm = [float(roc_auc_score(s.labels, s.groups[:, j])) for s in contrib_mage]
        cr = [float(roc_auc_score(s.labels, s.groups[:, j])) for s in contrib_raid]
        om = [a[g] for a in only_mage]
        orr = [a[g] for a in only_raid]
        for measure, m, r in (("contribution", cm, cr), ("group-only", om, orr)):
            groups_out.append(
                {
                    "group": g,
                    "measure": measure,
                    "mage": dict(zip([c.name for c in mage[:2]], m, strict=True)),
                    "mage_drop": m[0] - m[1],
                    "raid_ood": dict(zip([c.name for c in raid_rows], r, strict=True)),
                }
            )

    shift = None
    if ood:
        shift = {
            attack: dict(zip(first.std.names, values, strict=True))
            for attack, values in attack_shift(first, ood).items()
        }

    payload = {
        "runs": names,
        "raid": raid_out,
        "mage_paraphrase": mage_out,
        "feature_groups": {"run": names[0], "rows": groups_out},
        "attack_shift": shift,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return path
