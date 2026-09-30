"""RAID's adversarial variants, matched to the clean documents they were derived from."""

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from aivhuman.schema import LABEL_HUMAN, Doc
from aivhuman.sources import raid
from aivhuman.text.normalize import defang
from aivhuman.text.segment import Segmenter

TRAIN_ATTACKS: Final = (
    "alternative_spelling",
    "article_deletion",
    "insert_paragraphs",
    "number",
    "paraphrase",
    "perplexity_misspelling",
    "synonym",
    "upper_lower",
)
DEFANGED_ATTACKS: Final = ("homoglyph", "whitespace", "zero_width_space")
"""Undone by `defang` before segmentation, so evaluated but never trained on."""
ALL_ATTACKS: Final = TRAIN_ATTACKS + DEFANGED_ATTACKS

ADV_PARENT: Final = {"train-adv": "train", "dev-adv": "dev", "raid-ood-adv": "raid-ood"}
"""Adversarial split -> the clean split its documents are derived from."""
EVAL_PER_CLASS: Final = 1000
"""Clean documents per class sampled for evaluation, each paired with every attack."""
SEED: Final = 20240503

Wanted = dict[tuple[str, str], tuple[str, str]]
"""(clean RAID row id, attack) -> (adversarial split, group_id)."""


def _clean_docs(features: Path) -> list[tuple[str, int]]:
    """(doc_id, label) of every document in a clean feature file, in file order."""
    t = pq.read_table(features, columns=["doc_id", "span_idx", "label"])
    t = t.filter(pc.equal(t["span_idx"], 0))
    return list(zip(t["doc_id"].to_pylist(), t["label"].to_pylist(), strict=True))


def _row_id(doc_id: str) -> str:
    return doc_id.split(":", 1)[1]


def select(features_dir: Path, groups: dict[str, dict[str, str]], limit: int | None) -> Wanted:
    """Which attacked rows each adversarial split holds.

    Train pairs every clean train document with one training attack, shared across its group
    so splices never mix attacks. Evaluation splits pair a fixed per-class sample of clean
    documents with every attack, so each attack is compared on the same documents.
    `groups` maps each clean split to its manifest.
    """
    rng = random.Random(SEED)
    out: Wanted = {}
    for adv, parent in ADV_PARENT.items():
        docs = _clean_docs(features_dir / f"{parent}.parquet")
        group_of = groups[parent]
        if adv == "train-adv":
            attack = {g: rng.choice(TRAIN_ATTACKS) for g in sorted(set(group_of.values()))}
            chosen = [d for d, _ in docs]
            if limit:
                chosen = rng.sample(chosen, min(limit, len(chosen)))
            out.update({(_row_id(d), attack[group_of[d]]): (adv, group_of[d]) for d in chosen})
            continue
        per_class = min(limit // 2, EVAL_PER_CLASS) if limit else EVAL_PER_CLASS
        sample = []
        for is_human in (True, False):
            pool = sorted(d for d, y in docs if (y == LABEL_HUMAN) == is_human)
            sample += rng.sample(pool, min(per_class, len(pool)))
        attacks = ALL_ATTACKS[::3] if limit else ALL_ATTACKS
        out.update({(_row_id(d), a): (adv, group_of[d]) for d in sample for a in attacks})
    return out


def load_rows(by_attack: Path, wanted: Wanted) -> Iterator[raid.RawRow]:
    """The attacked rows named in `wanted`, read from the hive-partitioned parquet."""
    ids = sorted({row_id for row_id, _ in wanted})
    dataset = ds.dataset(by_attack, format="parquet", partitioning="hive")
    expr = (ds.field("attack") != "none") & ds.field("adv_source_id").isin(ids)
    scanner = dataset.scanner(columns=list(raid.RawRow.model_fields), filter=expr)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            if (row["adv_source_id"], row["attack"]) in wanted:
                yield raid.RawRow(**row)


def to_doc(row: raid.RawRow, seg: Segmenter) -> Doc | None:
    """`raid.to_doc` on the defanged generation. Pure, so it can run in a worker process."""
    return raid.to_doc(row.model_copy(update={"generation": defang(row.generation)}), seg)
