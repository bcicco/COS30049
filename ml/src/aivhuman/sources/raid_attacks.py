"""RAID adversarial variants matched back to their clean docs"""

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
# defang undoes these before segmenting, so eval only
DEFANGED_ATTACKS: Final = ("homoglyph", "whitespace", "zero_width_space")
ALL_ATTACKS: Final = TRAIN_ATTACKS + DEFANGED_ATTACKS

ADV_PARENT: Final = {"train-adv": "train", "dev-adv": "dev", "raid-ood-adv": "raid-ood"}
EVAL_PER_CLASS: Final = 1000  # clean docs per class, each gets every attack
SEED: Final = 20240503

# (clean row id, attack) -> (adv split, group_id)
# attacked docs keep the clean docs group so they never leak across splits
Wanted = dict[tuple[str, str], tuple[str, str]]


def _clean_docs(features: Path) -> list[tuple[str, int]]:
    # (doc_id, label) per doc, span_idx == 0 gives one row per doc
    t = pq.read_table(features, columns=["doc_id", "span_idx", "label"])
    t = t.filter(pc.equal(t["span_idx"], 0))
    return list(zip(t["doc_id"].to_pylist(), t["label"].to_pylist(), strict=True))


def _row_id(doc_id: str) -> str:
    # "raid:{id}" -> "{id}", matches adv_source_id in the attack partitions
    return doc_id.split(":", 1)[1]


def select(features_dir: Path, groups: dict[str, dict[str, str]], limit: int | None) -> Wanted:
    """Pick the attacked rows for each adv split. groups = clean split -> manifest"""
    # train: one attack per group so splices dont mix attacks
    # eval: same per class sample for every attack so theyre comparable
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
        attacks = ALL_ATTACKS[::3] if limit else ALL_ATTACKS  # fewer attacks for quick runs
        out.update({(_row_id(d), a): (adv, group_of[d]) for d in sample for a in attacks})
    return out


def load_rows(by_attack: Path, wanted: Wanted) -> Iterator[raid.RawRow]:
    """stream the wanted attacked rows from the hive partitioned parquet"""
    # filter is pushed into the scan so only matching rows get read
    ids = sorted({row_id for row_id, _ in wanted})
    dataset = ds.dataset(by_attack, format="parquet", partitioning="hive")
    expr = (ds.field("attack") != "none") & ds.field("adv_source_id").isin(ids)
    scanner = dataset.scanner(columns=list(raid.RawRow.model_fields), filter=expr)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            if (row["adv_source_id"], row["attack"]) in wanted:
                yield raid.RawRow(**row)


def to_doc(row: raid.RawRow, seg: Segmenter) -> Doc | None:
    """same as raid.to_doc but defangs the text first"""
    return raid.to_doc(row.model_copy(update={"generation": defang(row.generation)}), seg)
