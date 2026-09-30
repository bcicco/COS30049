"""Adversarial RAID splits: selection, and manifests that stay inside their clean split."""

from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("sklearn")

from aivhuman.config import MANIFESTS_DIR
from aivhuman.evaluate import ATTACK_MANIFESTS
from aivhuman.sources.raid_attacks import ADV_PARENT, ALL_ATTACKS, TRAIN_ATTACKS, select

ATTACKS_DIR = MANIFESTS_DIR / ATTACK_MANIFESTS


def _features(path: Path, docs: list[tuple[str, int, int]]) -> None:
    """A feature file with `n_spans` rows per (doc_id, label, n_spans)."""
    rows = [(d, i, y) for d, y, n in docs for i in range(n)]
    doc_ids, span_idx, labels = zip(*rows, strict=True)
    pq.write_table(
        pa.table({"doc_id": list(doc_ids), "span_idx": list(span_idx), "label": list(labels)}),
        path,
    )


def test_select_shares_one_attack_per_train_group_and_pairs_every_eval_attack(
    tmp_path: Path,
) -> None:
    groups: dict[str, dict[str, str]] = {}
    for parent in ADV_PARENT.values():
        docs = [(f"raid:{parent}-{g}-{i}", int(i > 0), 3) for g in range(20) for i in range(3)]
        _features(tmp_path / f"{parent}.parquet", docs)
        groups[parent] = {d: f"raid:{parent}-g{d.split('-')[-2]}" for d, _, _ in docs}
    wanted = select(tmp_path, groups, limit=None)

    train = [(a, g) for (_, a), (adv, g) in wanted.items() if adv == "train-adv"]
    assert len(train) == 60
    by_group: dict[str, set[str]] = {}
    for attack, group in train:
        by_group.setdefault(group, set()).add(attack)
    assert all(len(v) == 1 and v <= set(TRAIN_ATTACKS) for v in by_group.values())

    for adv in ("dev-adv", "raid-ood-adv"):
        pairs = [(row, a) for (row, a), (split, _) in wanted.items() if split == adv]
        rows = {row for row, _ in pairs}
        assert len(pairs) == len(rows) * len(ALL_ATTACKS)


requires_attack_manifests = pytest.mark.skipif(
    not ATTACKS_DIR.exists() or not list(ATTACKS_DIR.glob("*.json")),
    reason="no adversarial manifests yet",
)


def _load(path: Path) -> dict[str, str]:
    payload: dict[str, str] = orjson.loads(path.read_bytes())
    return payload


@requires_attack_manifests
def test_adversarial_groups_stay_inside_their_clean_split() -> None:
    for adv, parent in ADV_PARENT.items():
        path = ATTACKS_DIR / f"{adv}.json"
        if not path.exists():
            continue
        clean = set(_load(MANIFESTS_DIR / f"{parent}.json").values())
        extra = set(_load(path).values()) - clean
        assert not extra, f"{adv} has {len(extra)} groups outside {parent}"


@requires_attack_manifests
def test_adversarial_doc_ids_are_disjoint_from_clean_manifests() -> None:
    clean: set[str] = set()
    for path in MANIFESTS_DIR.glob("*.json"):
        clean |= set(_load(path))
    for path in ATTACKS_DIR.glob("*.json"):
        assert not set(_load(path)) & clean, path.name
