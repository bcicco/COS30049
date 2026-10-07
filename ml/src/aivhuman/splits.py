"""grouped split manifests over the normalised corpora"""

# assignment only depends on a stable hash of the group id (no rng), so manifests are
# reproducible from the jsonl alone and row order doesnt matter

from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Final

import orjson
from pydantic import BaseModel, ConfigDict, Field

from aivhuman.schema import LABEL_HUMAN
from aivhuman.text.normalize import stable_hash, text_key

HELD_OUT_GENERATORS: Final = frozenset({"cohere", "cohere-chat"})
HELD_OUT_DOMAINS: Final = frozenset({"reviews", "wiki"})

OOD_FRACTION: Final = 0.10  # seen-domain RAID groups that go to raid-ood
DEV_FRACTION: Final = 0.10  # of whats left after ood

CALIB_FRACTION: Final = 0.50


class Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    group_id: str
    label: int
    domain: str | None
    generator: str | None
    split_role: str
    key: str  # hash of text_key(text), for exact dupes


class SplitError(ValueError):
    pass


class SplitStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docs: dict[str, int] = Field(default_factory=dict)
    groups: dict[str, int] = Field(default_factory=dict)
    human: dict[str, int] = Field(default_factory=dict)
    machine: dict[str, int] = Field(default_factory=dict)
    domains: dict[str, dict[str, int]] = Field(default_factory=dict)
    generators: dict[str, dict[str, int]] = Field(default_factory=dict)
    dropped: dict[str, int] = Field(default_factory=dict)
    merged_groups: dict[str, int] = Field(default_factory=dict)  # merged bc of exact same text


def read_rows(path: Path) -> Iterator[Row]:
    with path.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = orjson.loads(line)
            yield Row(
                doc_id=d["doc_id"],
                group_id=d["group_id"],
                label=d["label"],
                domain=d["domain"],
                generator=d["generator"],
                split_role=d["split_role"],
                key=stable_hash(text_key(d["text"])),
            )


def merge_groups(rows: Iterable[Row]) -> dict[str, str]:
    """group_id -> unit root, groups sharing a text get merged"""
    # root = smallest group_id in the component so row order doesnt matter

    parent: dict[str, str] = {}

    def find(g: str) -> str:
        parent.setdefault(g, g)
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    first_group: dict[str, str] = {}
    for row in rows:
        find(row.group_id)
        other = first_group.setdefault(row.key, row.group_id)
        a, b = find(other), find(row.group_id)
        if a != b:
            lo, hi = sorted((a, b))
            parent[hi] = lo
    return {g: find(g) for g in parent}


def unit_fraction(root: str) -> float:
    return int(stable_hash(root, 16), 16) / 16**16


def split_raid(rows: list[Row], stats: SplitStats) -> dict[str, list[Row]]:
    units = merge_groups(rows)
    stats.merged_groups["raid"] = sum(g != root for g, root in units.items())

    ood_units = {units[r.group_id] for r in rows if r.domain in HELD_OUT_DOMAINS}
    out: dict[str, list[Row]] = {"train": [], "dev": [], "raid-ood": []}
    dropped = 0
    for row in rows:
        unit = units[row.group_id]
        u = unit_fraction(unit)
        if unit in ood_units or u < OOD_FRACTION:
            out["raid-ood"].append(row)
        elif row.generator in HELD_OUT_GENERATORS:
            dropped += 1
        elif u < OOD_FRACTION + DEV_FRACTION * (1 - OOD_FRACTION):
            out["dev"].append(row)
        else:
            out["train"].append(row)
    stats.dropped["raid_held_out_generator"] = dropped
    return out


def split_mage(rows: list[Row], foreign_keys: set[str], stats: SplitStats) -> dict[str, list[Row]]:
    # drops texts that also show up in another corpus
    para_groups = {r.group_id for r in rows if r.split_role == "xcorpus_para_test"}
    out: dict[str, list[Row]] = {"mage-x": [], "mage-para": []}
    foreign = para_overlap = 0
    for row in rows:
        if row.key in foreign_keys:
            foreign += 1
        elif row.split_role == "xcorpus_para_test":
            out["mage-para"].append(row)
        elif row.group_id in para_groups:
            para_overlap += 1
        else:
            out["mage-x"].append(row)
    stats.dropped["mage_shared_with_other_corpus"] = foreign
    stats.dropped["mage_x_shared_with_para"] = para_overlap
    return out


def split_daigt(rows: list[Row], foreign_keys: set[str], stats: SplitStats) -> dict[str, list[Row]]:
    kept = [r for r in rows if r.key not in foreign_keys]
    stats.dropped["daigt_shared_with_other_corpus"] = len(rows) - len(kept)
    return {"daigt": kept}


def split_seqxgpt(rows: list[Row], stats: SplitStats) -> dict[str, list[Row]]:
    # 50/50 by base doc
    units = merge_groups(rows)
    stats.merged_groups["seqxgpt"] = sum(g != root for g, root in units.items())
    out: dict[str, list[Row]] = {"seqxgpt-calib": [], "seqxgpt-test": []}
    for row in rows:
        name = (
            "seqxgpt-calib"
            if unit_fraction(units[row.group_id]) < CALIB_FRACTION
            else "seqxgpt-test"
        )
        out[name].append(row)
    return out


def check_disjoint(splits: dict[str, list[Row]]) -> None:
    owner: dict[str, str] = {}
    seen_docs: set[str] = set()
    for name, rows in splits.items():
        for row in rows:
            if row.doc_id in seen_docs:
                raise SplitError(f"{row.doc_id} appears in two splits")
            seen_docs.add(row.doc_id)
            prev = owner.setdefault(row.group_id, name)
            if prev != name:
                raise SplitError(f"group {row.group_id} is in both {prev} and {name}")


def assign(
    raid: list[Row], mage: list[Row], seqxgpt: list[Row], daigt: list[Row] | None = None
) -> tuple[dict[str, list[Row]], SplitStats]:
    stats = SplitStats()
    daigt = daigt or []
    foreign = {r.key for r in raid} | {r.key for r in seqxgpt}
    splits = (
        split_raid(raid, stats)
        | split_mage(mage, foreign | {r.key for r in daigt}, stats)
        | split_seqxgpt(seqxgpt, stats)
        | split_daigt(daigt, foreign | {r.key for r in mage}, stats)
    )
    check_disjoint(splits)

    for row in splits["train"] + splits["dev"]:
        if row.domain in HELD_OUT_DOMAINS or row.generator in HELD_OUT_GENERATORS:
            raise SplitError(f"{row.doc_id} is held out but in a training split")

    for name, rows in splits.items():
        stats.docs[name] = len(rows)
        stats.groups[name] = len({r.group_id for r in rows})
        stats.human[name] = sum(r.label == LABEL_HUMAN for r in rows)
        stats.machine[name] = len(rows) - stats.human[name]
        stats.domains[name] = dict(sorted(Counter(str(r.domain) for r in rows).items()))
        stats.generators[name] = dict(sorted(Counter(r.generator or "human" for r in rows).items()))
    return splits, stats


def write_manifests(splits: dict[str, list[Row]], directory: Path) -> None:
    # {split}.json: doc_id -> group_id
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        payload = {r.doc_id: r.group_id for r in rows}
        (directory / f"{name}.json").write_bytes(
            orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)
        )


def build(processed_dir: Path, manifests_dir: Path, report_path: Path) -> SplitStats:
    raid, mage, seqxgpt, daigt = (
        list(read_rows(processed_dir / f"{source}.jsonl"))
        for source in ("raid", "mage", "seqxgpt", "daigt")
    )
    splits, stats = assign(raid, mage, seqxgpt, daigt)
    write_manifests(splits, manifests_dir)

    report: dict[str, Any] = {
        "held_out_generators": sorted(HELD_OUT_GENERATORS),
        "held_out_domains": sorted(HELD_OUT_DOMAINS),
        "ood_fraction": OOD_FRACTION,
        "dev_fraction": DEV_FRACTION,
        "calib_fraction": CALIB_FRACTION,
        **stats.model_dump(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    return stats
