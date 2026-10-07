import json
from collections import Counter
from pathlib import Path

import pytest

from aivhuman.config import MANIFESTS_DIR
from aivhuman.schema import SPLIT_ROLES

# anything else is probably a typo
KNOWN_SPLITS = frozenset(
    {
        "train",
        "dev",
        "raid-ood",
        "mage-x",
        "mage-para",
        "seqxgpt-calib",
        "seqxgpt-test",
        "daigt",
    }
)


def manifests() -> list[Path]:
    return sorted(MANIFESTS_DIR.glob("*.json"))


def load(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "docs" in payload:
        docs = payload["docs"]
        assert isinstance(docs, dict)
        return {str(k): str(v) for k, v in docs.items()}
    assert isinstance(payload, dict)
    return {str(k): str(v) for k, v in payload.items()}


requires_manifests = pytest.mark.skipif(
    not MANIFESTS_DIR.exists() or not list(MANIFESTS_DIR.glob("*.json")),
    reason="no split manifests yet; Phase 2 writes them",
)


def test_split_roles_are_the_source_level_vocabulary() -> None:
    # train_pool is the adapter role, train is the actual split. dont mix them up
    assert "train" not in SPLIT_ROLES
    assert "train_pool" in SPLIT_ROLES
    assert not (KNOWN_SPLITS & SPLIT_ROLES)


@requires_manifests
def test_manifest_names_are_known_splits() -> None:
    for path in manifests():
        assert path.stem in KNOWN_SPLITS, f"unknown split manifest {path.name}"


@requires_manifests
def test_no_group_id_appears_in_two_splits() -> None:
    groups_by_split = {path.stem: set(load(path).values()) for path in manifests()}
    owners: Counter[str] = Counter()
    for groups in groups_by_split.values():
        owners.update(groups)

    shared = {group for group, count in owners.items() if count > 1}
    if shared:
        detail = {
            group: sorted(name for name, groups in groups_by_split.items() if group in groups)
            for group in sorted(shared)[:10]
        }
        pytest.fail(f"{len(shared)} group_ids span more than one split: {detail}")


@requires_manifests
def test_no_doc_id_appears_in_two_splits() -> None:
    seen: dict[str, str] = {}
    for path in manifests():
        for doc_id in load(path):
            if doc_id in seen:
                pytest.fail(f"{doc_id} is in both {seen[doc_id]} and {path.stem}")
            seen[doc_id] = path.stem


@requires_manifests
def test_seqxgpt_splits_are_disjoint_by_base_document() -> None:
    # same human source shows up under several generators, so split on the base doc
    calib = MANIFESTS_DIR / "seqxgpt-calib.json"
    test = MANIFESTS_DIR / "seqxgpt-test.json"
    if not (calib.exists() and test.exists()):
        pytest.skip("SeqXGPT splits not written yet")

    assert not (set(load(calib).values()) & set(load(test).values()))
