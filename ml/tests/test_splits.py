"""Split disjointness — the assertion Phase 2 drops into.

`ml/PLAN.md` says this test will fail at some point during the project and that
catching it automatically is worth the twenty minutes. It is written now, while
nothing depends on it, so that Phase 2 has somewhere to land rather than an
excuse.

It skips when `ml/manifests/` is empty, so it runs in CI with no data present
and starts asserting the moment the first manifest is written.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from aivhuman.config import MANIFESTS_DIR
from aivhuman.schema import SPLIT_ROLES

#: Splits from the Phase 2 table in `ml/PLAN.md`. A manifest outside this set is
#: a typo, and a typo that silently creates a new split is worse than a failure.
KNOWN_SPLITS = frozenset(
    {
        "train",
        "dev",
        "raid-ood",
        "mage-x",
        "mage-para",
        "seqxgpt-calib",
        "seqxgpt-test",
    }
)


def manifests() -> list[Path]:
    return sorted(MANIFESTS_DIR.glob("*.json"))


def load(path: Path) -> dict[str, str]:
    """A manifest maps `doc_id` to `group_id`."""
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
    """Runs with no data: `split_role` is not a Phase 2 split name.

    The confusion this guards against is real -- `train_pool` is what an adapter
    stamps on a document, and `train` is what Phase 2 decides. Mixing them up
    would put every RAID document in the training split by construction.
    """
    assert "train" not in SPLIT_ROLES
    assert "train_pool" in SPLIT_ROLES
    assert not (KNOWN_SPLITS & SPLIT_ROLES)


@requires_manifests
def test_manifest_names_are_known_splits() -> None:
    for path in manifests():
        assert path.stem in KNOWN_SPLITS, f"unknown split manifest {path.name}"


@requires_manifests
def test_no_group_id_appears_in_two_splits() -> None:
    """The assertion the whole of Phase 2 exists to satisfy.

    A document and the human text it derives from share a `group_id`. If that
    group straddles two splits, the model is evaluated on text it trained on,
    and every number downstream is inflated in the direction nobody checks.
    """
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
    """SeqXGPT's calibration and test halves must split on the base document.

    The same human source appears across several generator variants, so
    splitting on the variant puts near-identical text either side and the
    calibration fits on the text it is later measured against.
    """
    calib = MANIFESTS_DIR / "seqxgpt-calib.json"
    test = MANIFESTS_DIR / "seqxgpt-test.json"
    if not (calib.exists() and test.exists()):
        pytest.skip("SeqXGPT splits not written yet")

    assert not (set(load(calib).values()) & set(load(test).values()))
