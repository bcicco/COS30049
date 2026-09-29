"""Group sampling and length batching for the encoder baseline."""

import random

import pytest

pytest.importorskip("torch")

from aivhuman.baselines.encoder import _batches
from aivhuman.evaluate import EvalDoc, sample_per_group


def _doc(i: int, label: int, group: str) -> EvalDoc:
    return EvalDoc(
        doc_id=f"raid:{i}", text="x", label=label, group_id=group, domain=None, breakdown="x"
    )


def test_sample_keeps_every_human_and_caps_machine_per_group() -> None:
    docs = [_doc(0, 0, "g1"), _doc(1, 0, "g2")]
    docs += [_doc(10 + i, 1, "g1") for i in range(10)]
    docs += [_doc(30 + i, 1, "g2") for i in range(2)]
    keep = sample_per_group(docs, 3, random.Random(0))
    assert {0, 1} <= set(keep)
    picked = [docs[i] for i in keep if docs[i].label == 1]
    assert sum(d.group_id == "g1" for d in picked) == 3
    assert sum(d.group_id == "g2" for d in picked) == 2
    assert len(set(keep)) == len(keep)


def test_batches_cover_every_index_once() -> None:
    lengths = [random.Random(i).randint(1, 500) for i in range(1003)]
    order = list(range(1003))
    batches = _batches(order, lengths, 16)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == order
    assert all(len(b) <= 16 for b in batches)
