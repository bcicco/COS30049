"""Feature-group contributions add up to the sentence emissions they decompose."""

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from aivhuman.evaluate import COMMENTARY
from aivhuman.mil.data import Bags, Standardizer
from aivhuman.mil.model import MILConfig, MILModel
from aivhuman.mil.robustness import _score


def test_group_means_sum_to_the_mean_emission() -> None:
    names = ["lm_logprob", "lm_entropy", "lex_word_len", "pos_noun", "len_tokens"]
    rng = np.random.default_rng(0)
    sizes = [3, 5, 1]
    x = rng.normal(size=(sum(sizes), len(names))).astype(np.float32)
    bags = Bags(
        doc_ids=["a", "b", "c"],
        labels=np.array([0, 1, 1], dtype=np.float32),
        x=x,
        offsets=np.r_[0, np.cumsum(sizes)],
        span_labels=np.full(len(x), -1.0),
        straddles=np.zeros(len(x), dtype=bool),
    )
    std = Standardizer(names=names, mean=[0.0] * 5, std=[1.0] * 5)
    model = MILModel(len(names), MILConfig(head="gam"))
    torch.manual_seed(0)
    with torch.no_grad():
        model.head.linear.normal_()  # type: ignore[operator]
        model.head.hinge.normal_()  # type: ignore[operator]
    _, groups = _score(model, std, bags)
    assert groups.shape == (3, 4)
    emission = model.sentence_logits(torch.from_numpy(x)).detach().numpy() - model.bias
    per_doc = np.add.reduceat(emission, bags.offsets[:-1]) / np.array(sizes)
    np.testing.assert_allclose(groups.sum(1), per_doc, rtol=1e-5, atol=1e-5)


def test_commentary_catches_the_paraphraser_refusing() -> None:
    assert COMMENTARY.search("Sorry, I cannot paraphrase that sentence.")
    assert COMMENTARY.search("A rephrased statement for this could be")
    assert not COMMENTARY.search("The committee met on Tuesday to discuss the budget.")
