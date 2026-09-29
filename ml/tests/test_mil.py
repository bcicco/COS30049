"""MIL pooling, masking, standardisation, the CRF and toy fits."""

import itertools

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from aivhuman.mil.data import Bags, Standardizer, batches, in_sentence_validation
from aivhuman.mil.model import (
    MILConfig,
    MILModel,
    coverage,
    crf_marginals,
    pool_lse,
    pool_topk,
)
from aivhuman.mil.train import fit, score


def _pad(rows: list[list[float]], width: int, fill: float) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.full((len(rows), width), fill)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, r in enumerate(rows):
        logits[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = True
    return logits, mask


@pytest.mark.parametrize(
    "pool",
    [
        lambda lg, m: pool_lse(lg, m, 1.0),
        lambda lg, m: pool_lse(lg, m, 5.0),
        lambda lg, m: pool_topk(lg, m, 3),
        coverage,
    ],
)
def test_pooling_ignores_padding(pool: object) -> None:
    rows = [[0.5, -1.0], [2.0, 0.1, -3.0, 1.0], [-0.2]]
    narrow = pool(*_pad(rows, 4, 0.0))  # type: ignore[operator]
    wide = pool(*_pad(rows, 9, 50.0))  # type: ignore[operator]
    torch.testing.assert_close(narrow, wide)


def test_lse_is_invariant_to_bag_size() -> None:
    out = pool_lse(*_pad([[1.5] * 2, [1.5] * 40], 40, 0.0), tau=2.0)
    torch.testing.assert_close(out, torch.tensor([1.5, 1.5]))


def test_topk_on_a_bag_shorter_than_k() -> None:
    out = pool_topk(*_pad([[1.0, 3.0], [1.0, 2.0, 3.0, 4.0]], 4, 0.0), k=3)
    torch.testing.assert_close(out, torch.tensor([2.0, 3.0]))


def test_standardizer_imputes_nan_to_the_mean() -> None:
    x = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    std = Standardizer.fit(x, ["a", "b"])
    z = std.transform(x)
    assert z[0, 1] == 0.0
    outlier = std.transform(np.array([[1e6, 0.0]]))
    assert outlier[0, 0] == 5.0
    np.testing.assert_allclose(z[:, 0], [-1.2247449, 0.0, 1.2247449], rtol=1e-5)


def _bags(
    sizes: list[int], labels: list[int], x: np.ndarray, span_labels: np.ndarray | None = None
) -> Bags:
    return Bags(
        doc_ids=[f"d{i}" for i in range(len(sizes))],
        labels=np.array(labels, dtype=np.float32),
        x=x,
        offsets=np.r_[0, np.cumsum(sizes)],
        span_labels=np.full(len(x), -1.0) if span_labels is None else span_labels,
        straddles=np.zeros(len(x), dtype=bool),
    )


def test_scores_do_not_depend_on_batch_size() -> None:
    rng = np.random.default_rng(0)
    sizes = list(rng.integers(1, 30, size=50))
    bags = _bags(sizes, [0, 1] * 25, rng.normal(size=(sum(sizes), 4)).astype(np.float32))
    model = MILModel(4, MILConfig())
    a, b = score(model, bags, batch_size=3), score(model, bags, batch_size=64)
    np.testing.assert_allclose(a.doc_logits, b.doc_logits, atol=1e-5)
    np.testing.assert_allclose(a.sentence_logits, b.sentence_logits, atol=1e-5)


def test_batches_cover_every_bag_once() -> None:
    sizes = [3, 1, 7, 2, 5]
    bags = _bags(sizes, [0, 1, 0, 1, 0], np.zeros((sum(sizes), 2), dtype=np.float32))
    seen = [i for idx, _, mask in batches(bags, range(5), 2) for i in idx]
    assert sorted(seen) == list(range(5))


def test_fit_finds_the_planted_machine_sentence() -> None:
    # Machine documents contain one sentence with a high value of feature 0; the head must
    # learn a positive weight on it from document labels alone.
    rng = np.random.default_rng(1)
    sizes, labels, rows, spans = [], [], [], []
    for i in range(600):
        n = int(rng.integers(3, 12))
        x = rng.normal(size=(n, 3))
        s = np.zeros(n)
        if i % 2:
            j = rng.integers(n)
            x[j, 0] += 4.0
            s[j] = 1
        sizes.append(n)
        labels.append(i % 2)
        rows.append(x)
        spans.append(s)
    bags = _bags(sizes, labels, np.vstack(rows).astype(np.float32), np.concatenate(spans))
    model, result = fit(MILConfig(pooling="topk", k=1, lr=0.05, epochs=15), bags, bags, bags)
    w = model.slopes().numpy()
    assert w[0] > 0.5
    assert abs(w[0]) > 3 * max(abs(w[1]), abs(w[2]))
    assert result.dev_tpr > 0.5
    assert result.sentence_auroc > 0.9


def _gam(n_features: int = 3, **kw: object) -> MILModel:
    model = MILModel(n_features, MILConfig(head="gam", **kw))  # type: ignore[arg-type]
    torch.manual_seed(0)
    model.set_knots(torch.randn(500, n_features))
    with torch.no_grad():
        model.head.linear.normal_()  # type: ignore[operator]
        model.head.hinge.normal_()  # type: ignore[operator]
    return model


def test_gam_contributions_sum_to_the_logit() -> None:
    model = _gam()
    x = torch.randn(2, 7, 3)
    logits = model.sentence_logits(x)
    torch.testing.assert_close(model.contributions(x).sum(-1) + model.bias, logits)
    torch.testing.assert_close(model.contributions(torch.zeros(1, 3)), torch.zeros(1, 3))


def test_gam_pooling_ignores_padding() -> None:
    rng = np.random.default_rng(3)
    sizes = list(rng.integers(1, 20, size=30))
    bags = _bags(sizes, [0, 1] * 15, rng.normal(size=(sum(sizes), 3)).astype(np.float32))
    model = _gam()
    a, b = score(model, bags, batch_size=2), score(model, bags, batch_size=64)
    np.testing.assert_allclose(a.doc_logits, b.doc_logits, atol=1e-5)


def test_gam_with_zero_hinges_is_linear() -> None:
    model = _gam()
    with torch.no_grad():
        model.head.hinge.zero_()  # type: ignore[operator]
    x = torch.randn(5, 3)
    torch.testing.assert_close(model.contributions(x), x * model.head.linear)  # type: ignore[operator]
    torch.testing.assert_close(model.slopes(), model.head.linear.detach())  # type: ignore[union-attr]


def test_gam_finds_a_u_shaped_signal_that_linear_cannot() -> None:
    # One machine sentence per machine doc sits at an extreme of feature 0, in either direction.
    rng = np.random.default_rng(4)
    sizes, labels, rows, spans = [], [], [], []
    for i in range(800):
        n = int(rng.integers(3, 10))
        x = rng.normal(size=(n, 2)) * 0.5
        s = np.zeros(n)
        if i % 2:
            j = rng.integers(n)
            x[j, 0] = rng.choice([-3.0, 3.0])
            s[j] = 1
        sizes.append(n)
        labels.append(i % 2)
        rows.append(x)
        spans.append(s)
    bags = _bags(sizes, labels, np.vstack(rows).astype(np.float32), np.concatenate(spans))
    common = {"pooling": "topk", "k": 1, "lr": 0.05, "epochs": 15, "l1": 0.0}
    _, linear = fit(MILConfig(head="linear", **common), bags, bags, bags)  # type: ignore[arg-type]
    gam, spline = fit(MILConfig(head="gam", **common), bags, bags, bags)  # type: ignore[arg-type]
    assert spline.dev_tpr > linear.dev_tpr + 0.2
    ends = gam.contributions(torch.tensor([[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]]))[:, 0]
    assert ends[0] > ends[1] and ends[2] > ends[1]


def test_subset_keeps_whole_bags_in_order() -> None:
    sizes = [2, 3, 1]
    x = np.arange(6, dtype=np.float32)[:, None]
    sub = _bags(sizes, [0, 1, 0], x).subset(np.array([True, False, True]))
    assert sub.doc_ids == ["d0", "d2"]
    np.testing.assert_array_equal(sub.x.ravel(), [0, 1, 5])
    np.testing.assert_array_equal(sub.offsets, [0, 2, 3])


def test_sentence_validation_slice_follows_the_group() -> None:
    groups = {f"doc{i}": f"g{i // 3}" for i in range(3000)}
    in_val = in_sentence_validation(list(groups), groups)
    assert 0.15 < in_val.mean() < 0.25
    by_group: dict[str, set[bool]] = {}
    for doc, flag in zip(groups, in_val, strict=True):
        by_group.setdefault(groups[doc], set()).add(bool(flag))
    assert all(len(v) == 1 for v in by_group.values())


def test_sentence_loss_uses_only_known_labels() -> None:
    from aivhuman.mil.train import _sentence_loss

    x = np.zeros((5, 1), dtype=np.float32)
    unlabelled = _bags([2, 3], [0, 1], x)
    logits = torch.zeros(2, 3, requires_grad=True)
    assert _sentence_loss(unlabelled, [0, 1], logits).item() == 0.0

    spans = np.array([np.nan, np.nan, 0.0, 1.0, 1.0])
    labelled = _bags([2, 3], [0, 1], x, spans)
    loss = _sentence_loss(labelled, [0, 1], torch.zeros(2, 3))
    assert loss.item() == pytest.approx(np.log(2))
    # A confident, correct logit on the known spans lowers it; padding and NaN spans never count.
    good = torch.tensor([[9.0, 9.0, 9.0], [-9.0, 9.0, 9.0]])
    assert _sentence_loss(labelled, [0, 1], good).item() < 1e-3


def _brute_marginals(e: list[float], trans: torch.Tensor, start: torch.Tensor) -> list[float]:
    """Machine-state log-odds per position by enumerating every state path."""
    n = len(e)
    weight = {1: [0.0] * n, 0: [0.0] * n}
    for path in itertools.product((0, 1), repeat=n):
        score = start[path[0]].item() + sum(e[t] * path[t] for t in range(n))
        score += sum(trans[path[t - 1], path[t]].item() for t in range(1, n))
        for t in range(n):
            weight[path[t]][t] += float(np.exp(score))
    return [float(np.log(weight[1][t] / weight[0][t])) for t in range(n)]


def test_crf_with_zero_potentials_returns_the_emissions() -> None:
    e, mask = _pad([[0.5, -1.0, 2.0], [0.3]], 3, 0.0)
    out = crf_marginals(e, mask, torch.zeros(2, 2), torch.zeros(2))
    torch.testing.assert_close(out[mask], e[mask])


def test_crf_marginals_match_brute_force() -> None:
    torch.manual_seed(0)
    trans, start = torch.randn(2, 2), torch.randn(2)
    rows = [[0.5, -1.0, 2.0, 0.1, -0.4, 1.2], [0.3, -2.0], [1.0]]
    e, mask = _pad(rows, 6, 0.0)
    out = crf_marginals(e, mask, trans, start)
    for i, r in enumerate(rows):
        np.testing.assert_allclose(out[i, : len(r)], _brute_marginals(r, trans, start), rtol=1e-4)


def test_crf_ignores_padding() -> None:
    torch.manual_seed(1)
    trans, start = torch.randn(2, 2), torch.randn(2)
    rows = [[0.5, -1.0], [2.0, 0.1, -3.0, 1.0], [-0.2]]
    narrow, m1 = _pad(rows, 4, 0.0)
    wide, m2 = _pad(rows, 9, 50.0)
    a = crf_marginals(narrow, m1, trans, start)
    b = crf_marginals(wide, m2, trans, start)
    torch.testing.assert_close(a[m1], b[m2])


def test_sticky_crf_pools_a_run_of_weak_evidence() -> None:
    sticky = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
    run, mask = _pad([[-1.0, -1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1.0, -1.0]], 10, 0.0)
    scattered, _ = _pad([[0.5, -1.0, 0.5, -1.0, 0.5, -1.0, 0.5, 0.5, -1.0, 0.5]], 10, 0.0)
    in_run = crf_marginals(run, mask, sticky, torch.zeros(2))[0, 2:8]
    alone = crf_marginals(scattered, mask, sticky, torch.zeros(2))[0][scattered[0] > 0]
    # Six sentences at 0.5 each end up above 0.5; the same sentences scattered fall towards 0.
    assert in_run.mean() > 0.6
    assert in_run.mean() > alone.mean() + 0.5


def test_crf_scores_do_not_depend_on_batch_size() -> None:
    rng = np.random.default_rng(5)
    sizes = list(rng.integers(1, 25, size=40))
    bags = _bags(sizes, [0, 1] * 20, rng.normal(size=(sum(sizes), 3)).astype(np.float32))
    model = _gam(crf=True)
    with torch.no_grad():
        model.stickiness.fill_(1.5)
    a, b = score(model, bags, batch_size=2), score(model, bags, batch_size=64)
    np.testing.assert_allclose(a.doc_logits, b.doc_logits, atol=1e-5)
    np.testing.assert_allclose(a.sentence_logits, b.sentence_logits, atol=1e-5)


def test_crf_learns_sticky_transitions_from_document_labels() -> None:
    # Machine documents hold one contiguous run of weakly shifted sentences.
    rng = np.random.default_rng(6)
    sizes, labels, rows, spans = [], [], [], []
    for i in range(600):
        n = int(rng.integers(8, 16))
        x = rng.normal(size=(n, 2))
        s = np.zeros(n)
        if i % 2:
            lo = int(rng.integers(0, n - 5))
            x[lo : lo + 5, 0] += 1.0
            s[lo : lo + 5] = 1
        sizes.append(n)
        labels.append(i % 2)
        rows.append(x)
        spans.append(s)
    bags = _bags(sizes, labels, np.vstack(rows).astype(np.float32), np.concatenate(spans))
    common = {"lr": 0.05, "epochs": 15, "l1": 0.0, "tau": 2.0}
    _, plain = fit(MILConfig(**common), bags, bags, bags)  # type: ignore[arg-type]
    model, crf = fit(MILConfig(crf=True, **common), bags, bags, bags)  # type: ignore[arg-type]
    assert model.stickiness.item() > 0
    assert crf.sentence_auroc > plain.sentence_auroc


def test_symmetric_crf_is_invariant_to_reversing_the_document() -> None:
    model = MILModel(2, MILConfig(crf=True))
    with torch.no_grad():
        model.stickiness.fill_(1.2)
    x = torch.randn(1, 7, 2)
    mask = torch.ones(1, 7, dtype=torch.bool)
    forward = model(x, mask)[2]
    backward = model(x.flip(1), mask)[2]
    torch.testing.assert_close(forward, backward.flip(1))
