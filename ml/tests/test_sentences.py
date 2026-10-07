import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

from aivhuman.mil.calibrate import EDGES, Spans
from aivhuman.mil.sentences import doc_overlap, evaluate, pr_cell, threshold_at_fpr


def _spans(labels: list[int], docs: list[int], n_tokens: list[int] | None = None) -> Spans:
    n = len(labels)
    docs_arr = np.array(docs)
    idx = np.concatenate([np.arange((docs_arr == d).sum()) for d in dict.fromkeys(docs)])
    return Spans(
        name="test",
        labels=np.array(labels),
        logits=np.zeros(n),
        n_tokens=np.array(n_tokens or [20] * n),
        groups=docs_arr,
        doc_ids=docs_arr,
        span_idx=idx,
    )


def test_precision_recall_and_fpr_at_a_threshold() -> None:
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1])
    c = pr_cell(y, p, 0.5, "all")
    assert c.precision == pytest.approx(3 / 4)
    assert c.recall == pytest.approx(3 / 4)
    assert c.fpr == pytest.approx(1 / 4)
    assert c.precision_even == pytest.approx(0.75 / (0.75 + 0.25))
    assert c.f1 == pytest.approx(0.75)


def test_even_precision_ignores_the_base_rate() -> None:
    y = np.array([1] * 90 + [0] * 10)
    p = np.r_[np.full(45, 0.9), np.full(45, 0.1), np.full(5, 0.9), np.full(5, 0.1)]
    c = pr_cell(y, p, 0.5, "all")
    assert c.precision == pytest.approx(0.9)
    assert c.precision_even == pytest.approx(0.5)


def test_threshold_at_fpr_flags_at_most_the_target_share_of_humans() -> None:
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(1000, dtype=int), np.ones(1000, dtype=int)]
    probs = rng.random(2000)
    for fpr in (0.01, 0.05, 0.2):
        t = threshold_at_fpr(labels, probs, fpr)
        flagged = (probs[labels == 0] >= t).mean()
        assert flagged <= fpr
        assert flagged >= fpr - 0.002


def test_threshold_at_fpr_respects_ties() -> None:
    labels = np.zeros(100, dtype=int)
    probs = np.r_[np.full(10, 0.8), np.full(90, 0.2)]
    t = threshold_at_fpr(labels, probs, 0.05)
    assert (probs >= t).mean() == 0.0


def test_iou_is_one_for_a_perfect_match_and_zero_for_disjoint_flags() -> None:
    spans = _spans([0, 0, 1, 1], [0, 0, 0, 0])
    iou, regions, human = doc_overlap(spans, spans.labels == 1)
    assert iou[0] == pytest.approx(1.0) and regions[0] == 1 and not human[0]
    iou, _, _ = doc_overlap(spans, spans.labels == 0)
    assert iou[0] == pytest.approx(0.0)


def test_iou_weights_spans_by_tokens() -> None:
    spans = _spans([0, 1, 1], [0, 0, 0], n_tokens=[10, 30, 10])
    iou, _, _ = doc_overlap(spans, np.array([False, True, False]))
    assert iou[0] == pytest.approx(30 / 40)


def test_regions_count_separate_flagged_runs() -> None:
    spans = _spans([1, 1, 1, 1, 1], [0] * 5)
    _, regions, _ = doc_overlap(spans, np.array([True, False, True, True, False]))
    assert regions[0] == 2


def test_a_gap_left_by_a_dropped_span_splits_a_region() -> None:
    spans = _spans([1, 1], [0, 0]).model_copy(update={"span_idx": np.array([0, 2])})
    _, regions, _ = doc_overlap(spans, np.array([True, True]))
    assert regions[0] == 2


def test_false_highlight_rate_counts_human_docs_with_any_flag() -> None:
    spans = _spans([0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 1, 2, 2, 3, 3])
    probs = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    report = evaluate(spans, probs, {"p = 0.5": 0.5}, list(EDGES))
    point = report.points[0]
    assert point.human_docs == 3
    assert point.false_highlight_rate == pytest.approx(1 / 3)
    assert point.iou_docs == 1
    assert point.iou_mean == pytest.approx(1.0)


def test_cells_split_by_length_bucket() -> None:
    spans = _spans([0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2], n_tokens=[5, 5, 20, 20, 70, 70])
    probs = np.array([0.4, 0.6, 0.3, 0.7, 0.2, 0.8])
    point = evaluate(spans, probs, {"p = 0.5": 0.5}, list(EDGES)).points[0]
    assert [c.bucket for c in point.cells] == ["<15", "15-30", "60+", "all"]
    assert [c.spans for c in point.cells] == [2, 2, 2, 6]
    assert point.cells[-1].recall == pytest.approx(1.0)
