"""Evaluation harness metrics and prediction round-trip."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from aivhuman.evaluate import (
    EvalDoc,
    bootstrap_ci,
    compute_metrics,
    drop_commentary,
    partial_auroc,
    read_predictions,
    tpr_at_fpr,
    write_predictions,
)


def _doc(i: int, label: int, breakdown: str) -> EvalDoc:
    return EvalDoc(
        doc_id=f"raid:{i}",
        text="x",
        label=label,
        group_id=f"raid:g{i}",
        domain="news",
        breakdown=breakdown,
    )


def test_tpr_at_fpr_respects_the_fpr_budget() -> None:
    # 100 negatives at 0..0.99, positives at 0.995 and 0.5: at 1% FPR the threshold must sit
    # above the highest negative, so only the first positive is caught.
    labels = np.array([0] * 100 + [1, 1])
    scores = np.concatenate([np.arange(100) / 100, [0.995, 0.5]])
    tpr, threshold = tpr_at_fpr(labels, scores, 0.01)
    assert tpr == 0.5
    assert threshold > 0.5


def test_tpr_at_fpr_is_one_when_separable() -> None:
    labels = np.array([0, 0, 1, 1])
    tpr, _ = tpr_at_fpr(labels, np.array([0.1, 0.2, 0.8, 0.9]), 0.001)
    assert tpr == 1.0


def test_compute_metrics_breaks_down_by_generator() -> None:
    docs = [_doc(i, 0, "human") for i in range(200)]
    docs += [_doc(1000 + i, 1, "gpt2") for i in range(50)]
    docs += [_doc(2000 + i, 1, "gpt4") for i in range(50)]
    preds = {d.doc_id: 0.1 for d in docs if d.breakdown == "human"}
    preds |= {d.doc_id: 0.9 for d in docs if d.breakdown == "gpt2"}
    preds |= {d.doc_id: 0.05 for d in docs if d.breakdown == "gpt4"}

    m = compute_metrics("toy", "dev", docs, preds)
    assert (m.n_human, m.n_machine) == (200, 100)
    assert m.by_generator == {"gpt2": 1.0, "gpt4": 0.0, "human": 0.0}
    assert m.by_domain == {"news/human": 0.0, "news/machine": 0.5}
    assert m.tpr_at_1pct_fpr == 0.5
    assert m.balanced_acc == pytest.approx(0.75)


def test_compute_metrics_rejects_missing_predictions() -> None:
    docs = [_doc(0, 0, "human"), _doc(1, 1, "gpt2")]
    with pytest.raises(ValueError, match="without a prediction"):
        compute_metrics("toy", "dev", docs, {"raid:0": 0.1})


def test_predictions_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "m" / "dev.parquet"
    write_predictions(path, ["a", "b"], np.array([0.25, 0.75]))
    assert read_predictions(path) == {"a": 0.25, "b": 0.75}


def test_partial_auroc_is_one_when_separable_and_half_at_chance() -> None:
    labels = np.array([0] * 500 + [1] * 500)
    assert partial_auroc(labels, np.r_[np.zeros(500), np.ones(500)]) == 1.0
    noise = np.random.default_rng(0).random(1000)
    assert partial_auroc(labels, noise) == pytest.approx(0.5, abs=0.05)


def test_bootstrap_ci_brackets_the_estimate_and_resamples_groups() -> None:
    rng = np.random.default_rng(1)
    labels = np.array([0] * 400 + [1] * 400)
    scores = labels + rng.normal(scale=1.0, size=800)
    groups = np.arange(800) // 2
    ((lo, hi),) = bootstrap_ci(labels, scores, groups, [partial_auroc], n=100)
    assert lo < partial_auroc(labels, scores) < hi
    # With identical scores inside each two-document group, whole-group resampling keeps
    # every group's pair together, so doubling each group changes nothing.
    doubled = bootstrap_ci(
        np.repeat(labels, 2), np.repeat(scores, 2), np.repeat(groups, 2), [partial_auroc], n=100
    )
    assert doubled == [(lo, hi)]


def test_compute_metrics_reports_intervals() -> None:
    docs = [_doc(i, 0, "human") for i in range(300)] + [
        _doc(1000 + i, 1, "gpt2") for i in range(300)
    ]
    rng = np.random.default_rng(2)
    preds = {d.doc_id: float(d.label + rng.normal()) for d in docs}
    m = compute_metrics("toy", "dev", docs, preds)
    assert m.tpr_at_1pct_ci[0] <= m.tpr_at_1pct_fpr <= m.tpr_at_1pct_ci[1]
    assert m.pauc_ci[0] <= m.pauc_10pct <= m.pauc_ci[1]
    assert m.tpr_at_01pct_ci is not None
    assert m.tpr_at_01pct_ci[0] <= m.tpr_at_01pct_fpr <= m.tpr_at_01pct_ci[1]


def test_compute_metrics_breaks_down_by_generator_at_01pct() -> None:
    docs = [_doc(i, 0, "human") for i in range(2000)]
    docs += [_doc(10_000 + i, 1, "gpt2") for i in range(50)]
    docs += [_doc(20_000 + i, 1, "gpt4") for i in range(50)]
    # Humans spread over [0, 0.5); gpt4 sits between the 1% and 0.1% thresholds.
    preds = {d.doc_id: i / 4000 for i, d in enumerate(docs[:2000])}
    preds |= {d.doc_id: 0.9 for d in docs if d.breakdown == "gpt2"}
    preds |= {d.doc_id: 0.497 for d in docs if d.breakdown == "gpt4"}
    m = compute_metrics("toy", "dev", docs, preds)
    assert m.by_generator["gpt4"] == 1.0
    assert m.by_generator_01pct is not None
    assert m.by_generator_01pct["gpt2"] == 1.0
    assert m.by_generator_01pct["gpt4"] == 0.0
    assert m.by_generator_01pct["human"] <= 0.001
    assert m.threshold_01pct is not None and m.threshold_01pct > m.threshold_1pct


def test_drop_commentary_removes_only_paraphrased_commentary() -> None:
    docs = [
        _doc(0, 1, "gpt4_para").model_copy(update={"text": "Sure, here is a paraphrase:"}),
        _doc(1, 1, "gpt4_para").model_copy(update={"text": "The cat sat on the mat."}),
        _doc(2, 0, "human").model_copy(update={"text": "We paraphrase Smith (2001) here."}),
    ]
    kept, dropped = drop_commentary(docs)
    assert dropped == 1
    assert [d.doc_id for d in kept] == ["raid:1", "raid:2"]
