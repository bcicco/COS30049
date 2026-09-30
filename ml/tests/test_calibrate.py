import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

from aivhuman.mil.calibrate import (
    Calibrator,
    CellMetrics,
    Spans,
    bucket_names,
    bucket_of,
    ece,
    evaluate,
    shift_prior,
)


def _data(n: int, seed: int, share: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logits whose true even-prior probability is sigmoid(logit), across every bucket."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < share).astype(int)
    logits = rng.normal(0, 1.5, n) + np.where(labels == 1, 1.0, -1.0)
    n_tokens = rng.integers(0, 120, n).astype(float)
    return logits, n_tokens, labels


def test_bucket_edges_are_left_closed() -> None:
    assert bucket_of(np.array([0, 14, 15, 29, 30, 59, 60, 500])).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert bucket_names() == ["<15", "15-30", "30-60", "60+"]


def test_uninformative_logits_calibrate_to_even_odds_whatever_the_base_rate() -> None:
    logits, n_tokens, labels = _data(40_000, 0)
    rng = np.random.default_rng(0)
    mid = bucket_of(n_tokens) == 2
    labels[mid] = rng.random(mid.sum()) < 0.9
    logits[mid] = rng.normal(size=mid.sum())
    probs = Calibrator.fit(logits, n_tokens, labels).apply(logits[mid], n_tokens[mid])
    assert abs(float(np.median(probs)) - 0.5) < 0.05


def test_fit_is_monotone_and_well_calibrated_out_of_sample() -> None:
    cal = Calibrator.fit(*_data(40_000, 1))
    for y in cal.y:
        assert np.all(np.diff(y) >= 0)
    logits, n_tokens, labels = _data(40_000, 2)
    long = n_tokens >= 15
    assert ece(labels[long], cal.apply(logits[long], n_tokens[long])) < 0.03


def test_balanced_ece_ignores_the_base_rate() -> None:
    # Class-conditional N(+-1, 1.5): the even-prior probability is sigmoid(2x / 1.5**2).
    rng = np.random.default_rng(3)
    for share in (0.5, 0.8):
        labels = (rng.random(50_000) < share).astype(int)
        x = rng.normal(0, 1.5, 50_000) + np.where(labels == 1, 1.0, -1.0)
        probs = 1 / (1 + np.exp(-2 * x / 1.5**2))
        assert ece(labels, probs) < 0.01
        assert ece(labels, shift_prior(probs, share), balanced=False) < 0.01
    assert ece(labels, probs, balanced=False) > 0.1


def test_cap_clips_only_the_short_bucket_on_both_sides() -> None:
    cal = Calibrator.fit(*_data(20_000, 4))
    logits = np.array([-20.0, 20.0, -20.0, 20.0])
    n_tokens = np.array([5.0, 5.0, 40.0, 40.0])
    capped = cal.apply(logits, n_tokens)
    assert capped[:2].tolist() == [0.25, 0.75]
    assert capped[2] < 0.25 and capped[3] > 0.75
    raw = cal.apply(logits, n_tokens, cap=False)
    assert raw[0] < 0.25 and raw[1] > 0.75


def test_json_round_trip_gives_identical_scores(tmp_path) -> None:
    cal = Calibrator.fit(*_data(5_000, 5))
    cal.save(tmp_path / "calibrator.json")
    logits, n_tokens, _ = _data(1_000, 6)
    loaded = Calibrator.load(tmp_path / "calibrator.json")
    np.testing.assert_array_equal(cal.apply(logits, n_tokens), loaded.apply(logits, n_tokens))


def test_fit_needs_both_classes_in_every_bucket() -> None:
    logits, n_tokens, labels = _data(2_000, 7)
    labels[n_tokens >= 60] = 1
    with pytest.raises(ValueError, match="bucket 3"):
        Calibrator.fit(logits, n_tokens, labels)


def test_evaluate_reports_every_bucket_and_the_whole_split() -> None:
    cal = Calibrator.fit(*_data(20_000, 8))
    logits, n_tokens, labels = _data(8_000, 9)
    spans = Spans(
        name="test",
        labels=labels,
        logits=logits,
        n_tokens=n_tokens,
        styles=np.array(["natural", "moses_lower"] * 4_000),
        groups=np.arange(8_000) // 4,
        doc_ids=np.arange(8_000) // 4,
        span_idx=np.arange(8_000) % 4,
    )
    cells = evaluate(cal, spans)
    assert [c.bucket for c in cells] == [*bucket_names(), "all"]
    assert all(isinstance(c, CellMetrics) and c.ece_ci is not None for c in cells)
    styles = evaluate(cal, spans, by_style=True)
    assert {c.style for c in styles} == {"natural", "moses_lower"}
