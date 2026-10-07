import numpy as np
import pytest

pytest.importorskip("sklearn")

from aivhuman.mil.infer import SHORT_TOKENS, ScoreResult, Thresholds, build_result

THRESHOLDS = Thresholds(
    sentence=0.8, sentence_recall=0.05, false_highlight_rate=0.1, document=0.6, document_tpr=0.2
)


def _result(probs: list[float], n_tokens: list[int], doc: float = 0.5) -> ScoreResult:
    spans = [(i * 10, i * 10 + 9) for i in range(len(probs))]
    return build_result(
        "x" * (10 * len(probs)),
        spans,
        n_tokens,
        np.array(probs),
        doc,
        [[("lm_logprob", 0.1)]] * len(probs),
        THRESHOLDS,
    )


def test_only_long_sentences_above_the_threshold_are_flagged() -> None:
    r = _result([0.9, 0.9, 0.5, 0.95], [20, SHORT_TOKENS - 1, 20, 30])
    assert [s.flagged for s in r.sentences] == [True, False, False, True]
    assert [s.too_short for s in r.sentences] == [False, True, False, False]


def test_document_flag_uses_the_document_threshold() -> None:
    assert _result([0.1] * 4, [20] * 4, doc=0.61).document_flagged
    assert not _result([0.1] * 4, [20] * 4, doc=0.59).document_flagged


def test_little_scorable_text_adds_a_caveat() -> None:
    short = _result([0.5, 0.5], [5, 20])
    long = _result([0.5] * 3, [20] * 3)
    assert any("little scorable text" in c for c in short.caveats)
    assert not any("little scorable text" in c for c in long.caveats)
