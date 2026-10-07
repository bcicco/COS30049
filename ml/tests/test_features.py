import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("spacy")

import pyarrow.parquet as pq
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aivhuman.features import FEATURE_NAMES, POS_TAGS, extract, lexical, lm, syntax
from aivhuman.features.load import SpanDoc

VOCAB = 50


class WhitespaceTokenizer:
    bos_token_id = 0

    def __call__(self, texts: list[str], **_: Any) -> dict[str, Any]:
        ids, offsets = [], []
        for text in texts:
            spans = [m.span() for m in re.finditer(r"\S+", text)]
            offsets.append(spans)
            ids.append([1 + sum(map(ord, text[s:e])) % (VOCAB - 1) for s, e in spans])
        return {"input_ids": ids, "offset_mapping": offsets}


@pytest.fixture(scope="module")
def tiny_lm() -> lm.ReferenceLM:
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=16, n_layer=1, n_head=2)
    return lm.ReferenceLM(
        torch.device("cpu"),
        token_budget=64,
        model=GPT2LMHeadModel(config),
        tokenizer=WhitespaceTokenizer(),
        max_context=16,
        carried=8,
    )


@pytest.fixture(scope="module")
def nlp() -> Any:
    return syntax.load()


@pytest.mark.parametrize("n", [0, 1, 15, 16, 17, 40, 100])
def test_windows_score_every_token_once_with_context(n: int) -> None:
    scored = []
    for i, (start, end, score_from) in enumerate(lm.windows(n, max_context=16, carried=8)):
        assert end - start <= 16
        if i:
            assert score_from - start >= 8
        scored.extend(range(score_from, end))
    assert scored == list(range(1, n + 1))


def test_token_scores_do_not_depend_on_batch_composition(tiny_lm: lm.ReferenceLM) -> None:
    rng = np.random.default_rng(0)
    docs = [list(rng.integers(1, VOCAB, size=k)) for k in (3, 40, 9, 17)]
    together = tiny_lm.token_scores(docs)
    for doc, scores in zip(docs, together, strict=True):
        (alone,) = tiny_lm.token_scores([doc])
        np.testing.assert_allclose(alone, scores, atol=1e-5)
        assert scores.shape == (len(doc), 4)


def test_first_window_matches_a_single_full_pass(tiny_lm: lm.ReferenceLM) -> None:
    doc = list(np.random.default_rng(1).integers(1, VOCAB, size=40))
    (scores,) = tiny_lm.token_scores([doc])
    seq = torch.tensor([[0, *doc[:15]]])
    with torch.no_grad():
        logp = tiny_lm.model(input_ids=seq).logits[0, :-1].log_softmax(-1)
    expected = logp.gather(-1, seq[0, 1:, None]).squeeze(-1).numpy()
    np.testing.assert_allclose(scores[:15, lm.LOGPROB], expected, atol=1e-5)


def test_rank_and_top10_agree(tiny_lm: lm.ReferenceLM) -> None:
    doc = list(np.random.default_rng(2).integers(1, VOCAB, size=30))
    (scores,) = tiny_lm.token_scores([doc])
    rank = np.exp(scores[:, lm.LOGRANK])
    assert (rank >= 1 - 1e-4).all()
    assert ((rank <= 10 + 1e-3) == (scores[:, lm.TOP10] == 1)).all()


def test_lm_span_features_attribute_by_midpoint() -> None:
    # tokens keep the leading space like gpt2, so " b" (3, 5) has midpoint 4 -> span 1
    offsets = [(0, 1), (1, 3), (3, 5), (5, 7)]
    scores = np.array([[-1.0, 0, 1, 1], [-3.0, 0, 1, 1], [-2.0, 1, 0, 2], [-4.0, 1, 0, 2]])
    out = lm.span_features(offsets, scores, [(0, 3), (4, 7), (8, 9)])
    np.testing.assert_allclose(out[:2, lm.LOGPROB], [-2.0, -3.0])
    np.testing.assert_allclose(out[:2, 4], [0.5, -0.5])
    assert np.isnan(out[2, :5]).all()
    assert out[0, 5] == pytest.approx(0.5)


def test_lexical_features_ignore_punctuation_spacing_and_case() -> None:
    natural = "High-salt diets don't help, we found."
    moses = "high - salt diets do n't help , we found ."
    a = lexical.span_features(natural, [(0, len(natural))])
    b = lexical.span_features(moses, [(0, len(moses))])
    np.testing.assert_allclose(a, b)


def test_mattr() -> None:
    assert lexical.mattr(["a", "b", "a"]) == pytest.approx(2 / 3)
    assert lexical.mattr(list("abcdefghijk"), window=10) == 1.0
    assert lexical.mattr(["a"] * 12, window=10) == pytest.approx(0.1)
    assert np.isnan(lexical.mattr([]))


def test_repetition_looks_back_three_spans() -> None:
    text = "red fox. blue sky. green sea. dark night. red fox."
    spans = [(m.start(), m.end()) for m in re.finditer(r"[^.]+\.", text)]
    spans = [(s + (text[s] == " "), e) for s, e in spans]
    out = lexical.span_features(text, spans)
    assert out[0, 4] == 0.0
    assert out[4, 4] == 0.0  # the repeat is four spans back
    out = lexical.span_features(text, [spans[0], spans[1], spans[4]])
    assert out[2, 4] == pytest.approx(1.0)


def test_syntax_features(nlp: Any) -> None:
    text = "The cat sat. It slept."
    out = syntax.span_features(nlp(text), [(0, 12), (13, 22)])
    assert out.shape == (2, len(POS_TAGS) + 1)
    np.testing.assert_allclose(out[:, : len(POS_TAGS)].sum(1), [1.0, 1.0])
    assert (out[:, -1] > 0).all()


def test_dep_depths_are_zero_only_at_roots(nlp: Any) -> None:
    doc = nlp("The quick brown fox jumps over the lazy dog.")
    depths = syntax.dep_depths(doc)
    for tok, d in zip(doc, depths, strict=True):
        assert (d == 0) == (tok.head.i == tok.i)
        if d:
            assert d == depths[tok.head.i] + 1


def _span_doc(i: int, text: str, spans: list[tuple[int, int]]) -> SpanDoc:
    return SpanDoc(
        doc_id=f"raid:{i}",
        text=text,
        label=i % 2,
        group_id=f"raid:g{i}",
        domain="news",
        breakdown="human" if i % 2 == 0 else "gpt2",
        spans=spans,
        span_tokens=[e - s for s, e in spans],
        span_labels=None,
        straddles=[False] * len(spans),
    )


def test_extract_writes_one_row_per_span(tmp_path: Path, tiny_lm: lm.ReferenceLM, nlp: Any) -> None:
    docs = [
        _span_doc(0, "One sentence here. Another one follows.", [(0, 18), (19, 39)]),
        _span_doc(1, "Short.", [(0, 6)]),
        _span_doc(2, "A b c. D e f. G h i.", [(0, 6), (7, 13), (14, 20)]),
    ]
    path = tmp_path / "dev.parquet"
    rows = extract.extract(docs, path, tiny_lm, chunk=2)
    table = pq.read_table(path)
    assert rows == table.num_rows == 6
    assert table["doc_id"].to_pylist() == ["raid:0"] * 2 + ["raid:1"] + ["raid:2"] * 3
    assert table["span_idx"].to_pylist() == [0, 1, 0, 0, 1, 2]
    assert set(FEATURE_NAMES) <= set(table.column_names)
    assert not path.with_suffix(".parquet.partial").exists()
