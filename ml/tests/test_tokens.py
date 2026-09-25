"""Token counting and span attribution."""

from __future__ import annotations

import pytest

from aivhuman.text.tokens import N_SPECIAL_TOKENS, count_tokens, tokenizer


class _FakeEncoding:
    def __init__(self, offsets: list[tuple[int, int]]) -> None:
        self.offsets = offsets
        self.ids = list(range(len(offsets)))


class _FakeTokenizer:
    def __init__(self, offsets: list[tuple[int, int]]) -> None:
        self._offsets = offsets

    def encode(self, text: str, add_special_tokens: bool = True) -> _FakeEncoding:
        return _FakeEncoding(self._offsets)


@pytest.fixture
def fake_tokens(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Install a tokenizer with caller-supplied offsets."""

    def install(offsets: list[tuple[int, int]]) -> None:
        monkeypatch.setattr(
            "aivhuman.text.tokens.tokenizer",
            lambda revision=None: _FakeTokenizer(offsets),
        )

    return install


def test_tokens_partition_across_spans(fake_tokens) -> None:  # type: ignore[no-untyped-def]
    text = "aaa bbb ccc"
    fake_tokens([(0, 3), (3, 7), (7, 11)])
    total, per = count_tokens(text, [(0, 3), (4, 7), (8, 11)])
    assert total == 3
    assert sum(per) <= total
    assert per == [1, 1, 1]


def test_overlapping_offsets_land_in_one_span_each(fake_tokens) -> None:  # type: ignore[no-untyped-def]
    """A multi-byte character split across BPE tokens yields overlapping offsets.

    The real tokenizer does this: an emoji produced ``(12, 14)`` and ``(13, 14)``.
    Midpoint attribution keeps the counts a partition; containment would double
    count or drop.
    """
    text = "x" * 20
    fake_tokens([(12, 14), (13, 14)])
    total, per = count_tokens(text, [(0, 10), (12, 20)])
    assert total == 2
    assert per == [0, 2]


def test_tokens_in_gaps_count_for_the_document_not_a_span(fake_tokens) -> None:  # type: ignore[no-untyped-def]
    """Whitespace between sentences is real text; it just belongs to no sentence."""
    text = "ab\n\ncd"
    fake_tokens([(0, 2), (2, 4), (4, 6)])
    total, per = count_tokens(text, [(0, 2), (4, 6)])
    assert total == 3
    assert per == [1, 1]
    assert sum(per) < total


def test_zero_width_offsets_are_skipped(fake_tokens) -> None:  # type: ignore[no-untyped-def]
    text = "abcdef"
    fake_tokens([(0, 0), (0, 3), (3, 3), (3, 6)])
    total, per = count_tokens(text, [(0, 3), (3, 6)])
    assert total == 4
    assert per == [1, 1]


def test_empty_text_and_no_spans(fake_tokens) -> None:  # type: ignore[no-untyped-def]
    fake_tokens([])
    assert count_tokens("", [(0, 1)]) == (0, [0])
    fake_tokens([(0, 3)])
    assert count_tokens("abc", []) == (1, [])


@pytest.mark.network
def test_modernbert_offsets_are_character_based() -> None:
    """The assumption the whole span-alignment design rests on.

    ``tokenizers`` offsets are byte-based for some ByteLevel-BPE configurations
    and character-based for others. If this were byte-based we would need a
    byte-to-character map per document, and every ``n_tokens`` would be subtly
    wrong in a way no other assertion catches.
    """
    tok = tokenizer()
    text = "Héllo wörld. 😀 naïve café — test."
    enc = tok.encode(text, add_special_tokens=False)

    assert max(end for _, end in enc.offsets) == len(text)
    assert len(text.encode("utf-8")) > len(text), "fixture must be multi-byte"
    for token, (start, end) in zip(enc.tokens, enc.offsets, strict=True):
        assert 0 <= start <= end <= len(text), f"{token}: offsets out of char range"


@pytest.mark.network
def test_special_token_count() -> None:
    """The 8192-token context limit counts CLS and SEP, so the delta is reported."""
    tok = tokenizer()
    text = "One sentence here."
    bare = len(tok.encode(text, add_special_tokens=False).ids)
    with_specials = len(tok.encode(text, add_special_tokens=True).ids)
    assert with_specials - bare == N_SPECIAL_TOKENS


@pytest.mark.network
def test_real_counts_partition() -> None:
    from aivhuman.text.segment import Segmenter

    text = "The first one. And a second! A third?"
    spans = Segmenter().segment(text)
    total, per = count_tokens(text, spans)
    assert sum(per) == total, "no gap tokens expected in single-space text"
