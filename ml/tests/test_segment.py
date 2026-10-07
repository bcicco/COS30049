import time

import pytest

from aivhuman.text.segment import Segmenter, n_words


def _assert_spans_valid(text: str, spans: list[tuple[int, int]]) -> None:
    prev_end = 0
    for i, (start, end) in enumerate(spans):
        assert 0 <= start < end <= len(text), f"span {i} out of bounds"
        assert start >= prev_end, f"span {i} overlaps its predecessor"
        assert not text[prev_end:start].strip(), f"non-whitespace gap before span {i}"
        assert text[start:end] == text[start:end].strip(), f"span {i} not trimmed"
        prev_end = end
    assert not text[prev_end:].strip(), "non-whitespace tail after last span"


CASES: dict[str, tuple[str, int]] = {
    "plain": ("The first one. And a second! A third?", 3),
    "abbreviations": ("Dr. Smith went to Washington D.C. yesterday. He liked it.", 2),
    "repeated": ("Yes. Yes. Yes.", 3),
    "newlines": ("line one\nline two\nline three", 3),
    "crlf": ("line one\r\nline two\r\nline three", 3),
    "blank_lines": ("Para one ends here.\n\nPara two starts here.", 2),
    "poetry": ("Roses are red\nViolets are blue\nSugar is sweet\nAnd so are you", 4),
    "recipe": ("1. Preheat oven to 350F\n2. Mix flour and sugar\n3. Bake 20 minutes", 3),
    "unicode_quotes": ("He said “hello”. Then she left. 😀 Done.", 3),
    "surrounding_ws": ("  Leading space. Trailing too.   ", 2),
}


@pytest.mark.parametrize(("name", "case"), CASES.items(), ids=list(CASES))
def test_spans_are_valid_and_count_as_expected(
    name: str, case: tuple[str, int], segmenter: Segmenter
) -> None:
    text, expected = case
    spans = segmenter.segment(text)
    _assert_spans_valid(text, spans)
    assert len(spans) == expected
    assert segmenter.stats.failed == 0
    assert segmenter.stats.nonws_gap_chars == 0


def test_end_offsets_are_exclusive(segmenter: Segmenter) -> None:
    text = "One. Two."
    spans = segmenter.segment(text)
    assert spans == [(0, 4), (5, 9)]
    assert text[spans[0][0] : spans[0][1]] == "One."
    assert spans[-1][1] == len(text)


def test_pysbd_still_splits_on_bare_newlines(segmenter: Segmenter) -> None:
    # pysbd 0.3.4 splits on bare newlines so we dont pre-split. if this breaks after an upgrade,
    # poetry/recipe docs turn into one giant sentence
    unpunctuated = "Roses are red\nViolets are blue\nSugar is sweet"
    assert len(segmenter.segment(unpunctuated)) == 3


def test_repeated_sentences_anchor_to_successive_occurrences(segmenter: Segmenter) -> None:
    text = "Yes. Yes. Yes."
    spans = segmenter.segment(text)
    starts = [s for s, _ in spans]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_moses_spaced_punctuation_splits_after_the_period(
    segmenter: Segmenter, seqxgpt_moses_text: str
) -> None:
    spans = segmenter.segment(seqxgpt_moses_text)
    _assert_spans_valid(seqxgpt_moses_text, spans)
    assert len(spans) >= 2
    assert seqxgpt_moses_text[spans[0][0] : spans[0][1]].endswith("disease .")


def test_natural_cased_text_segments(segmenter: Segmenter, seqxgpt_natural_text: str) -> None:
    spans = segmenter.segment(seqxgpt_natural_text)
    _assert_spans_valid(seqxgpt_natural_text, spans)
    assert len(spans) == 3


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "  \t \n "])
def test_empty_and_whitespace_only(text: str, segmenter: Segmenter) -> None:
    assert segmenter.segment(text) == []
    assert segmenter.stats.empty_docs == 1


def test_stats_accumulate(segmenter: Segmenter) -> None:
    segmenter.segment("One. Two.")
    segmenter.segment("Three. Four. Five.")
    stats = segmenter.stats
    assert stats.docs == 2
    assert stats.spans == 5
    assert stats.failed == 0


def test_n_words() -> None:
    assert n_words("a bb  ccc\nd") == 4
    assert n_words("   ") == 0


# from raid abstract 70db035f, hung a worker for 8+ min. keep the missing bracket
CITATION_HANG = "des.[126 127 128 129 130 131 132 133 134 135 136 1"


def test_a_citation_list_does_not_hang(segmenter: Segmenter) -> None:
    # pysbd replace_periods_before_numeric_references backtracks forever on this
    start = time.perf_counter()
    spans = segmenter.segment(CITATION_HANG)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, "pysbd is being handed the construct again"
    _assert_spans_valid(CITATION_HANG, spans)
    assert segmenter.stats.numeric_ref_fallbacks == 1
    assert segmenter.stats.is_healthy


def test_the_fallback_still_covers_every_character(segmenter: Segmenter) -> None:
    text = f"Known devices are listed. {CITATION_HANG} And prose resumes here."

    spans = segmenter.segment(text)

    _assert_spans_valid(text, spans)
    assert segmenter.stats.numeric_ref_fallbacks == 1
    assert "".join(text[a:b] for a, b in spans).replace(" ", "") == text.replace(" ", "")


@pytest.mark.parametrize(
    "text",
    [
        "Devices are known.[126 127 128] Next sentence here.",
        "Refs are cited.[1 2 3] And more text.",
        "A list of 126 127 128 129 130 131 132 numbers with no bracket. Fine.",
        "Numbers follow the colon: [126 127 128 129 130 131 132 133]. Fine.",
    ],
    ids=["closed_3", "closed_short", "no_bracket", "no_terminator"],
)
def test_safe_number_runs_still_go_through_pysbd(text: str, segmenter: Segmenter) -> None:
    spans = segmenter.segment(text)

    _assert_spans_valid(text, spans)
    assert segmenter.stats.numeric_ref_fallbacks == 0
