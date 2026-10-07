import re

import pytest

from aivhuman.text.segment import Segmenter

# strings that tend to break offset math after normalising
UNICODE_HOSTILE: tuple[str, ...] = (
    "café au lait",  # e + combining acute -> composes
    "áb̀c",  # two combining marks
    "plain ascii text",
    "가 hangul jamo",  # jamo compose; combining() == 0
    "mañana",  # already NFC
    "zero​width",  # U+200B must survive
    "ﬁnally",  # ligature; NFKC would fold, NFC must not
    "",
)


_WORD_RE = re.compile(r"\S+")


class _WordEncoding:
    def __init__(self, text: str) -> None:
        self.offsets = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
        self.ids = list(range(len(self.offsets)))


class _WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> _WordEncoding:
        return _WordEncoding(text)


@pytest.fixture
def segmenter() -> Segmenter:
    return Segmenter()


@pytest.fixture
def word_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    # fake tokenizer so adapter tests dont need to download modernbert
    monkeypatch.setattr(
        "aivhuman.text.tokens.tokenizer",
        lambda revision=None: _WordTokenizer(),
    )


@pytest.fixture
def seqxgpt_moses_text() -> str:
    # lowercase moses spacing like the pubmed/arxiv parts of seqxgpt
    return (
        "high - salt has been shown to play a role in the pathogenesis of "
        "autoimmune disease . in this study , we investigated the effect of "
        "high - salt on the production of inflammatory mediators by cells ."
    )


@pytest.fixture
def seqxgpt_natural_text() -> str:
    return (
        "Media playback is unsupported on your device 21 June 2013 Last updated "
        "at 12:31 BST. The Market Hall Cinema in Brynmawr used to be run by the "
        "local council. Thanks to a group of volunteers, it has reopened."
    )
