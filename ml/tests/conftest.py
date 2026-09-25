"""Shared fixtures.

Unit tests run offline against synthetic fixtures that reproduce each corpus
format exactly. Synthetic rather than sampled, for two reasons: committing real
corpus rows raises a redistribution question, and a synthetic fixture can be
*built* to contain the pathological case (a combining mark straddling the
SeqXGPT boundary, a ``cnn_human_para`` src, an embedded newline inside a quoted
CSV field) rather than hoping a sampled row happens to.

Anything that touches the network is marked ``@pytest.mark.network`` and excluded
by default, so CI needs no token, no data and no egress.
"""

from __future__ import annotations

import pytest

from aivhuman.text.segment import Segmenter

# Strings chosen to break offset arithmetic: NFC-shortening sequences, a
# ligature NFKC would fold but NFC must not, a zero-width space RAID's
# zero_width_space attack depends on, and Hangul jamo whose composition
# unicodedata.combining() cannot see.
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


@pytest.fixture
def segmenter() -> Segmenter:
    """A fresh segmenter with zeroed stats."""
    return Segmenter()


@pytest.fixture
def seqxgpt_moses_text() -> str:
    """Lowercased, Moses-spaced text in SeqXGPT's PubMed/arXiv style."""
    return (
        "high - salt has been shown to play a role in the pathogenesis of "
        "autoimmune disease . in this study , we investigated the effect of "
        "high - salt on the production of inflammatory mediators by cells ."
    )


@pytest.fixture
def seqxgpt_natural_text() -> str:
    """Natural-cased text in SeqXGPT's XSum/CNN style."""
    return (
        "Media playback is unsupported on your device 21 June 2013 Last updated "
        "at 12:31 BST. The Market Hall Cinema in Brynmawr used to be run by the "
        "local council. Thanks to a group of volunteers, it has reopened."
    )
