"""Detect how a document was preprocessed before it reached the pipeline. Bit confusing.

SeqXGPT need to be heterogeneously detokenised. Documents sourced from XSum and CNN are
natural-cased, e.g....:

    "Media playback is unsupported on your device 21 June 2013 Last updated ..."

Documents sourced from PubMed and arXiv are lowercased with Moses-style spaced
punctuation::

    "... pathogenesis of autoimmune disease . in this study , we investigated ..."
"""

import re
from typing import Final, NamedTuple

STYLES: Final = ("natural", "moses_lower", "moses_cased", "lower_natural", "unknown")

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_UPPER_RE = re.compile(r"[A-ZÀ-Þ]")
_SENT_PUNCT_RE = re.compile(r"[,.;:!?]")
# " ," / " ." -- punctuation preceded by a space, the Moses detokenisation tell.
_SPACED_PUNCT_RE = re.compile(r"\s[,.;:!?]")

# Below this fraction of letters being uppercase, treat the text as lowercased.
# Natural English prose runs ~2-5% uppercase; fully lowercased text runs ~0%.
_LOWER_THRESHOLD: Final = 0.005
# Above this fraction of sentence punctuation being space-preceded, treat the
# text as Moses-detokenised. Natural prose is ~0; Moses output is ~1.
_SPACED_THRESHOLD: Final = 0.30
# Ratios are meaningless on very short strings.
_MIN_LETTERS: Final = 40
_MIN_PUNCT: Final = 3


class StyleInfo(NamedTuple):
    """Detected preprocessing style plus the ratios it was decided from."""

    style: str
    uppercase_ratio: float
    spaced_punct_ratio: float


def detect_style(text: str) -> StyleInfo:
    """Classify a document's casing and detokenisation.

    Returns one of STYLES:
    """
    letters = _LETTER_RE.findall(text)
    punct = _SENT_PUNCT_RE.findall(text)

    upper_ratio = len(_UPPER_RE.findall(text)) / len(letters) if letters else 0.0
    spaced_ratio = len(_SPACED_PUNCT_RE.findall(text)) / len(punct) if punct else 0.0

    if len(letters) < _MIN_LETTERS or len(punct) < _MIN_PUNCT:
        return StyleInfo("unknown", round(upper_ratio, 4), round(spaced_ratio, 4))

    is_lower = upper_ratio < _LOWER_THRESHOLD
    is_moses = spaced_ratio > _SPACED_THRESHOLD

    if is_moses:
        style = "moses_lower" if is_lower else "moses_cased"
    else:
        style = "lower_natural" if is_lower else "natural"

    return StyleInfo(style, round(upper_ratio, 4), round(spaced_ratio, 4))
