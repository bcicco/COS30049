"""Lexical and repetition features"""

import functools
import math
import re
from collections import Counter
from typing import Final

import numpy as np
from wordfreq import zipf_frequency

MATTR_WINDOW: Final = 10
RARE_ZIPF: Final = 3.0
"""Words below this Zipf frequency (about 1 per million) count as rare."""
REPETITION_LOOKBACK: Final = 3

# Moses splits clitics off their host ("do n't", "it 's"); rejoin them before splitting words.
_CLITIC_RE = re.compile(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")

FUNCTION_WORDS: Final = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "might",
        "more",
        "most",
        "must",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shall",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "upon",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "also",
        "although",
        "among",
        "around",
        "away",
        "else",
        "ever",
        "however",
        "may",
        "much",
        "neither",
        "never",
        "often",
        "perhaps",
        "quite",
        "rather",
        "since",
        "still",
        "though",
        "thus",
        "together",
        "toward",
        "towards",
        "whether",
        "yet",
    ]
)


def words(text: str) -> list[str]:
    """Lowercased word runs, independent of casing and punctuation spacing."""
    text = text.lower().replace("’", "'")  # noqa: RUF001
    return _WORD_RE.findall(_CLITIC_RE.sub(r"\1", text))


def mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float:
    """Moving-average type-token ratio; plain TTR below one window."""
    if not tokens:
        return math.nan
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    total = len(counts)
    for i in range(window, len(tokens)):
        old, new = tokens[i - window], tokens[i]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        counts[new] += 1
        total += len(counts)
    return total / ((len(tokens) - window + 1) * window)


@functools.lru_cache(maxsize=500_000)
def _is_rare(word: str) -> bool:
    return zipf_frequency(word, "en") < RARE_ZIPF


# a.b = |a||b|*cos(theta)
def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    norm = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return dot / norm


def span_features(text: str, spans: list[tuple[int, int]]) -> np.ndarray:
    """[n_spans, 5] MATTR, mean word length, function-word rate, rare-word rate, and max
    content-word cosine to the preceding spans."""
    # NOTE:
    # NaN where a span has no words.
    out = np.full((len(spans), 5), np.nan, dtype=np.float64)
    bags: list[Counter[str]] = []
    for i, (start, end) in enumerate(spans):
        toks = words(text[start:end])
        bag = Counter(t for t in toks if t not in FUNCTION_WORDS)
        if toks:
            out[i, 0] = mattr(toks)
            out[i, 1] = sum(map(len, toks)) / len(toks)
            out[i, 2] = sum(t in FUNCTION_WORDS for t in toks) / len(toks)
            out[i, 3] = sum(map(_is_rare, toks)) / len(toks)
        out[i, 4] = max((_cosine(bag, b) for b in bags[-REPETITION_LOOKBACK:]), default=0.0)
        bags.append(bag)
    return out
