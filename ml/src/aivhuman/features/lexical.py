"""lexical + repetition features from plain word tokens"""

import functools
import math
import re
from collections import Counter
from typing import Final

import numpy as np
from wordfreq import zipf_frequency

MATTR_WINDOW: Final = 10  # sentences are short, 10 keeps most of them windowed
RARE_ZIPF: Final = 3.0  # ~1 per million
REPETITION_LOOKBACK: Final = 3  # compare against the previous 3 sentences only

# moses splits clitics off ("do n't", "it 's") so glue them back first
_CLITIC_RE = re.compile(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")

# closed class words (articles, pronouns, preps, aux, conj). their rate is a style
# signal that doesnt depend on topic, and theyre left out of the repetition bags
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
    """lowercase word tokens, curly apostrophes folded and clitics rejoined"""
    text = text.lower().replace("’", "'")  # noqa: RUF001
    return _WORD_RE.findall(_CLITIC_RE.sub(r"\1", text))


def mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float:
    """moving average type token ratio, vocab variety w/o the raw TTR length bias"""
    # plain TTR if shorter than the window
    if not tokens:
        return math.nan
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    # slide the window one word at a time, distinct count updated incrementally
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


# zipf lookups are slow and words repeat a lot across docs, so cache
@functools.lru_cache(maxsize=500_000)
def _is_rare(word: str) -> bool:
    return bool(zipf_frequency(word, "en") < RARE_ZIPF)


# cosine of two word count bags, a.b = |a||b|*cos(theta)
def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    norm = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return dot / norm


def span_features(text: str, spans: list[tuple[int, int]]) -> np.ndarray:
    """[n_spans, 5]: mattr, word len, function rate, rare rate, max cos to prev spans"""
    # NaN where a span has no words
    out = np.full((len(spans), 5), np.nan, dtype=np.float64)
    bags: list[Counter[str]] = []
    for i, (start, end) in enumerate(spans):
        toks = words(text[start:end])
        # content words only, else every sentence "repeats" the/and/of
        bag = Counter(t for t in toks if t not in FUNCTION_WORDS)
        if toks:
            out[i, 0] = mattr(toks)
            out[i, 1] = sum(map(len, toks)) / len(toks)
            out[i, 2] = sum(t in FUNCTION_WORDS for t in toks) / len(toks)
            out[i, 3] = sum(map(_is_rare, toks)) / len(toks)
        # repetition: closest of the previous few sentences, 1.0 = same content words
        out[i, 4] = max((_cosine(bag, b) for b in bags[-REPETITION_LOOKBACK:]), default=0.0)
        bags.append(bag)
    return out
