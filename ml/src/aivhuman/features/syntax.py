"""Syntactic features from spaCy's pipeline."""

from typing import Any, Final

import numpy as np
import spacy

from aivhuman.features import POS_TAGS
from aivhuman.text.tokens import assign_to_spans

SPACY_MODEL: Final = "en_core_web_sm"
_POS_INDEX: Final = {tag: i for i, tag in enumerate(POS_TAGS)}


def load() -> Any:
    return spacy.load(SPACY_MODEL, disable=["ner", "lemmatizer"])


def dep_depths(doc: Any) -> list[int]:
    """Head-chain length from each token to its sentence root."""
    depth = [-1] * len(doc)
    for tok in doc:
        chain = []
        cur = tok
        while depth[cur.i] < 0 and cur.head.i != cur.i:
            chain.append(cur.i)
            cur = cur.head
        base = depth[cur.i] if depth[cur.i] >= 0 else 0
        depth[cur.i] = base
        for i in reversed(chain):
            base += 1
            depth[i] = base
    return depth


def span_features(doc: Any, spans: list[tuple[int, int]]) -> np.ndarray:
    """[n_spans, len(POS_TAGS) + 1]: POS proportions, then mean dependency depth."""
    n_cols = len(POS_TAGS) + 1
    out = np.full((len(spans), n_cols), np.nan, dtype=np.float64)
    toks = [t for t in doc if not t.is_space]
    owner = assign_to_spans([(t.idx, t.idx + len(t.text)) for t in toks], spans)
    depths = dep_depths(doc)
    counts = np.zeros((len(spans), n_cols), dtype=np.float64)
    n = np.zeros(len(spans), dtype=np.float64)
    for tok, si in zip(toks, owner, strict=True):
        if si < 0:
            continue
        n[si] += 1
        col = _POS_INDEX.get(tok.pos_)
        if col is not None:
            counts[si, col] += 1
        counts[si, -1] += depths[tok.i]
    has = n > 0
    out[has] = counts[has] / n[has, None]
    return out
