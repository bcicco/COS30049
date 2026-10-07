"""spacy POS shares + dependency depth per span"""

from typing import Any, Final

import numpy as np
import spacy

from aivhuman.features import POS_TAGS
from aivhuman.text.tokens import assign_to_spans

SPACY_MODEL: Final = "en_core_web_sm"
_POS_INDEX: Final = {tag: i for i, tag in enumerate(POS_TAGS)}


def load() -> Any:
    """spacy pipeline, ner + lemmatizer off since only tags and parses are used"""
    return spacy.load(SPACY_MODEL, disable=["ner", "lemmatizer"])


def dep_depths(doc: Any) -> list[int]:
    """hops from each token to its sentence root (root = 0)"""
    # walk up the heads until a token w/ a known depth (or the root), then fill the
    # chain back down. memoised so each token is only walked once, O(n) per doc
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
    """[n_spans, 13]: share of each POS tag then mean dep depth, NaN if no tokens"""
    # doc is parsed once as a whole, tokens are then mapped back onto our spans by
    # char offset. spacy sentence splits can differ from pysbd so we dont use them
    n_cols = len(POS_TAGS) + 1
    out = np.full((len(spans), n_cols), np.nan, dtype=np.float64)
    toks = [t for t in doc if not t.is_space]
    owner = assign_to_spans([(t.idx, t.idx + len(t.text)) for t in toks], spans)
    depths = dep_depths(doc)
    counts = np.zeros((len(spans), n_cols), dtype=np.float64)
    n = np.zeros(len(spans), dtype=np.float64)
    for tok, si in zip(toks, owner, strict=True):
        if si < 0:  # token falls between spans
            continue
        n[si] += 1
        col = _POS_INDEX.get(tok.pos_)
        if col is not None:
            counts[si, col] += 1
        counts[si, -1] += depths[tok.i]
    has = n > 0
    out[has] = counts[has] / n[has, None]
    return out
