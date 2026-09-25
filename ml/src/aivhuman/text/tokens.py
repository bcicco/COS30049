"""Token counting with the exact encoder Phase 4 will use.

Counts come from ``answerdotai/ModernBERT-base``'s tokenizer, but this may
need to be updated further down the track.
"""

from __future__ import annotations

import functools
from typing import Final

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

__all__ = [
    "N_SPECIAL_TOKENS",
    "TOKENIZER_FILE",
    "TOKENIZER_REPO",
    "count_tokens",
    "tokenizer",
]

TOKENIZER_REPO: Final = "answerdotai/ModernBERT-base"
TOKENIZER_FILE: Final = "tokenizer.json"

# CLS + SEP
N_SPECIAL_TOKENS: Final = 2


# how sick is this
@functools.lru_cache(maxsize=1)
def tokenizer(revision: str | None = None) -> Tokenizer:
    """Load and cache the ModernBERT-base tokenizer."""
    path = hf_hub_download(TOKENIZER_REPO, TOKENIZER_FILE, revision=revision)
    return Tokenizer.from_file(path)


def count_tokens(
    text: str, spans: list[tuple[int, int]], revision: str | None = None
) -> tuple[int, list[int]]:
    """Count tokens for the whole document and for each span.
    Returns a tuple of ``(n_tokens, [n_tokens_per_span])``. Spans are half-open"""
    if not text:
        return 0, [0] * len(spans)

    enc = tokenizer(revision).encode(text, add_special_tokens=False)
    counts = [0] * len(spans)
    if not spans:
        return len(enc.ids), counts

    # Both tokens and spans are in ascending order, so one forward walk suffices.
    si = 0
    for start, end in enc.offsets:
        if end <= start:
            # Zero-width tokens carry no characters to attribute.
            continue
        mid = (start + end) / 2.0
        while si < len(spans) and spans[si][1] <= mid:
            si += 1
        if si >= len(spans):
            break
        if spans[si][0] <= mid:
            counts[si] += 1

    return len(enc.ids), counts
