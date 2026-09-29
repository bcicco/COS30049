"""Token counting with the exact encoder Phase 4 will use.

Counts come from `answerdotai/ModernBERT-base`'s tokenizer, but this may
need to be updated further down the track.
"""

import functools
from typing import Final

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

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


def assign_to_spans(
    offsets: list[tuple[int, int]], spans: list[tuple[int, int]]
) -> list[int]:
    """Span index per token by character midpoint"""
    # NOTE:
    # Returns -1 for tokens outside every spans

    out = [-1] * len(offsets)
    si = 0
    for ti, (start, end) in enumerate(offsets):
        if end <= start:
            # Zero-width tokens carry no characters to attribute.
            continue
        mid = (start + end) / 2.0
        while si < len(spans) and spans[si][1] <= mid:
            si += 1
        if si >= len(spans):
            break
        if spans[si][0] <= mid:
            out[ti] = si
    return out


def count_tokens(
    text: str, spans: list[tuple[int, int]], revision: str | None = None
) -> tuple[int, list[int]]:
    """Count tokens for the whole document and for each span.
    Returns a tuple of `(n_tokens, [n_tokens_per_span])`."""
    if not text:
        return 0, [0] * len(spans)

    enc = tokenizer(revision).encode(text, add_special_tokens=False)
    counts = [0] * len(spans)
    for si in assign_to_spans(enc.offsets, spans):
        if si >= 0:
            counts[si] += 1
    return len(enc.ids), counts
