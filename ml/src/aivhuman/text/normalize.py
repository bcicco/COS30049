"""Unicode normalisation, offset-safe boundary carrying, and stable hashing."""

# ************ NOTE ************
# I did a big deep dive into unicode normalisation and different forms, NFC, NFD, NFKC, NFKD
# and the various ways they can break offsets. NFC is the way to go, will explain why in the report

import hashlib
import re
import unicodedata
from typing import NamedTuple

# How far the boundary may be retracted to find a safe cut point before we give
# up and quarantine the record. Will explain further in report
MAX_BOUNDARY_RETRACT = 8

_WS_RE = re.compile(r"\s+")
# Moses-style detokenisation artifacts: " ." -> ".", "( " -> "(", etc. SeqXGPT's
# PubMed/arXiv documents arrive in this form; XSum/CNN ones do not.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%)\]}])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")
_SPACED_HYPHEN_RE = re.compile(r"(?<=\w) - (?=\w)")


class CompositionStraddlesBoundary(ValueError):
    """A boundary offset cannot be carried through NFC without corrupting it."""

    # Raised by nfc_split when no safe cut point exists nearby, makes life easier to debug


class NfcSplit(NamedTuple):
    """Result of carrying a pre-NFC offset through normalisation."""

    text: str
    """The NFC-normalised string. Guaranteed equal to `nfc(original)`."""

    cut: int
    """Boundary offset into :attr:`text`. Marks the same logical position."""

    retract: int
    """How far the pre-NFC cut moved to reach a safe boundary (<= 0)."""

    nfc_delta: int
    """Length change NFC applied to the prefix. Negative when NFC composed."""


def nfc(s: str) -> str:
    """Normalise to NFC"""
    return unicodedata.normalize("NFC", s)


def is_nfc(s: str) -> bool:
    """True if `s` is already NFC-normalised."""
    return unicodedata.is_normalized("NFC", s)


def nfc_split(s: str, cut: int) -> NfcSplit:
    """NFC-normalise `s` while carrying the pre-NFC offset `cut` through."""

    # ************* NOTE **************
    # This is tricky... specific to SeqXGPT's prompt_len because its offset into the raw string
    # and NFC can shorten a string. Can't just normalise and reuse the offset
    # instead, split first, normalise each side, then check two halves = whole

    if not 0 <= cut <= len(s):
        raise ValueError(f"cut {cut} outside [0, {len(s)}]")

    # A combining mark composes leftwards onto the character before it, so a cut
    # sitting  before one would split a composition sequence. Retract
    # until the character at the cut is a starter.

    cut_adj = cut
    while 0 < cut_adj < len(s) and unicodedata.combining(s[cut_adj]) != 0:
        cut_adj -= 1
        if cut - cut_adj > MAX_BOUNDARY_RETRACT:
            raise CompositionStraddlesBoundary(
                f"no starter within {MAX_BOUNDARY_RETRACT} chars of offset {cut}"
            )

    head = nfc(s[:cut_adj])
    tail = nfc(s[cut_adj:])

    # half + half = whole safety check

    if head + tail != nfc(s):
        raise CompositionStraddlesBoundary(
            f"nfc(head) + nfc(tail) != nfc(whole) at offset {cut} (adjusted {cut_adj})"
        )

    return NfcSplit(
        text=head + tail,
        cut=len(head),
        retract=cut_adj - cut,
        nfc_delta=len(head) - cut_adj,
    )


def collapse_ws(s: str) -> str:
    """Collapse every whitespace run to a single space and strip the ends."""
    return _WS_RE.sub(" ", s).strip()


def strip_moses_spacing(s: str) -> str:
    """Undo Moses-style detokenisation spacing: `"disease ."` -> `"disease."`"""
    s = _SPACED_HYPHEN_RE.sub("-", s)
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    return _SPACE_AFTER_OPEN_RE.sub(r"\1", s)


def text_key(s: str) -> str:
    """Aggressive normalisation for duplicate detection across corpora."""
    return collapse_ws(strip_moses_spacing(nfc(s).casefold()))


def content_key(s: str, n_chars: int) -> str:
    """Prefix key used to recover SeqXGPT base documents"""

    # NOTE: n char: Measured cross-file recovery on SeqXGPT-Bench: ~98% at 30, ~97.5% at 40,
    # ~96% at 60, ~83% at 120.

    if n_chars <= 0:
        raise ValueError(f"n_chars must be positive, got {n_chars}")
    return collapse_ws(nfc(s).casefold())[:n_chars]


def stable_hash(s: str, n: int = 16) -> str:
    """Process-stable hex digest. Use this, never the builtin `hash()`."""

    # ************* NOTE **************
    # Do not use the built in hash bc/ hash will change across runs & processes, if we need to
    # modify / rerun some things it will be a nightmare to track down what changed.

    if not 1 <= n <= 32:
        raise ValueError(f"n must be in [1, 32], got {n}")
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()[:n]
