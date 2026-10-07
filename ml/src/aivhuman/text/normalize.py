"""unicode normalisation, carrying offsets through NFC, stable hashing"""

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
# moses style spacing: " ." -> ".", "( " -> "(" etc. seqxgpt pubmed/arxiv docs have it,
# xsum/cnn dont
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%)\]}])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")
_SPACED_HYPHEN_RE = re.compile(r"(?<=\w) - (?=\w)")


class CompositionStraddlesBoundary(ValueError):
    """boundary offset cant be carried through NFC safely"""

    # Raised by nfc_split when no safe cut point exists nearby, makes life easier to debug


class NfcSplit(NamedTuple):
    text: str  # == nfc(original)
    cut: int  # offset into text
    retract: int  # how far the pre-NFC cut moved back, <= 0
    nfc_delta: int  # length change of the prefix, negative if NFC composed


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def is_nfc(s: str) -> bool:
    return unicodedata.is_normalized("NFC", s)


def nfc_split(s: str, cut: int) -> NfcSplit:
    # ************* NOTE **************
    # This is tricky... specific to SeqXGPT's prompt_len because its offset into the raw string
    # and NFC can shorten a string. Can't just normalise and reuse the offset
    # instead, split first, normalise each side, then check two halves = whole

    if not 0 <= cut <= len(s):
        raise ValueError(f"cut {cut} outside [0, {len(s)}]")

    # combining marks compose onto the char before, so cutting right before one splits
    # the sequence. retract until the char at the cut is a starter

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
    return _WS_RE.sub(" ", s).strip()


def strip_moses_spacing(s: str) -> str:
    # "disease ." -> "disease."
    s = _SPACED_HYPHEN_RE.sub("-", s)
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    return _SPACE_AFTER_OPEN_RE.sub(r"\1", s)


def text_key(s: str) -> str:
    # aggressive, for dedup across corpora
    return collapse_ws(strip_moses_spacing(nfc(s).casefold()))


def content_key(s: str, n_chars: int) -> str:
    """prefix key used to recover seqxgpt base docs"""
    # NOTE: n char: Measured cross-file recovery on SeqXGPT-Bench: ~98% at 30, ~97.5% at 40,
    # ~96% at 60, ~83% at 120.

    if n_chars <= 0:
        # debug for weird error during dev
        raise ValueError(f"n_chars must be positive, got {n_chars}")
    return collapse_ws(nfc(s).casefold())[:n_chars]


def stable_hash(s: str, n: int = 16) -> str:
    # ************* NOTE **************
    # Do not use the built in hash bc/ hash will change across runs & processes, if we need to
    # modify / rerun some things it will be a nightmare to track down what changed.

    if not 1 <= n <= 32:
        raise ValueError(f"n must be in [1, 32], got {n}")
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()[:n]


# cyrillic/greek lookalikes that RAIDs homoglyph attack swaps in
_HOMOGLYPH_TABLE = str.maketrans(
    "\u0430\u0435\u0456\u043e\u0441\u0440\u0443\u0445"  # Cyrillic lowercase
    "\u0410\u0412\u0415\u041a\u041c\u041d"
    "\u041e\u0420\u0421\u0422\u0425\u0406"  # Cyrillic capitals
    "\u0391\u0392\u0395\u0396\u0397\u0399\u039a"
    "\u039c\u039d\u039f\u03a1\u03a4\u03a7",  # Greek capitals
    "aeiocpyxABEKMHOPCTXIABEZHIKMNOPTX",
)
_SPACE_RUN_RE = re.compile(r"[^\S\n]{2,}")


def defang(s: str) -> str:
    """Undo char level attacks: drop zero width etc, fold homoglyphs, squash spaces."""
    # newlines are kept
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    return _SPACE_RUN_RE.sub(" ", s.translate(_HOMOGLYPH_TABLE))
