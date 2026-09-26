"""Unicode normalisation and offset carrying.

The tests that matter here are the ones covering failures that produce no
exception: a boundary that slides by one character during NFC, or a group_id
that changes between runs. Both would surface weeks later as unexplained model
behaviour.
"""

import subprocess
import sys
import unicodedata

import pytest

from aivhuman.text.normalize import (
    MAX_BOUNDARY_RETRACT,
    CompositionStraddlesBoundary,
    collapse_ws,
    content_key,
    is_nfc,
    nfc,
    nfc_split,
    stable_hash,
    strip_moses_spacing,
    text_key,
)
from conftest import UNICODE_HOSTILE


@pytest.mark.parametrize("s", UNICODE_HOSTILE)
def test_nfc_is_idempotent(s: str) -> None:
    assert nfc(nfc(s)) == nfc(s)
    assert is_nfc(nfc(s))


def test_nfc_changes_length() -> None:
    """The whole reason nfc_split exists: normalisation is not length-preserving."""
    decomposed = "café"
    assert len(decomposed) == 5
    assert len(nfc(decomposed)) == 4


def test_nfc_not_nfkc() -> None:
    """NFKC would fold these; NFC must not.

    Folding the ligature would destroy an orthographic signal, and folding
    would pre-mangle RAID's homoglyph attack before Phase 5 can measure it.
    """
    assert nfc("ﬁ") == "ﬁ"  # ligature fi, unchanged
    assert unicodedata.normalize("NFKC", "ﬁ") == "fi"  # what we avoid
    assert "​" in nfc("zero​width")  # zero_width_space survives


@pytest.mark.parametrize("s", UNICODE_HOSTILE)
def test_nfc_split_holds_at_every_cut(s: str) -> None:
    """Sweep every possible boundary, not a sampled few.

    For any cut that does not raise, three things must hold: the text is exactly
    `nfc(s)`, the returned offset marks the same logical position, and the
    retraction stayed within budget.
    """
    for cut in range(len(s) + 1):
        try:
            r = nfc_split(s, cut)
        except CompositionStraddlesBoundary:
            continue
        assert r.text == nfc(s)
        assert r.text[: r.cut] == nfc(s[: cut + r.retract])
        assert r.text[r.cut :] == nfc(s[cut + r.retract :])
        assert -MAX_BOUNDARY_RETRACT <= r.retract <= 0


def test_nfc_split_carries_boundary_through_composition() -> None:
    """The silent-slide case, pinned.

    `"cafe" + combining acute` is 5 code points; NFC makes it 4. A boundary at
    5 must come back as 4, or every sentence label after it is wrong.
    """
    r = nfc_split("café au lait", 5)
    assert r.text == "café au lait"
    assert r.cut == 4
    assert r.text[: r.cut] == "café"
    assert r.nfc_delta == -1


def test_nfc_split_retracts_off_a_combining_mark() -> None:
    """A cut immediately before a combining mark splits a grapheme.

    The whole cluster must travel with the tail rather than being severed.
    """
    r = nfc_split("áb", 1)  # cut between 'a' and its acute
    assert r.text == nfc("áb")
    assert r.retract == -1
    assert r.cut == 0


def test_nfc_split_raises_on_jamo_boundary() -> None:
    """Hangul jamo compose but report `combining() == 0`.

    Retraction cannot see this, so only the `head + tail == nfc(whole)`
    equality catches it. This is the test that justifies keeping that check.
    """
    jamo = "가"  # leading G + vowel A -> composes to a single syllable
    assert unicodedata.combining(jamo[1]) == 0
    assert len(nfc(jamo)) == 1
    with pytest.raises(CompositionStraddlesBoundary):
        nfc_split(jamo, 1)


@pytest.mark.parametrize("cut", [-1, 99])
def test_nfc_split_rejects_out_of_range(cut: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        nfc_split("short", cut)


def test_collapse_ws() -> None:
    assert collapse_ws("  a\n\tb   c  ") == "a b c"


def test_strip_moses_spacing() -> None:
    assert strip_moses_spacing("disease . in this study , we ( a ) high - salt") == (
        "disease. in this study, we (a) high-salt"
    )


def test_text_key_unifies_detokenisation_styles() -> None:
    """Cross-corpus dedup must see SeqXGPT's spaced form and RAID's as one doc."""
    assert text_key("Disease . Next  one") == text_key("disease.  next one")


def test_content_key_truncates_after_normalising() -> None:
    assert content_key("  HIGH - Salt  has\nbeen  ", 20) == "high - salt has been"


def test_content_key_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="positive"):
        content_key("text", 0)


def test_stable_hash_shape() -> None:
    assert len(stable_hash("abc")) == 16
    assert len(stable_hash("abc", 8)) == 8
    assert stable_hash("abc", 8) == stable_hash("abc")[:8]
    assert stable_hash("abc") != stable_hash("abd")


def test_stable_hash_golden() -> None:
    """Pinned so swapping the hash function is a visible, deliberate change."""
    assert stable_hash("abc") == "cf4ab791c62b8d2b"


def test_stable_hash_survives_hash_randomisation() -> None:
    """The trap this function exists to avoid.

    Builtin `hash()` is salted per process by PYTHONHASHSEED. Had group_id
    used it, Phase 2's split-disjointness assertion would fail intermittently
    long after anyone remembered why.
    """
    code = "from aivhuman.text.normalize import stable_hash; print(stable_hash('seqxgpt base doc'))"
    outs = set()
    for seed in ("0", "1", "random"):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
        )
        outs.add(proc.stdout.strip())
    assert len(outs) == 1, f"hash varied across PYTHONHASHSEED: {outs}"
