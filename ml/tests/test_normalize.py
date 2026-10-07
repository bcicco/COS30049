import subprocess
import sys
import unicodedata

import pytest

from aivhuman.text.normalize import (
    MAX_BOUNDARY_RETRACT,
    CompositionStraddlesBoundary,
    collapse_ws,
    content_key,
    defang,
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
    decomposed = "café"
    assert len(decomposed) == 5
    assert len(nfc(decomposed)) == 4


def test_nfc_not_nfkc() -> None:
    # nfkc would fold the ligature + mess up the raid homoglyph attacks
    assert nfc("ﬁ") == "ﬁ"  # ligature fi, unchanged
    assert unicodedata.normalize("NFKC", "ﬁ") == "fi"  # what we avoid
    assert "​" in nfc("zero​width")  # zero_width_space survives


@pytest.mark.parametrize("s", UNICODE_HOSTILE)
def test_nfc_split_holds_at_every_cut(s: str) -> None:
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
    # "cafe" + combining acute is 5 chars, 4 after nfc so the cut has to move to 4
    r = nfc_split("café au lait", 5)
    assert r.text == "café au lait"
    assert r.cut == 4
    assert r.text[: r.cut] == "café"
    assert r.nfc_delta == -1


def test_nfc_split_retracts_off_a_combining_mark() -> None:
    r = nfc_split("áb", 1)  # cut between 'a' and its acute
    assert r.text == nfc("áb")
    assert r.retract == -1
    assert r.cut == 0


def test_nfc_split_raises_on_jamo_boundary() -> None:
    # jamo compose but combining() is 0, so only the head + tail == nfc(whole) check catches it
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
    assert stable_hash("abc") == "cf4ab791c62b8d2b"


def test_stable_hash_survives_hash_randomisation() -> None:
    # builtin hash() changes with PYTHONHASHSEED
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


def test_defang_undoes_character_level_attacks() -> None:
    assert defang("I\u200bm\u200ba\u200bg\u200be") == "Image"
    assert (
        defang("\u0399m\u0430g\u0435 s\u0435gm\u0435nt\u0430t\u0456\u043en") == "Image segmentation"
    )
    assert defang("of  individual   image \t patches") == "of individual image patches"


def test_defang_keeps_newlines_and_ordinary_text() -> None:
    s = "Line one.\n\nLine two — with “quotes”, café, \u03b1 = 0.5 and 日本語."
    assert defang(s) == s
    assert defang("end.  \n  next") == "end. \n next"
