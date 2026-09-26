"""SeqXGPT's human/machine boundary, from `prompt_len` to sentence labels.

`prompt_len` is a character offset into the raw record: `text[:prompt_len]`
is human and the rest is machine. Four things can go wrong without raising.
NFC can shorten the prefix and slide the boundary by a character. Our segmenter
can disagree with the one upstream used, putting the boundary inside a sentence.
A generator file can hold a record with no machine text at all. And a record can
carry a boundary that cannot be carried through normalisation.

Every one of those produces sentence labels that are confidently wrong rather
than an error, and these labels are the ground truth Phase 7 measures the
sentence scores against. A quiet defect here does not look like a data problem;
it looks like a model that cannot find sentence boundaries.
"""

import json
import unicodedata
from pathlib import Path

import pytest

from aivhuman.labels import UnknownLabelError
from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE, Doc, doc_from_json, doc_to_json
from aivhuman.sources.seqxgpt import (
    SeqXGPTStats,
    _boundary_snap_dist,
    assign_sentence_labels,
    build_docs,
    load_records,
)
from aivhuman.text.normalize import nfc
from aivhuman.text.segment import Segmenter

HUMAN_SENT = "Human wrote the first sentence here."
MACHINE_SENT = "Machine wrote the second sentence."
MIXED = f"{HUMAN_SENT} {MACHINE_SENT}"

# Two spans and the gap between them, for the label arithmetic.
SPANS = [(0, 10), (11, 20)]


def row(text: str, label: str, prompt_len: int | None = None) -> dict[str, object]:
    """One raw record. `prompt_len` is omitted when None, as `en_human_lines` does."""
    out: dict[str, object] = {"text": text, "label": label}
    if prompt_len is not None:
        out["prompt_len"] = prompt_len
    return out


def write_records(directory: Path, stem: str, rows: list[dict[str, object]]) -> None:
    lines = [json.dumps(r) for r in rows]
    (directory / f"{stem}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def docs_from(directory: Path, stats: SeqXGPTStats) -> list[Doc]:
    return list(build_docs(directory, "calib_pool", segmenter=Segmenter(), stats=stats))


# --------------------------------------------------------------------------- #
# assign_sentence_labels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("boundary", [10, 11])
def test_a_boundary_between_spans_straddles_nothing(boundary: int) -> None:
    """The 92% case: upstream's boundary lands on a sentence edge or in the gap."""
    assert assign_sentence_labels(SPANS, boundary) == [
        (LABEL_HUMAN, 0.0, False),
        (LABEL_MACHINE, 1.0, False),
    ]


@pytest.mark.parametrize(
    ("boundary", "frac", "label"),
    [(4, 0.6, LABEL_MACHINE), (5, 0.5, LABEL_MACHINE), (6, 0.4, LABEL_HUMAN)],
)
def test_a_straddling_span_is_labelled_by_character_majority(
    boundary: int, frac: float, label: int
) -> None:
    """A tie goes to machine, and every straddler is flagged.

    The majority rule is a choice; the flag is what keeps it from being a silent
    one. Because SeqXGPT's boundary is a sentence boundary by construction, a
    straddle means our segmenter disagrees with theirs, so Phase 7 can exclude
    these from strict precision and recall and report how many it excluded.
    """
    assert assign_sentence_labels([(0, 10)], boundary) == [(label, pytest.approx(frac), True)]


def test_boundary_at_zero_makes_every_span_machine() -> None:
    assert assign_sentence_labels(SPANS, 0) == [(LABEL_MACHINE, 1.0, False)] * 2


def test_boundary_past_the_end_makes_every_span_human() -> None:
    assert assign_sentence_labels(SPANS, 999) == [(LABEL_HUMAN, 0.0, False)] * 2


def test_no_spans_gives_no_labels() -> None:
    assert assign_sentence_labels([], 5) == []


def test_a_zero_width_span_does_not_divide_by_zero() -> None:
    """`SentenceSpan` forbids empty spans, so this guard covers raw offsets only."""
    assert assign_sentence_labels([(5, 5)], 0) == [(LABEL_HUMAN, 0.0, False)]


# --------------------------------------------------------------------------- #
# Boundary snap distance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("boundary", "expected"), [(0, 0), (10, 0), (20, 0), (11, 1), (15, 5), (99, 79)]
)
def test_boundary_snap_distance(boundary: int, expected: int) -> None:
    """Measures whether `prompt_len` actually lands on sentence edges.

    A median of 0 (92.2% exact, as measured) means SeqXGPT's per-sentence
    provenance is real. A median of 30 characters would mean the transitions are
    mid-sentence and the ground truth is an approximation -- which changes what
    Phase 7's sentence precision means, so it is reported rather than assumed.
    """
    assert _boundary_snap_dist(SPANS, boundary) == expected


def test_snap_distance_with_no_spans_is_zero() -> None:
    assert _boundary_snap_dist([], 7) == 0


# --------------------------------------------------------------------------- #
# build_docs
# --------------------------------------------------------------------------- #


def test_a_boundary_on_a_sentence_edge_labels_every_span_cleanly(
    tmp_path: Path, word_tokenizer: None
) -> None:
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "gpt2", len(HUMAN_SENT))])
    stats = SeqXGPTStats()

    (doc,) = docs_from(tmp_path, stats)

    assert doc.doc_id == "seqxgpt:en_gpt2_lines:000000"
    assert doc.source == "seqxgpt"
    assert doc.split_role == "calib_pool"
    assert doc.label == LABEL_MACHINE
    assert doc.generator == "gpt2"
    assert doc.group_id.startswith("seqxgpt:base:")
    assert [s.label for s in doc.sentences] == [LABEL_HUMAN, LABEL_MACHINE]
    assert [s.machine_char_frac for s in doc.sentences] == [0.0, 1.0]
    assert not any(s.straddles_boundary for s in doc.sentences)
    assert doc.text[: doc.meta["boundary_nfc"]] == HUMAN_SENT
    assert stats.records == 1
    assert stats.straddle_spans == 0
    assert stats.boundary_snap_dists == [0]


def test_a_boundary_inside_a_sentence_is_flagged(tmp_path: Path, word_tokenizer: None) -> None:
    """Our segmenter disagreeing with theirs has to be visible in the stats."""
    cut = 10
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "gpt2", cut)])
    stats = SeqXGPTStats()

    (doc,) = docs_from(tmp_path, stats)

    first = doc.sentences[0]
    assert first.straddles_boundary
    assert first.label == LABEL_MACHINE
    assert first.machine_char_frac == pytest.approx(
        (len(HUMAN_SENT) - cut) / len(HUMAN_SENT), abs=1e-4
    )
    assert doc.sentences[1].label == LABEL_MACHINE
    assert stats.straddle_spans == 1
    assert stats.total_spans == 2
    assert stats.straddle_rate == 0.5
    assert stats.boundary_snap_dists == [cut]


def test_a_record_without_prompt_len_is_wholly_human(tmp_path: Path, word_tokenizer: None) -> None:
    """`en_human_lines.jsonl` has no `prompt_len` key at all."""
    text = f"{HUMAN_SENT} Another human sentence follows it."
    write_records(tmp_path, "en_human_lines", [row(text, "human")])
    stats = SeqXGPTStats()

    (doc,) = docs_from(tmp_path, stats)

    assert doc.label == LABEL_HUMAN
    assert doc.generator is None
    assert doc.label_raw == "human"
    assert all(s.label == LABEL_HUMAN for s in doc.sentences)
    assert all(s.machine_char_frac == 0.0 for s in doc.sentences)
    assert doc.meta["prompt_len_orig"] is None
    assert doc.meta["boundary_nfc"] == len(doc.text)
    assert stats.boundary_snap_dists == [], "no boundary to measure against"
    assert stats.quarantined == 0


def test_prompt_len_zero_is_wholly_machine(tmp_path: Path, word_tokenizer: None) -> None:
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "gpt2", 0)])
    stats = SeqXGPTStats()

    (doc,) = docs_from(tmp_path, stats)

    assert doc.label == LABEL_MACHINE
    assert all(s.label == LABEL_MACHINE for s in doc.sentences)
    assert all(s.machine_char_frac == 1.0 for s in doc.sentences)
    assert not any(s.straddles_boundary for s in doc.sentences)
    assert stats.boundary_snap_dists == [0]


def test_prompt_len_at_the_end_is_human_and_names_no_generator(
    tmp_path: Path, word_tokenizer: None
) -> None:
    """A generator file can hold a record whose machine continuation is empty.

    `Doc` rejects a human document that names a generator, so the adapter has
    to drop it; `label_raw` keeps the provenance for the report.
    """
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "gpt2", len(MIXED))])

    (doc,) = docs_from(tmp_path, SeqXGPTStats())

    assert doc.label == LABEL_HUMAN
    assert doc.generator is None
    assert doc.label_raw == "gpt2"
    assert all(s.label == LABEL_HUMAN for s in doc.sentences)


def test_an_nfc_shortening_prefix_carries_the_boundary(
    tmp_path: Path, word_tokenizer: None
) -> None:
    """The silent-slide case, end to end.

    The fixture is built with NFD rather than written as a literal: a decomposed
    sequence typed into a source file gets precomposed by most editors on save,
    which would make this pass for the wrong reason. Reusing `prompt_len`
    against the normalised text would put the boundary one character late, which
    relabels the characters either side of every transition.
    """
    human = unicodedata.normalize("NFD", "Café life makes a human sentence.")
    assert not unicodedata.is_normalized("NFC", human), "fixture must be decomposed"
    raw = f"{human} {MACHINE_SENT}"
    write_records(tmp_path, "en_gpt2_lines", [row(raw, "gpt2", len(human))])
    stats = SeqXGPTStats()

    (doc,) = docs_from(tmp_path, stats)

    boundary = doc.meta["boundary_nfc"]
    assert doc.text == nfc(raw)
    assert boundary == len(human) - 1
    assert doc.text[:boundary] == nfc(human)
    assert doc.meta["nfc_delta"] == -1
    assert doc.meta["nfc_retract"] == 0
    assert doc.meta["prompt_len_orig"] == len(human)
    assert [s.label for s in doc.sentences] == [LABEL_HUMAN, LABEL_MACHINE]
    assert not any(s.straddles_boundary for s in doc.sentences)
    assert stats.nfc_shortened == 1


def test_an_uncarryable_boundary_is_quarantined_not_guessed(
    tmp_path: Path, word_tokenizer: None
) -> None:
    """Hangul jamo compose under NFC but report `combining() == 0`.

    Retraction cannot see the hazard, so only the `nfc(head) + nfc(tail) ==
    nfc(whole)` check catches it. Such a record is dropped rather than cut
    somewhere plausible, and the surviving records keep their original row
    indices, since `doc_id` is positional.
    """
    # Escapes rather than a literal: an editor would precompose it on save.
    jamo = "가"  # leading G + vowel A -> one syllable under NFC
    assert len(nfc(jamo)) == 1, "fixture must compose under NFC"
    write_records(
        tmp_path,
        "en_gpt2_lines",
        [
            row(f"{jamo} opens a jamo prefix. {MACHINE_SENT}", "gpt2", 1),
            row(MIXED, "gpt2", len(HUMAN_SENT)),
        ],
    )
    stats = SeqXGPTStats()

    docs = docs_from(tmp_path, stats)

    assert [d.doc_id for d in docs] == ["seqxgpt:en_gpt2_lines:000001"]
    assert stats.records == 2
    assert stats.quarantined == 1


def test_a_label_disagreeing_with_its_file_is_counted_not_dropped(
    tmp_path: Path, word_tokenizer: None
) -> None:
    """The file stem cross-checks the per-record label rather than replacing it."""
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "llama", len(HUMAN_SENT))])
    stats = SeqXGPTStats()

    (doc,) = docs_from(tmp_path, stats)

    assert doc.generator == "llama"
    assert doc.label_raw == "llama"
    assert stats.label_file_mismatches == 1


def test_an_unknown_generator_label_raises(tmp_path: Path, word_tokenizer: None) -> None:
    """Never silently defaulted: an unseen label means the wrong file or a new release."""
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "gpt5", len(HUMAN_SENT))])

    with pytest.raises(UnknownLabelError, match="not in the known set"):
        docs_from(tmp_path, SeqXGPTStats())


def test_doc_id_is_the_physical_line_index(tmp_path: Path, word_tokenizer: None) -> None:
    """`doc_id` is positional, so a blank line must not renumber what follows it."""
    body = json.dumps(row(MIXED, "gpt2", len(HUMAN_SENT)))
    (tmp_path / "en_gpt2_lines.jsonl").write_text(f"{body}\n\n{body}\n", encoding="utf-8")

    assert [r.row_index for r in load_records(tmp_path)] == [0, 2]

    docs = docs_from(tmp_path, SeqXGPTStats())
    assert [d.doc_id for d in docs] == [
        "seqxgpt:en_gpt2_lines:000000",
        "seqxgpt:en_gpt2_lines:000002",
    ]


def test_files_are_read_in_sorted_name_order(tmp_path: Path, word_tokenizer: None) -> None:
    """Ingest output has to be reproducible, and glob order is not."""
    write_records(tmp_path, "en_gptj_lines", [row(MIXED, "gptj", len(HUMAN_SENT))])
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "gpt2", len(HUMAN_SENT))])

    docs = docs_from(tmp_path, SeqXGPTStats())

    assert [d.doc_id.split(":")[1] for d in docs] == ["en_gpt2_lines", "en_gptj_lines"]


def test_built_docs_survive_a_jsonl_roundtrip(tmp_path: Path, word_tokenizer: None) -> None:
    """The invariants are validated on construction, so a roundtrip revalidates them."""
    write_records(
        tmp_path,
        "en_gpt2_lines",
        [row(MIXED, "gpt2", len(HUMAN_SENT)), row(MIXED, "gpt2", 10)],
    )
    write_records(tmp_path, "en_human_lines", [row(MIXED, "human")])
    stats = SeqXGPTStats()

    docs = docs_from(tmp_path, stats)

    assert len(docs) == 3
    for doc in docs:
        assert doc_from_json(doc_to_json(doc)) == doc
        assert all(s.label is not None for s in doc.sentences)
        assert all(
            doc.text[s.start : s.end].strip() == doc.text[s.start : s.end] for s in doc.sentences
        )
    assert sum(stats.styles.values()) == len(docs)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def test_stats_as_dict_is_report_shaped() -> None:
    stats = SeqXGPTStats(
        records=4, total_spans=8, straddle_spans=2, boundary_snap_dists=[0, 0, 0, 5]
    )
    payload = stats.as_dict()

    assert payload["straddle_rate"] == 0.25
    assert payload["boundary_snap_exact_frac"] == 0.75
    assert payload["boundary_snap_p50"] == 0
    assert payload["boundary_snap_max"] == 5


def test_empty_stats_do_not_divide_by_zero() -> None:
    payload = SeqXGPTStats().as_dict()

    assert payload["straddle_rate"] == 0.0
    assert payload["boundary_snap_exact_frac"] == 0.0
    assert payload["boundary_snap_p50"] == 0
    assert payload["boundary_snap_max"] == 0
