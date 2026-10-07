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

SPANS = [(0, 10), (11, 20)]


def row(text: str, label: str, prompt_len: int | None = None) -> dict[str, object]:
    out: dict[str, object] = {"text": text, "label": label}
    if prompt_len is not None:
        out["prompt_len"] = prompt_len
    return out


def write_records(directory: Path, stem: str, rows: list[dict[str, object]]) -> None:
    lines = [json.dumps(r) for r in rows]
    (directory / f"{stem}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def docs_from(directory: Path, stats: SeqXGPTStats) -> list[Doc]:
    return list(build_docs(directory, "calib_pool", segmenter=Segmenter(), stats=stats))


@pytest.mark.parametrize("boundary", [10, 11])
def test_a_boundary_between_spans_straddles_nothing(boundary: int) -> None:
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
    # tie goes to machine. straddlers get flagged so they can be excluded later
    assert assign_sentence_labels([(0, 10)], boundary) == [(label, pytest.approx(frac), True)]


def test_boundary_at_zero_makes_every_span_machine() -> None:
    assert assign_sentence_labels(SPANS, 0) == [(LABEL_MACHINE, 1.0, False)] * 2


def test_boundary_past_the_end_makes_every_span_human() -> None:
    assert assign_sentence_labels(SPANS, 999) == [(LABEL_HUMAN, 0.0, False)] * 2


def test_no_spans_gives_no_labels() -> None:
    assert assign_sentence_labels([], 5) == []


def test_a_zero_width_span_does_not_divide_by_zero() -> None:
    assert assign_sentence_labels([(5, 5)], 0) == [(LABEL_HUMAN, 0.0, False)]


@pytest.mark.parametrize(
    ("boundary", "expected"), [(0, 0), (10, 0), (20, 0), (11, 1), (15, 5), (99, 79)]
)
def test_boundary_snap_distance(boundary: int, expected: int) -> None:
    # median 0 means prompt_len really lands on sentence edges (92.2% exact on the real data)
    assert _boundary_snap_dist(SPANS, boundary) == expected


def test_snap_distance_with_no_spans_is_zero() -> None:
    assert _boundary_snap_dist([], 7) == 0


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
    # empty machine continuation, Doc wont allow a human doc with a generator
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "gpt2", len(MIXED))])

    (doc,) = docs_from(tmp_path, SeqXGPTStats())

    assert doc.label == LABEL_HUMAN
    assert doc.generator is None
    assert doc.label_raw == "gpt2"
    assert all(s.label == LABEL_HUMAN for s in doc.sentences)


def test_an_nfc_shortening_prefix_carries_the_boundary(
    tmp_path: Path, word_tokenizer: None
) -> None:
    # NFD so the editor doesnt precompose it. boundary should move back by one
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
    # jamo compose but combining() is 0, record gets dropped and other doc_ids stay the same
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
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "llama", len(HUMAN_SENT))])
    stats = SeqXGPTStats()

    (doc,) = docs_from(tmp_path, stats)

    assert doc.generator == "llama"
    assert doc.label_raw == "llama"
    assert stats.label_file_mismatches == 1


def test_an_unknown_generator_label_raises(tmp_path: Path, word_tokenizer: None) -> None:
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "gpt5", len(HUMAN_SENT))])

    with pytest.raises(UnknownLabelError, match="not in the known set"):
        docs_from(tmp_path, SeqXGPTStats())


def test_doc_id_is_the_physical_line_index(tmp_path: Path, word_tokenizer: None) -> None:
    # blank lines shouldnt renumber the rest
    body = json.dumps(row(MIXED, "gpt2", len(HUMAN_SENT)))
    (tmp_path / "en_gpt2_lines.jsonl").write_text(f"{body}\n\n{body}\n", encoding="utf-8")

    assert [r.row_index for r in load_records(tmp_path)] == [0, 2]

    docs = docs_from(tmp_path, SeqXGPTStats())
    assert [d.doc_id for d in docs] == [
        "seqxgpt:en_gpt2_lines:000000",
        "seqxgpt:en_gpt2_lines:000002",
    ]


def test_files_are_read_in_sorted_name_order(tmp_path: Path, word_tokenizer: None) -> None:
    write_records(tmp_path, "en_gptj_lines", [row(MIXED, "gptj", len(HUMAN_SENT))])
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "gpt2", len(HUMAN_SENT))])

    docs = docs_from(tmp_path, SeqXGPTStats())

    assert [d.doc_id.split(":")[1] for d in docs] == ["en_gpt2_lines", "en_gptj_lines"]


def test_built_docs_survive_a_jsonl_roundtrip(tmp_path: Path, word_tokenizer: None) -> None:
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
