import csv
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import pytest

from aivhuman.acquire import MAGE_FILES
from aivhuman.labels import UnknownLabelError
from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE, SPLIT_ROLES, Doc, doc_from_json, doc_to_json
from aivhuman.sources.mage import (
    COLUMNS,
    MAX_UNPARSED_EXAMPLES,
    SPLIT_ROLE,
    UNPARSED,
    MageStats,
    build_docs,
    load_rows,
)
from aivhuman.text.segment import Segmenter

HUMAN_TEXT = "A person wrote this sentence. Then they wrote a second one."
MACHINE_TEXT = "A model produced this sentence. It produced a second one too."

Row = tuple[str, str, str]

# mage labels are flipped, 1 is human
HUMAN_LABEL = "1"
MACHINE_LABEL = "0"


def write_split(
    directory: Path,
    split: str,
    rows: Sequence[Row],
    *,
    bom: bool = True,
    header: Sequence[str] = COLUMNS,
) -> Path:
    path = directory / MAGE_FILES[split]
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def docs_from(directory: Path, splits: Sequence[str], stats: MageStats) -> list[Doc]:
    return list(build_docs(directory, splits=splits, segmenter=Segmenter(), stats=stats))


def test_polarity_survives_the_adapter(tmp_path: Path, word_tokenizer: None) -> None:
    write_split(
        tmp_path,
        "test",
        [(HUMAN_TEXT, HUMAN_LABEL, "cmv_human"), (MACHINE_TEXT, MACHINE_LABEL, "cmv_gpt4")],
    )
    stats = MageStats()

    human, machine = docs_from(tmp_path, ["test"], stats)

    assert human.label == LABEL_HUMAN
    assert human.generator == "human"
    assert machine.label == LABEL_MACHINE
    assert machine.generator == "gpt4"
    assert stats.human_rows == 1
    assert stats.machine_rows == 1
    assert stats.label_generator_disagreements == 0
    assert stats.is_green


def test_an_unknown_label_value_raises(tmp_path: Path, word_tokenizer: None) -> None:
    write_split(tmp_path, "test", [(HUMAN_TEXT, "2", "cmv_human")])

    with pytest.raises(UnknownLabelError):
        docs_from(tmp_path, ["test"], MageStats())


def test_a_paraphrased_human_row_is_machine_and_keeps_its_generator(
    tmp_path: Path, word_tokenizer: None
) -> None:
    # upstream labels paraphrased human text as machine, keeping that
    write_split(
        tmp_path,
        "ood_gpt_para",
        [
            (HUMAN_TEXT, MACHINE_LABEL, "cnn_human_para"),
            (MACHINE_TEXT, MACHINE_LABEL, "cnn_gpt4_para"),
        ],
    )
    stats = MageStats()

    para_human, para_machine = docs_from(tmp_path, ["ood_gpt_para"], stats)

    assert para_human.label == LABEL_MACHINE
    assert para_human.generator == "human"
    assert para_human.label_raw == "cnn_human_para"
    assert para_human.meta["is_paraphrased"] is True
    assert para_human.split_role == "xcorpus_para_test"
    assert para_machine.generator == "gpt4"
    assert stats.para_rows == 2
    assert stats.para_human_rows == 1
    assert stats.label_generator_disagreements == 0, "paraphrase rows are exempt"
    assert stats.human_rows == 0, "the para testbed is almost entirely machine-labelled"


def test_a_human_src_on_a_machine_row_outside_the_para_testbed_is_a_disagreement(
    tmp_path: Path, word_tokenizer: None
) -> None:
    # src and label never disagree outside the para files in the real data
    write_split(tmp_path, "test", [(HUMAN_TEXT, MACHINE_LABEL, "cmv_human")])
    stats = MageStats()

    docs_from(tmp_path, ["test"], stats)

    assert stats.label_generator_disagreements == 1
    assert not stats.is_green


@pytest.mark.parametrize("bom", [True, False], ids=["with_bom", "without_bom"])
def test_the_bom_never_reaches_a_column_name(
    tmp_path: Path, word_tokenizer: None, bom: bool
) -> None:
    # with plain utf-8 the first header comes out as "\ufefftext"
    write_split(tmp_path, "test", [(HUMAN_TEXT, HUMAN_LABEL, "cmv_human")], bom=bom)

    (row,) = list(load_rows(tmp_path / MAGE_FILES["test"], "test"))

    assert row.text == HUMAN_TEXT
    assert row.src == "cmv_human"


def test_unexpected_columns_raise_before_any_row_is_read(tmp_path: Path) -> None:
    write_split(tmp_path, "test", [], header=["text", "label"])

    with pytest.raises(ValueError, match="expected columns"):
        list(load_rows(tmp_path / MAGE_FILES["test"], "test"))


def test_embedded_newlines_do_not_end_a_row(tmp_path: Path, word_tokenizer: None) -> None:
    # text has real newlines in it, needs newline="" when opening
    text = "First line ends here.\n\nSecond paragraph starts here."
    write_split(
        tmp_path,
        "test",
        [(text, HUMAN_LABEL, "cmv_human"), (MACHINE_TEXT, MACHINE_LABEL, "cmv_gpt4")],
    )
    stats = MageStats()

    docs = docs_from(tmp_path, ["test"], stats)

    assert len(docs) == 2
    assert docs[0].text == text
    assert len(docs[0].sentences) == 2
    assert docs[1].doc_id.endswith(":0000001")


def test_a_field_over_the_default_size_limit_loads(tmp_path: Path) -> None:
    # some docs are over csv's 128kb default, limit is global so reset it here
    default_limit = 131072
    big = "word " * 40_000  # 200 KB
    assert len(big) > default_limit
    write_split(tmp_path, "test", [(big, HUMAN_LABEL, "cmv_human")])
    previous = csv.field_size_limit(default_limit)
    try:
        (row,) = list(load_rows(tmp_path / MAGE_FILES["test"], "test"))
    finally:
        csv.field_size_limit(previous)

    assert row.text == big


def test_text_is_nfc_normalised(tmp_path: Path, word_tokenizer: None) -> None:
    text = unicodedata.normalize("NFD", "Café life is a sentence. Naïve too.")
    assert not unicodedata.is_normalized("NFC", text), "fixture must be decomposed"
    write_split(tmp_path, "test", [(text, HUMAN_LABEL, "cmv_human")])

    (doc,) = docs_from(tmp_path, ["test"], MageStats())

    assert unicodedata.is_normalized("NFC", doc.text)
    assert len(doc.text) == len(text) - 2
    assert doc.text[doc.sentences[0].start : doc.sentences[0].end] == "Café life is a sentence."


def test_split_role_covers_every_acquired_file() -> None:
    assert set(SPLIT_ROLE) == set(MAGE_FILES)
    assert set(SPLIT_ROLE.values()) <= SPLIT_ROLES


def test_split_roles_separate_the_three_testbeds(tmp_path: Path, word_tokenizer: None) -> None:
    for split, src in [
        ("test", "cmv_human"),
        ("ood_gpt", "cnn_gpt4"),
        ("ood_gpt_para", "cnn_gpt4_para"),
    ]:
        write_split(tmp_path, split, [(MACHINE_TEXT, MACHINE_LABEL, src)])

    docs = docs_from(tmp_path, ["test", "ood_gpt", "ood_gpt_para"], MageStats())

    assert [d.split_role for d in docs] == [
        "xcorpus_test",
        "xcorpus_ood_test",
        "xcorpus_para_test",
    ]
    assert [d.doc_id for d in docs] == [
        "mage:test:0000000",
        "mage:ood_gpt:0000000",
        "mage:ood_gpt_para:0000000",
    ]


def test_group_id_is_a_content_hash_so_duplicates_cannot_straddle_a_split(
    tmp_path: Path, word_tokenizer: None
) -> None:
    # same human doc shows up in more than one mage file
    write_split(tmp_path, "test", [(HUMAN_TEXT, HUMAN_LABEL, "cmv_human")])
    write_split(tmp_path, "valid", [(HUMAN_TEXT, HUMAN_LABEL, "cmv_human")])
    write_split(tmp_path, "ood_gpt", [(MACHINE_TEXT, MACHINE_LABEL, "cnn_gpt4")])

    first, second, other = docs_from(tmp_path, ["test", "valid", "ood_gpt"], MageStats())

    assert first.group_id == second.group_id
    assert first.doc_id != second.doc_id
    assert other.group_id != first.group_id
    assert first.group_id.startswith("mage:")


def test_mage_docs_never_carry_sentence_labels(tmp_path: Path, word_tokenizer: None) -> None:
    write_split(tmp_path, "test", [(HUMAN_TEXT, HUMAN_LABEL, "cmv_human")])

    (doc,) = docs_from(tmp_path, ["test"], MageStats())

    assert doc.sentences
    assert all(s.label is None for s in doc.sentences)
    assert all(s.machine_char_frac is None for s in doc.sentences)


def test_docs_survive_a_jsonl_roundtrip(tmp_path: Path, word_tokenizer: None) -> None:
    write_split(
        tmp_path,
        "test",
        [(HUMAN_TEXT, HUMAN_LABEL, "cmv_human"), (MACHINE_TEXT, MACHINE_LABEL, "xsum_gpt4")],
    )

    for doc in docs_from(tmp_path, ["test"], MageStats()):
        assert doc_from_json(doc_to_json(doc)) == doc


def test_an_unparsed_src_is_counted_and_still_emitted(tmp_path: Path, word_tokenizer: None) -> None:
    write_split(
        tmp_path,
        "test",
        [
            (HUMAN_TEXT, HUMAN_LABEL, "cmv_human"),
            (MACHINE_TEXT, MACHINE_LABEL, "newdomain_machine_continuation_gpt5"),
        ],
    )
    stats = MageStats()

    good, unparsed = docs_from(tmp_path, ["test"], stats)

    assert good.domain == "cmv"
    assert unparsed.domain is None
    assert unparsed.generator is None
    assert unparsed.label == LABEL_MACHINE, "polarity does not depend on the src parse"
    assert unparsed.label_raw == "newdomain_machine_continuation_gpt5"
    assert stats.unparsed_src == 1
    assert stats.unparsed_examples == ["newdomain_machine_continuation_gpt5"]
    assert stats.by_domain[UNPARSED] == 1
    assert not stats.is_green


def test_unparsed_examples_are_deduplicated_and_capped(
    tmp_path: Path, word_tokenizer: None
) -> None:
    rows = [
        (MACHINE_TEXT, MACHINE_LABEL, f"nodomain{i}_gpt4") for i in range(MAX_UNPARSED_EXAMPLES + 5)
    ]
    write_split(tmp_path, "test", [*rows, *rows])
    stats = MageStats()

    docs_from(tmp_path, ["test"], stats)

    assert stats.unparsed_src == len(rows) * 2
    assert len(stats.unparsed_examples) == MAX_UNPARSED_EXAMPLES
    assert len(set(stats.unparsed_examples)) == MAX_UNPARSED_EXAMPLES


def test_whitespace_only_text_is_dropped_without_shifting_doc_ids(
    tmp_path: Path, word_tokenizer: None
) -> None:
    # doc_id is positional so dropping a row shouldnt renumber the rest
    write_split(
        tmp_path,
        "test",
        [
            (HUMAN_TEXT, HUMAN_LABEL, "cmv_human"),
            ("   \n  ", HUMAN_LABEL, "cmv_human"),
            (MACHINE_TEXT, MACHINE_LABEL, "cmv_gpt4"),
        ],
    )
    stats = MageStats()

    docs = docs_from(tmp_path, ["test"], stats)

    assert [d.doc_id for d in docs] == ["mage:test:0000000", "mage:test:0000002"]
    assert stats.rows == 3
    assert stats.docs == 2
    assert stats.empty_text == 1
    assert stats.is_green, "a dropped empty row is accounted for, not a failure"


def test_stats_as_dict_is_report_shaped(tmp_path: Path, word_tokenizer: None) -> None:
    write_split(
        tmp_path,
        "test",
        [(HUMAN_TEXT, HUMAN_LABEL, "cmv_human"), (MACHINE_TEXT, MACHINE_LABEL, "cmv_gpt4")],
    )
    stats = MageStats()

    docs_from(tmp_path, ["test"], stats)
    payload = stats.as_dict()

    assert payload["human_frac"] == 0.5
    assert payload["by_split"] == {"test": 2}
    assert payload["by_domain"] == {"cmv": 2}
    assert payload["by_generator"] == {"gpt4": 1, "human": 1}
    assert payload["docs"] == 2


def test_empty_stats_do_not_divide_by_zero() -> None:
    stats = MageStats()

    assert stats.human_frac == 0.0
    assert not stats.is_green
    assert stats.as_dict()["by_domain"] == {}
