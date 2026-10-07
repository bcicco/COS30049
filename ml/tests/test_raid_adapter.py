import csv
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aivhuman.labels import UnknownLabelError
from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE, Doc, doc_from_json, doc_to_json
from aivhuman.sources.raid import RaidStats, build_docs, load_rows, scan
from aivhuman.sources.raid_parquet import (
    ATTACK_DIR,
    CLEAN_FILE,
    COLUMNS,
    OUTPUT_COLUMNS,
    RaidDeriveStats,
    derive,
    open_clean,
)
from aivhuman.text.normalize import stable_hash
from aivhuman.text.segment import Segmenter

HUMAN_TEXT = "A person wrote this abstract. It has a second sentence."
MACHINE_TEXT = "A model wrote this abstract. It also has a second sentence."
PROMPT = "Write an abstract titled: Trustworthy AI"

_CLEAN_SCHEMA = pa.schema(
    [(name, pa.int32() if name == "prompt_len_chars" else pa.string()) for name in OUTPUT_COLUMNS]
)


def csv_row(**overrides: str) -> dict[str, str]:
    row = {
        "id": "id-0001",
        "adv_source_id": "src-0001",
        "source_id": "src-0001",
        "model": "human",
        "decoding": "",
        "repetition_penalty": "",
        "attack": "none",
        "domain": "abstracts",
        "title": "Trustworthy AI",
        "prompt": "",
        "generation": HUMAN_TEXT,
    }
    row.update(overrides)
    return row


def clean_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "id-0001",
        "adv_source_id": "src-0001",
        "source_id": "src-0001",
        "model": "human",
        "decoding": "",
        "repetition_penalty": "",
        "attack": "none",
        "domain": "abstracts",
        "title": "Trustworthy AI",
        "generation": HUMAN_TEXT,
        "prompt_sha": stable_hash(""),
        "prompt_len_chars": 0,
    }
    row.update(overrides)
    return row


def write_csv(path: Path, rows: Sequence[dict[str, str]], header: Sequence[str] = COLUMNS) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header))
        writer.writeheader()
        writer.writerows([{k: r[k] for k in header} for r in rows])
    return path


def write_clean(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    table = pa.table(
        {name: [row[name] for row in rows] for name in OUTPUT_COLUMNS}, schema=_CLEAN_SCHEMA
    )
    pq.write_table(table, path)
    return path


def docs_from(path: Path, stats: RaidStats) -> list[Doc]:
    return list(build_docs(path, segmenter=Segmenter(), stats=stats))


def test_only_the_none_attack_reaches_the_clean_file(tmp_path: Path) -> None:
    src = write_csv(
        tmp_path / "train.csv",
        [
            csv_row(id="a", attack="none"),
            csv_row(id="b", attack="homoglyph"),
            csv_row(id="c", attack="whitespace"),
            csv_row(id="d", attack="none"),
        ],
    )
    out = tmp_path / "derived"

    stats = derive(src, out)

    assert stats.rows == 4
    assert stats.clean_rows == 2
    assert stats.by_attack == {"none": 2, "homoglyph": 1, "whitespace": 1}
    assert open_clean(out / CLEAN_FILE).metadata.num_rows == 2
    assert sorted(p.name for p in (out / ATTACK_DIR).iterdir()) == [
        "attack=homoglyph",
        "attack=none",
        "attack=whitespace",
    ]


def test_attack_partitions_can_be_skipped(tmp_path: Path) -> None:
    src = write_csv(tmp_path / "train.csv", [csv_row(attack="homoglyph"), csv_row()])
    out = tmp_path / "derived"

    derive(src, out, include_attacks=False)

    assert not (out / ATTACK_DIR).exists()
    assert (out / CLEAN_FILE).exists()


def test_the_prompt_is_replaced_by_a_digest_and_a_length(tmp_path: Path) -> None:
    src = write_csv(
        tmp_path / "train.csv",
        [
            csv_row(id="a", model="gpt4", decoding="greedy", prompt=PROMPT),
            csv_row(id="b", model="mpt", decoding="greedy", prompt=PROMPT),
            csv_row(id="c", model="mpt", decoding="sampling", prompt="A different prompt"),
        ],
    )
    out = tmp_path / "derived"

    derive(src, out)
    table = open_clean(out / CLEAN_FILE).read().to_pydict()

    assert "prompt" not in table
    assert table["prompt_sha"][0] == stable_hash(PROMPT)
    assert table["prompt_sha"][0] == table["prompt_sha"][1], "one prompt, one digest"
    assert table["prompt_sha"][2] != table["prompt_sha"][0]
    assert table["prompt_len_chars"] == [len(PROMPT), len(PROMPT), len("A different prompt")]


def test_generations_containing_newlines_survive_block_boundaries(tmp_path: Path) -> None:
    # needs newlines_in_values=True. block size is small so rows actually cross blocks,
    # with one block this passes either way
    text = "First paragraph ends here.\n\nSecond paragraph, with a comma.\nThird line."
    rows = [csv_row(id=f"id-{i}", generation=f"{text} Row {i}.") for i in range(200)]
    src = write_csv(tmp_path / "train.csv", rows)
    out = tmp_path / "derived"

    stats = derive(src, out, block_size=1024)

    assert stats.rows == 200
    assert stats.blocks > 1, "the fixture must span more than one block"
    generations = open_clean(out / CLEAN_FILE).read().to_pydict()["generation"]
    assert generations == [f"{text} Row {i}." for i in range(200)]


def test_unexpected_csv_columns_raise(tmp_path: Path) -> None:
    # raid has no label column
    src = write_csv(tmp_path / "train.csv", [csv_row()], header=[*COLUMNS[:4], *COLUMNS[5:]])

    with pytest.raises(ValueError, match="expected columns"):
        derive(src, tmp_path / "derived")


def test_open_clean_rejects_a_foreign_parquet(tmp_path: Path) -> None:
    path = tmp_path / "clean.parquet"
    pq.write_table(pa.table({"text": ["hello"], "label": ["1"]}), path)

    with pytest.raises(ValueError, match="expected columns"):
        open_clean(path)


def test_derive_stats_report_the_class_ratio(tmp_path: Path) -> None:
    rows = [csv_row(id="h", model="human")]
    rows += [csv_row(id=f"m{i}", model="gpt4", decoding="greedy") for i in range(4)]
    stats = derive(write_csv(tmp_path / "train.csv", rows), tmp_path / "derived")

    assert stats.clean_human_rows == 1
    assert stats.clean_machine_rows == 4
    assert stats.machine_per_human == 4.0
    assert stats.as_dict()["clean_by_model"] == {"gpt4": 4, "human": 1}


def test_empty_derive_stats_do_not_divide_by_zero() -> None:
    stats = RaidDeriveStats()

    assert stats.clean_frac == 0.0
    assert stats.machine_per_human == 0.0
    assert stats.as_dict()["by_attack"] == {}


def test_derived_parquet_feeds_the_adapter(tmp_path: Path, word_tokenizer: None) -> None:
    src = write_csv(
        tmp_path / "train.csv",
        [
            csv_row(id="h", model="human", generation=HUMAN_TEXT),
            csv_row(
                id="m", model="gpt4", decoding="greedy", prompt=PROMPT, generation=MACHINE_TEXT
            ),
            csv_row(id="x", model="gpt4", attack="homoglyph", generation=MACHINE_TEXT),
        ],
    )
    out = tmp_path / "derived"
    derive(src, out)
    stats = RaidStats()

    docs = docs_from(out / CLEAN_FILE, stats)

    assert [d.doc_id for d in docs] == ["raid:h", "raid:m"]
    assert [d.label for d in docs] == [LABEL_HUMAN, LABEL_MACHINE]
    assert stats.is_green


def test_the_model_column_is_the_label(tmp_path: Path, word_tokenizer: None) -> None:
    write_clean(
        tmp_path / "clean.parquet",
        [
            clean_row(id="h", model="human", generation=HUMAN_TEXT),
            clean_row(id="m", model="gpt4", decoding="greedy", generation=MACHINE_TEXT),
        ],
    )
    stats = RaidStats()

    human, machine = docs_from(tmp_path / "clean.parquet", stats)

    assert human.label == LABEL_HUMAN
    assert human.generator is None, "a human document must not name a generator"
    assert human.label_raw == "human"
    assert machine.label == LABEL_MACHINE
    assert machine.generator == "gpt4"
    assert stats.human_rows == 1
    assert stats.machine_rows == 1
    assert stats.machine_per_human == 1.0


@pytest.mark.parametrize("model", ["Human", "HUMAN", "gpt5", ""])
def test_an_unknown_model_raises(tmp_path: Path, word_tokenizer: None, model: str) -> None:
    # "Human" would get labelled machine if the check was loose
    write_clean(tmp_path / "clean.parquet", [clean_row(model=model)])

    with pytest.raises(UnknownLabelError):
        docs_from(tmp_path / "clean.parquet", RaidStats())


def test_group_id_is_the_source_document_not_the_adversarial_parent(
    tmp_path: Path, word_tokenizer: None
) -> None:
    # grouping on adv_source_id would split the human doc and its generations apart -> leak
    write_clean(
        tmp_path / "clean.parquet",
        [
            clean_row(id="h", model="human", source_id="src-1", adv_source_id="adv-9"),
            clean_row(
                id="m",
                model="gpt4",
                decoding="greedy",
                source_id="src-1",
                adv_source_id="adv-7",
                generation=MACHINE_TEXT,
            ),
        ],
    )

    human, machine = docs_from(tmp_path / "clean.parquet", RaidStats())

    assert human.group_id == machine.group_id == "raid:src-1"
    assert human.meta["adv_source_id"] == "adv-9"
    assert machine.meta["adv_source_id"] == "adv-7"


def test_doc_id_is_raids_own_row_id(tmp_path: Path, word_tokenizer: None) -> None:
    write_clean(tmp_path / "clean.parquet", [clean_row(id="e5e058ce-be2b-459d")])

    (doc,) = docs_from(tmp_path / "clean.parquet", RaidStats())

    assert doc.doc_id == "raid:e5e058ce-be2b-459d"
    assert doc.doc_id.split(":", 1)[0] == "raid"


def test_every_clean_row_lands_in_the_training_pool(tmp_path: Path, word_tokenizer: None) -> None:
    write_clean(tmp_path / "clean.parquet", [clean_row()])

    (doc,) = docs_from(tmp_path / "clean.parquet", RaidStats())

    assert doc.split_role == "train_pool"


def test_raid_docs_never_carry_sentence_labels(tmp_path: Path, word_tokenizer: None) -> None:
    write_clean(tmp_path / "clean.parquet", [clean_row()])

    (doc,) = docs_from(tmp_path / "clean.parquet", RaidStats())

    assert doc.sentences
    assert all(s.label is None for s in doc.sentences)


def test_decoding_survives_into_meta_for_rebalancing(tmp_path: Path, word_tokenizer: None) -> None:
    write_clean(
        tmp_path / "clean.parquet",
        [clean_row(model="mpt", decoding="sampling", repetition_penalty="yes")],
    )

    (doc,) = docs_from(tmp_path / "clean.parquet", RaidStats())

    assert doc.meta["decoding"] == "sampling"
    assert doc.meta["repetition_penalty"] == "yes"
    assert doc.meta["attack"] == "none"


def test_text_is_nfc_normalised(tmp_path: Path, word_tokenizer: None) -> None:
    text = unicodedata.normalize("NFD", "Café life is a sentence. Naïve too.")
    assert not unicodedata.is_normalized("NFC", text), "fixture must be decomposed"
    write_clean(tmp_path / "clean.parquet", [clean_row(generation=text)])

    (doc,) = docs_from(tmp_path / "clean.parquet", RaidStats())

    assert unicodedata.is_normalized("NFC", doc.text)
    assert doc.meta["n_chars"] == len(text) - 2


def test_docs_survive_a_jsonl_roundtrip(tmp_path: Path, word_tokenizer: None) -> None:
    write_clean(
        tmp_path / "clean.parquet",
        [clean_row(), clean_row(id="m", model="gpt4", decoding="greedy", generation=MACHINE_TEXT)],
    )

    for doc in docs_from(tmp_path / "clean.parquet", RaidStats()):
        assert doc_from_json(doc_to_json(doc)) == doc


def test_scan_establishes_integrity_without_touching_the_text(tmp_path: Path) -> None:
    write_clean(
        tmp_path / "clean.parquet",
        [
            clean_row(id="h", model="human", source_id="src-1"),
            clean_row(id="m", model="gpt4", decoding="greedy", source_id="src-1"),
        ],
    )

    stats = scan(tmp_path / "clean.parquet")

    assert stats.rows == 2
    assert stats.docs == 0, "scan reads no text"
    assert stats.n_groups == 1
    assert stats.groups_without_human == 0
    assert stats.integrity_ok
    assert not stats.is_green, "a scan is not an ingest"


def test_an_adversarial_row_in_the_clean_file_is_caught(tmp_path: Path) -> None:
    write_clean(
        tmp_path / "clean.parquet",
        [clean_row(id="h", model="human"), clean_row(id="x", model="gpt4", attack="homoglyph")],
    )

    stats = scan(tmp_path / "clean.parquet")

    assert stats.attacks_seen == {"none": 1, "homoglyph": 1}
    assert not stats.integrity_ok


def test_a_group_spanning_two_domains_is_reported(tmp_path: Path) -> None:
    write_clean(
        tmp_path / "clean.parquet",
        [
            clean_row(id="a", source_id="src-1", domain="news"),
            clean_row(id="b", source_id="src-1", domain="poetry"),
        ],
    )

    stats = scan(tmp_path / "clean.parquet")

    assert stats.multi_domain_groups == ["src-1"]
    assert not stats.integrity_ok


def test_a_group_with_no_human_row_is_counted(tmp_path: Path) -> None:
    # not an error, just means no negatives for that domain
    write_clean(
        tmp_path / "clean.parquet",
        [
            clean_row(id="h", model="human", source_id="src-1"),
            clean_row(id="m", model="gpt4", decoding="greedy", source_id="src-2"),
        ],
    )

    stats = scan(tmp_path / "clean.parquet")

    assert stats.n_groups == 2
    assert stats.groups_without_human == 1
    assert stats.integrity_ok, "a missing human row is a measurement limit, not a defect"


def test_an_unknown_domain_is_counted(tmp_path: Path) -> None:
    write_clean(tmp_path / "clean.parquet", [clean_row(domain="code")])

    stats = scan(tmp_path / "clean.parquet")

    assert stats.unknown_domains == {"code": 1}
    assert not stats.integrity_ok


def test_an_empty_generation_is_dropped_and_counted(tmp_path: Path, word_tokenizer: None) -> None:
    write_clean(
        tmp_path / "clean.parquet",
        [
            clean_row(id="a"),
            clean_row(id="b", generation="   \n "),
            clean_row(id="c", model="gpt4", decoding="greedy", generation=MACHINE_TEXT),
        ],
    )
    stats = RaidStats()

    docs = docs_from(tmp_path / "clean.parquet", stats)

    assert [d.doc_id for d in docs] == ["raid:a", "raid:c"]
    assert stats.rows == 3
    assert stats.docs == 2
    assert stats.empty_text == 1
    assert stats.is_green, "an accounted-for empty row is not a failure"


def test_load_rows_streams_in_file_order(tmp_path: Path) -> None:
    write_clean(tmp_path / "clean.parquet", [clean_row(id=f"id-{i}") for i in range(5)])

    rows = list(load_rows(tmp_path / "clean.parquet", batch_size=2))

    assert [r.id for r in rows] == [f"id-{i}" for i in range(5)]
