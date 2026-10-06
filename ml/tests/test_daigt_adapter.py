"""The DAIGT adapter, from CSV row to :class:`Doc`."""

import csv
from collections.abc import Sequence
from pathlib import Path

import pytest

from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE, SPLIT_ROLES, Doc, doc_from_json, doc_to_json
from aivhuman.sources.daigt import COLUMNS, SPLIT_ROLE, DaigtStats, account, iter_rows, scan, to_doc
from aivhuman.text.segment import Segmenter

HUMAN_TEXT = "A student wrote this essay. Then they wrote a second sentence."
MACHINE_TEXT = "A model produced this essay. It produced a second sentence too."


def write_csv(path: Path, rows: Sequence[tuple[str, str, str, str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return path


def build(path: Path, stats: DaigtStats) -> list[Doc]:
    seg = Segmenter()
    docs = []
    for row in iter_rows(path):
        account(row, stats)
        doc = to_doc(row, seg)
        if doc is not None:
            stats.docs += 1
            docs.append(doc)
    return docs


def test_rows_become_labelled_docs(tmp_path: Path, word_tokenizer: None) -> None:
    path = write_csv(
        tmp_path / "d.csv",
        [
            (HUMAN_TEXT, "0", "Car-free cities", "persuade_corpus", "True"),
            (MACHINE_TEXT, "1", "Exploring Venus", "mistral7binstruct_v1", "False"),
        ],
    )
    stats = DaigtStats()
    human, machine = build(path, stats)

    assert human.label == LABEL_HUMAN and human.generator is None
    assert machine.label == LABEL_MACHINE and machine.generator == "mistral7binstruct_v1"
    assert human.domain == "Car-free cities"
    assert human.meta["raw_source"] == "persuade_corpus"
    assert human.meta["rdizzl3_seven"] is True and machine.meta["rdizzl3_seven"] is False
    assert human.doc_id == "daigt:0000000" and machine.doc_id == "daigt:0000001"
    assert human.group_id != machine.group_id
    assert len(human.sentences) == 2
    assert doc_from_json(doc_to_json(machine)) == machine
    assert stats.is_green
    assert stats.by_generator == {"human": 1, "mistral7binstruct_v1": 1}


def test_empty_text_is_counted_not_emitted(tmp_path: Path, word_tokenizer: None) -> None:
    path = write_csv(
        tmp_path / "d.csv",
        [
            (HUMAN_TEXT, "0", "p", "persuade_corpus", "False"),
            ("   ", "1", "p", "llama2_chat", "False"),
            (MACHINE_TEXT, "1", "p", "llama2_chat", "False"),
        ],
    )
    stats = DaigtStats()
    docs = build(path, stats)
    assert len(docs) == 2
    assert stats.empty_text == 1
    assert stats.is_green


def test_unknown_label_fails_the_integrity_gate(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "d.csv",
        [
            (HUMAN_TEXT, "0", "p", "persuade_corpus", "False"),
            (MACHINE_TEXT, "2", "p", "llama2_chat", "False"),
        ],
    )
    assert not scan(path).integrity_ok


def test_wrong_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    path.write_text("text,label\nhello,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        list(iter_rows(path))


def test_split_role_is_known() -> None:
    assert SPLIT_ROLE in SPLIT_ROLES
