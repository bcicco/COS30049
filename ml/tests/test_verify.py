"""Verification of the ingested JSONL.

The failures worth catching here all produce a file that parses. A span whose
`n_words` drifted, a text that lost its NFC normalisation, a file truncated
mid-write, two corpora built under different label rules -- none of those raise
when the file is read, and all of them are wrong in ways that only surface as
odd model behaviour weeks later.
"""

import json
from pathlib import Path
from typing import Any

import orjson

from aivhuman.schema import doc_to_json
from aivhuman.verify import verify_all, verify_file
from test_schema_roundtrip import raid_doc, seqxgpt_doc


def write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> Path:
    path.write_bytes(b"".join(orjson.dumps(p) + b"\n" for p in payloads))
    return path


def as_payload(doc: Any) -> dict[str, Any]:
    """A document as the dict that lands on disk, so a test can corrupt it."""
    return json.loads(doc_to_json(doc))


def write_sidecar(path: Path, docs: int, green: bool = True) -> None:
    sidecar = path.with_name(f"{path.stem}.stats.json")
    sidecar.write_bytes(orjson.dumps({"docs": docs, "is_green": green}))


def good_raid(tmp_path: Path, n: int = 2) -> Path:
    payloads = []
    for i in range(n):
        payload = as_payload(raid_doc(doc_id=f"raid:doc-{i}"))
        payloads.append(payload)
    path = write_jsonl(tmp_path / "raid.jsonl", payloads)
    write_sidecar(path, n)
    return path


def test_a_good_file_verifies_clean(tmp_path: Path) -> None:
    path = good_raid(tmp_path, 3)

    report = verify_file(path)

    assert report.ok
    assert report.docs == 3
    assert report.spans == 6
    assert report.source == "raid"
    assert report.machine_docs == 3
    assert report.problems == []


def test_a_drifted_word_count_is_caught(tmp_path: Path) -> None:
    """`n_words` is the one stored field that can be rechecked without a tokenizer.

    Nothing validates it at construction, so a span edited by hand or an
    adapter bug that miscounts would otherwise reach Phase 4 unnoticed.
    """
    payload = as_payload(raid_doc())
    payload["sentences"][0]["n_words"] = 99
    path = write_jsonl(tmp_path / "raid.jsonl", [payload])
    write_sidecar(path, 1)

    report = verify_file(path)

    assert not report.ok
    assert any("claims 99 words" in p for p in report.problems)


def test_an_untrimmed_span_is_caught(tmp_path: Path) -> None:
    """A span that includes its trailing space misaligns every highlight by one."""
    payload = as_payload(raid_doc())
    payload["sentences"][0]["end"] = payload["sentences"][0]["end"] + 1
    payload["sentences"][0]["n_words"] = 2
    path = write_jsonl(tmp_path / "raid.jsonl", [payload])
    write_sidecar(path, 1)

    report = verify_file(path)

    assert not report.ok
    assert any("not trimmed" in p for p in report.problems)


def test_non_nfc_text_is_rejected_at_load(tmp_path: Path) -> None:
    """Caught by the schema, and reported as an unreadable file rather than ignored."""
    payload = as_payload(raid_doc())
    payload["text"] = "café is decomposed. Two sentences here."
    path = write_jsonl(tmp_path / "raid.jsonl", [payload])
    write_sidecar(path, 1)

    report = verify_file(path)

    assert not report.ok
    assert any("unreadable" in p for p in report.problems)


def test_a_truncated_file_is_caught_by_its_sidecar(tmp_path: Path) -> None:
    """The case the atomic rename exists to prevent, checked from the other side.

    A file that lost its tail parses perfectly and is simply short. Only the
    recorded document count reveals it.
    """
    path = good_raid(tmp_path, 3)
    write_sidecar(path, 5)

    report = verify_file(path)

    assert not report.ok
    assert report.sidecar_docs == 5
    assert any("sidecar says 5 docs, file has 3" in p for p in report.problems)


def test_a_missing_sidecar_is_a_problem(tmp_path: Path) -> None:
    """Without one there is no record of which label rules produced the file."""
    path = write_jsonl(tmp_path / "raid.jsonl", [as_payload(raid_doc())])

    report = verify_file(path)

    assert not report.ok
    assert any("no sidecar" in p for p in report.problems)


def test_a_sidecar_that_was_not_green_is_a_problem(tmp_path: Path) -> None:
    path = good_raid(tmp_path, 1)
    write_sidecar(path, 1, green=False)

    report = verify_file(path)

    assert any("not green" in p for p in report.problems)


def test_a_doc_id_shared_between_two_corpora_is_caught(tmp_path: Path) -> None:
    """`doc_id` has to be unique across all three corpora, not within one.

    Phase 2's manifests are keyed by `doc_id`, so a collision would silently
    drop one of the two documents from every split that references it.
    """
    clash = as_payload(raid_doc(doc_id="raid:shared"))
    first = write_jsonl(tmp_path / "raid.jsonl", [clash])
    write_sidecar(first, 1)
    second = write_jsonl(tmp_path / "mage.jsonl", [clash])
    write_sidecar(second, 1)

    reports = verify_all(tmp_path)

    assert any(any("duplicate doc_id" in p for p in r.problems) for r in reports)


def test_seqxgpt_keeps_its_sentence_labels_and_others_do_not(tmp_path: Path) -> None:
    seq = write_jsonl(tmp_path / "seqxgpt.jsonl", [as_payload(seqxgpt_doc())])
    write_sidecar(seq, 1)

    report = verify_file(seq)

    assert report.ok
    assert report.labelled_span_docs == 1


def test_problems_are_capped_but_still_counted(tmp_path: Path) -> None:
    """A corrupt file produces one problem per document, and nobody reads 400,000."""
    payload = as_payload(raid_doc())
    payload["sentences"][0]["n_words"] = 99
    payloads = [dict(payload, doc_id=f"raid:doc-{i}") for i in range(80)]
    path = write_jsonl(tmp_path / "raid.jsonl", payloads)
    write_sidecar(path, 80)

    report = verify_file(path)

    assert report.n_problems == 80
    assert len(report.problems) == 50


def test_verify_all_on_an_empty_directory(tmp_path: Path) -> None:
    assert verify_all(tmp_path) == []
