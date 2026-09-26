"""Cross-corpus overlap and the Phase 1 report.

The overlap number decides what Phase 7's cross-corpus result means. If a RAID
training document also sits in MAGE, the transfer headline is partly a
memorisation measurement, and that has to be known before the number is quoted.
"""

import csv
from pathlib import Path
from typing import Any

import orjson

from aivhuman.overlap import overlap_report
from aivhuman.report import FINDINGS, LENGTH_BUCKETS, build, metric_rows, summarise
from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE, Doc, SentenceSpan, doc_to_json


def doc(
    source: str,
    doc_id: str,
    text: str,
    *,
    label: int = LABEL_MACHINE,
    domain: str | None = "news",
    generator: str | None = "gpt4",
    group_id: str | None = None,
    split_role: str = "train_pool",
    n_tokens: int = 10,
    style: str = "natural",
) -> Doc:
    """One document with a single span covering its whole text.

    SeqXGPT spans must carry a label and no other source may, so the span is
    built from the source rather than passed in -- the schema enforces this and
    a helper that ignored it could not build a valid SeqXGPT document at all.
    """
    span_label = label if source == "seqxgpt" else None
    return Doc(
        doc_id=doc_id,
        text=text,
        label=label,
        source=source,
        domain=domain,
        generator=generator,
        group_id=group_id or f"{source}:g-{doc_id}",
        split_role=split_role,
        sentences=[
            SentenceSpan(
                start=0,
                end=len(text),
                n_tokens=n_tokens,
                n_words=len(text.split()),
                label=span_label,
            )
        ],
        label_raw="gpt4",
        meta={"detok_style": style},
    )


def write(path: Path, docs: list[Doc]) -> Path:
    path.write_bytes(b"".join(doc_to_json(d) + b"\n" for d in docs))
    return path


def sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.with_name(f"{path.stem}.stats.json").write_bytes(orjson.dumps(payload))


# --------------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------------- #


def test_no_shared_text_is_clean(tmp_path: Path) -> None:
    write(tmp_path / "raid.jsonl", [doc("raid", "raid:a", "One text here.")])
    write(tmp_path / "mage.jsonl", [doc("mage", "mage:a", "A different text.")])

    stats = overlap_report(tmp_path)

    assert stats.shared_keys == {"mage|raid": 0}
    assert stats.is_clean


def test_the_same_document_in_two_corpora_is_found(tmp_path: Path) -> None:
    """The leak that would make cross-corpus transfer a memorisation result."""
    shared = "The quick brown fox jumped over the lazy dog."
    write(tmp_path / "raid.jsonl", [doc("raid", "raid:a", shared)])
    write(tmp_path / "mage.jsonl", [doc("mage", "mage:a", shared)])

    stats = overlap_report(tmp_path)

    assert stats.shared_keys["mage|raid"] == 1
    assert not stats.is_clean
    assert stats.examples["mage|raid"] == [("mage:a", "raid:a")]


def test_matching_ignores_detokenisation_and_case(tmp_path: Path) -> None:
    """SeqXGPT's PubMed text is Moses-spaced and lowercased; RAID's is not.

    An exact string match would miss the same document arriving under two
    conventions, which is the realistic form of this leak.
    """
    write(tmp_path / "raid.jsonl", [doc("raid", "raid:a", "Disease is here. Next one.")])
    write(
        tmp_path / "seqxgpt.jsonl", [doc("seqxgpt", "seqxgpt:a", "disease is here . next  one .")]
    )

    stats = overlap_report(tmp_path)

    assert stats.shared_keys["raid|seqxgpt"] == 1


def test_duplicates_within_one_corpus_are_counted_separately(tmp_path: Path) -> None:
    """An internal duplicate is a statement about the corpus, not the evaluation."""
    text = "Repeated text appears twice."
    write(
        tmp_path / "raid.jsonl",
        [doc("raid", "raid:a", text), doc("raid", "raid:b", text), doc("raid", "raid:c", "Other.")],
    )

    stats = overlap_report(tmp_path)

    assert stats.docs["raid"] == 3
    assert stats.unique_keys["raid"] == 2
    assert stats.internal_duplicate_docs["raid"] == 1
    assert stats.is_clean, "an internal duplicate is not a cross-corpus leak"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_summarise_counts_what_the_report_quotes(tmp_path: Path) -> None:
    path = write(
        tmp_path / "raid.jsonl",
        [
            doc("raid", "raid:h", "A human wrote this.", label=LABEL_HUMAN, generator=None),
            doc("raid", "raid:m", "A model wrote this.", domain="books", n_tokens=70),
        ],
    )

    summary = summarise(path)

    assert summary.source == "raid"
    assert summary.docs == 2
    assert summary.spans == 2
    assert summary.human_docs == 1
    assert summary.machine_docs == 1
    assert summary.machine_per_human == 1.0
    assert summary.by_domain == {"news": 1, "books": 1}
    assert summary.by_generator == {"(human)": 1, "gpt4": 1}
    assert summary.span_length_buckets == {"0-15": 1, "15-30": 0, "30-60": 0, "60+": 1}


def test_token_limits_are_counted_for_the_phase_4_windowing_path(tmp_path: Path) -> None:
    """Documents over 8,192 tokens cannot be encoded in one pass."""
    path = write(
        tmp_path / "raid.jsonl",
        [
            doc("raid", "raid:small", "Short one.", n_tokens=100),
            doc("raid", "raid:big", "Long one.", n_tokens=9000),
        ],
    )

    summary = summarise(path)

    assert summary.docs_over_token_limit == {"4096": 1, "8192": 1}


def test_content_keys_are_collected_during_the_summary_pass(tmp_path: Path) -> None:
    """One pass over 2.4 GB, not two: the report and the overlap share it."""
    path = write(tmp_path / "raid.jsonl", [doc("raid", "raid:a", "Some text here.")])
    keys: dict[str, tuple[str, int]] = {}

    summarise(path, content_keys=keys)

    assert list(keys.values()) == [("raid:a", LABEL_MACHINE)]


def test_metric_rows_carry_the_class_ratio(tmp_path: Path) -> None:
    path = write(
        tmp_path / "raid.jsonl",
        [doc("raid", "raid:h", "Human text.", label=LABEL_HUMAN, generator=None)]
        + [doc("raid", f"raid:m{i}", f"Machine text {i}.") for i in range(9)],
    )

    rows = metric_rows([summarise(path)], {"raid": {"is_green": True}})
    corpus = {m: v for sec, src, m, v in rows if sec == "corpus" and src == "raid"}

    assert corpus["machine_per_human"] == 9.0
    assert corpus["majority_baseline_acc"] == 0.9
    assert corpus["groups"] == 10
    assert corpus["is_green"] is True


def test_findings_record_contradictions_and_gaps() -> None:
    subjects = {row[1]: row for row in FINDINGS}

    assert "inverted" in subjects["MAGE polarity"][3]
    assert "762 originals" in subjects["MAGE paraphrase set"][3]
    assert subjects["non-native English writing"][0] == "gap"


def test_build_writes_the_csvs_and_data(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    path = write(processed / "raid.jsonl", [doc("raid", "raid:a", "Some text.")])
    sidecar(path, {"is_green": True, "segment_stats": {"spans": 1}})
    reports = tmp_path / "reports"

    out = build(processed, reports)

    assert out.name == "phase1_metrics.csv"
    with out.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["section", "source", "metric", "value"]
    lookup = {(r["section"], r["source"], r["metric"]): r["value"] for r in rows}
    assert lookup[("corpus", "raid", "is_green")] == "True"
    assert lookup[("segmentation", "raid", "spans")] == "1"
    assert lookup[("overlap", "all", "is_clean")] == "True"
    assert (reports / "phase1_findings.csv").exists()
    data = orjson.loads((reports / "phase1_data.json").read_bytes())
    assert data["corpora"][0]["source"] == "raid"
    assert data["overlap"]["is_clean"] is True


def test_the_bucket_edges_match_the_calibration_plan() -> None:
    """Phase 6 fits one calibrator per bucket, so the edges are a contract."""
    assert LENGTH_BUCKETS[0][0] == 0
    assert [hi for _lo, hi in LENGTH_BUCKETS][:3] == [15, 30, 60]
