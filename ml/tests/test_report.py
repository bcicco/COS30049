import csv
from pathlib import Path
from typing import Any

import orjson

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
) -> Doc:
    # seqxgpt spans need a label, other sources cant have one
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
    )


def write(path: Path, docs: list[Doc]) -> Path:
    path.write_bytes(b"".join(doc_to_json(d) + b"\n" for d in docs))
    return path


def sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.with_name(f"{path.stem}.stats.json").write_bytes(orjson.dumps(payload))


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
    # docs over 8192 tokens dont fit in one pass
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
    assert LENGTH_BUCKETS[0][0] == 0
    assert [hi for _lo, hi in LENGTH_BUCKETS][:3] == [15, 30, 60]
