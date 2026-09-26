"""Ingest: the JSONL writer and its integrity gate.

The interesting failures here are not crashes. A pass that writes 400k
documents in the wrong order, or loses a batch's segmentation counters, or
leaves a truncated file that the next step reads as a complete corpus, all look
like success. So the assertions are about identity and accounting: same bytes
whatever the worker count, counters that sum to the row count, and nothing at
the output path unless the pass finished.

Everything here runs with `workers=1`, which takes the in-process path. That
is not only for speed: a tokenizer monkeypatched in the parent cannot reach a
spawned child, so a pooled test would silently need the network. The one test
that exercises real workers is marked network for that reason.
"""

import csv
import json
import multiprocessing
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from aivhuman import ingest
from aivhuman.acquire import MAGE_FILES
from aivhuman.ingest import (
    IngestResult,
    IntegrityGateError,
    ingest_mage,
    ingest_raid,
    ingest_seqxgpt,
    write_sidecar,
)
from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE, doc_from_json
from aivhuman.sources.mage import COLUMNS as MAGE_COLUMNS
from aivhuman.text.segment import SegmentStats
from test_mage_adapter import HUMAN_LABEL, HUMAN_TEXT, MACHINE_LABEL, MACHINE_TEXT
from test_raid_adapter import clean_row, write_clean
from test_seqxgpt_boundary import HUMAN_SENT, MIXED, row, write_records


def write_mage(directory: Path, split: str, rows: Sequence[tuple[str, str, str]]) -> Path:
    path = directory / MAGE_FILES[split]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(MAGE_COLUMNS)
        writer.writerows(rows)
    return path


def mage_rows(n: int) -> list[tuple[str, str, str]]:
    """Alternating human and machine rows, each with distinct text."""
    out: list[tuple[str, str, str]] = []
    for i in range(n):
        if i % 2:
            out.append((f"{MACHINE_TEXT} Row {i}.", MACHINE_LABEL, "cmv_gpt4"))
        else:
            out.append((f"{HUMAN_TEXT} Row {i}.", HUMAN_LABEL, "cmv_human"))
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def run_mage(directory: Path, out: Path, **kwargs: Any) -> IngestResult:
    """Ingest only the splits a fixture actually wrote.

    `ingest_mage` defaults to all five MAGE files, which is right in
    production and wrong for a fixture holding one.
    """
    kwargs.setdefault("splits", ["test"])
    kwargs.setdefault("workers", 1)
    return ingest_mage(directory, out, **kwargs)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def test_an_adversarial_row_stops_raid_before_any_segmentation(
    tmp_path: Path, word_tokenizer: None
) -> None:
    """Six seconds of metadata scanning against an hour of wasted segmentation."""
    write_clean(
        tmp_path / "clean.parquet",
        [clean_row(id="h", model="human"), clean_row(id="x", model="gpt4", attack="homoglyph")],
    )

    with pytest.raises(IntegrityGateError, match="RAID integrity check failed"):
        ingest_raid(tmp_path / "clean.parquet", tmp_path / "out", workers=1)

    assert not (tmp_path / "out" / "raid.jsonl").exists()


def test_an_unparsed_src_stops_mage_before_any_segmentation(
    tmp_path: Path, word_tokenizer: None
) -> None:
    write_mage(tmp_path, "test", [*mage_rows(2), (MACHINE_TEXT, MACHINE_LABEL, "nodomain_gpt9")])

    with pytest.raises(IntegrityGateError, match="MAGE integrity check failed"):
        run_mage(tmp_path, tmp_path / "out")

    assert not (tmp_path / "out" / "mage.jsonl").exists()


def test_the_integrity_gate_can_be_skipped_for_one_split(
    tmp_path: Path, word_tokenizer: None
) -> None:
    """A single-file pass legitimately fails the both-classes polarity canary."""
    write_mage(tmp_path, "ood_gpt_para", [(HUMAN_TEXT, MACHINE_LABEL, "cnn_human_para")])

    result = run_mage(
        tmp_path,
        tmp_path / "out",
        splits=["ood_gpt_para"],
        workers=1,
        check_integrity=False,
    )

    assert result.docs == 1
    assert not result.is_green, "one-class pass is recorded as not green"


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def test_raid_jsonl_is_ordered_validated_and_labelled(tmp_path: Path, word_tokenizer: None) -> None:
    write_clean(
        tmp_path / "clean.parquet",
        [
            clean_row(id="h", model="human"),
            clean_row(id="m1", model="gpt4", decoding="greedy", generation=MACHINE_TEXT),
            clean_row(id="m2", model="mpt", decoding="sampling", generation=MACHINE_TEXT),
        ],
    )

    result = ingest_raid(tmp_path / "clean.parquet", tmp_path / "out", workers=1, batch_size=2)

    docs = [doc_from_json(line) for line in result.path.read_bytes().splitlines()]
    assert [d.doc_id for d in docs] == ["raid:h", "raid:m1", "raid:m2"]
    assert [d.label for d in docs] == [LABEL_HUMAN, LABEL_MACHINE, LABEL_MACHINE]
    assert result.rows == 3
    assert result.docs == 3
    assert result.bytes_written == result.path.stat().st_size
    assert result.is_green
    assert result.segment_healthy


def test_mage_jsonl_spans_splits_in_file_order(tmp_path: Path, word_tokenizer: None) -> None:
    write_mage(tmp_path, "test", mage_rows(2))
    write_mage(tmp_path, "valid", mage_rows(2))

    result = run_mage(tmp_path, tmp_path / "out", splits=["valid", "test"], workers=1, batch_size=1)

    ids = [d["doc_id"] for d in read_jsonl(result.path)]
    assert ids == [
        "mage:valid:0000000",
        "mage:valid:0000001",
        "mage:test:0000000",
        "mage:test:0000001",
    ]


def test_seqxgpt_ingest_keeps_its_sentence_labels(tmp_path: Path, word_tokenizer: None) -> None:
    """SeqXGPT is the only corpus whose spans carry labels, and ingest must not drop them."""
    write_records(tmp_path, "en_gpt2_lines", [row(MIXED, "gpt2", len(HUMAN_SENT))])

    result = ingest_seqxgpt(tmp_path, tmp_path / "out")

    (doc,) = [doc_from_json(line) for line in result.path.read_bytes().splitlines()]
    assert [s.label for s in doc.sentences] == [LABEL_HUMAN, LABEL_MACHINE]
    assert result.workers == 1, "deliberately sequential"
    assert result.is_green


def test_every_written_line_is_a_valid_document(tmp_path: Path, word_tokenizer: None) -> None:
    """The output is only useful if it reloads, so the writer is checked by reloading."""
    write_mage(tmp_path, "test", mage_rows(12))

    result = run_mage(tmp_path, tmp_path / "out", workers=1, batch_size=5)

    docs = [doc_from_json(line) for line in result.path.read_bytes().splitlines()]
    assert len(docs) == 12
    for doc in docs:
        assert doc.source == "mage"
        assert doc.sentences


def test_empty_rows_are_dropped_but_still_accounted(tmp_path: Path, word_tokenizer: None) -> None:
    """`docs + empty_text == rows` is what makes a short file provably complete."""
    write_clean(
        tmp_path / "clean.parquet",
        [
            clean_row(id="a"),
            clean_row(id="b", generation="  \n "),
            clean_row(id="c", model="gpt4", decoding="greedy", generation=MACHINE_TEXT),
        ],
    )

    result = ingest_raid(tmp_path / "clean.parquet", tmp_path / "out", workers=1)

    assert result.rows == 3
    assert result.docs == 2
    assert result.adapter_stats["empty_text"] == 1
    assert result.is_green


def test_a_batch_boundary_does_not_change_the_output(tmp_path: Path, word_tokenizer: None) -> None:
    """Batching is an implementation detail and must not be observable."""
    write_mage(tmp_path, "test", mage_rows(9))

    outputs = []
    for batch_size in (1, 2, 4, 9, 64):
        out = tmp_path / f"out{batch_size}"
        result = run_mage(tmp_path, out, workers=1, batch_size=batch_size)
        outputs.append(result.path.read_bytes())

    assert len({len(o) for o in outputs}) == 1
    assert len(set(outputs)) == 1


def test_segment_counters_are_summed_across_batches(tmp_path: Path, word_tokenizer: None) -> None:
    """Each task returns its own delta, so a lost merge would undercount silently."""
    write_mage(tmp_path, "test", mage_rows(8))

    one = run_mage(tmp_path, tmp_path / "a", batch_size=8)
    many = run_mage(tmp_path, tmp_path / "b", batch_size=2)

    assert one.segment_stats == many.segment_stats
    assert one.segment_stats["docs"] == 8
    assert one.segment_stats["spans"] > 8
    assert sum(one.adapter_stats["styles"].values()) == 8


def test_merging_segment_stats_covers_every_field() -> None:
    """A new SegmentStats field must not be silently left out of the merge."""
    total, delta = SegmentStats(), SegmentStats()
    for i, name in enumerate(SegmentStats.model_fields, start=1):
        setattr(total, name, i)
        setattr(delta, name, i * 10)

    ingest._merge_segment_stats(total, delta)

    for i, name in enumerate(SegmentStats.model_fields, start=1):
        assert getattr(total, name) == i * 11, name


# --------------------------------------------------------------------------- #
# Crash safety and the sidecar
# --------------------------------------------------------------------------- #


def test_a_failed_pass_leaves_nothing_at_the_output_path(
    tmp_path: Path, word_tokenizer: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 76-minute pass that dies must not leave a file verify would trust.

    The partial file is deliberately left behind for inspection; what matters is
    that it is not named `*.jsonl`, because that is the glob everything
    downstream reads.
    """
    write_mage(tmp_path, "test", mage_rows(6))

    def explode(batch: Any) -> Any:
        raise RuntimeError("worker died")

    monkeypatch.setattr(ingest, "_task", explode)
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="worker died"):
        run_mage(tmp_path, out, batch_size=2)

    assert not (out / "mage.jsonl").exists()
    assert (out / "mage.jsonl.partial").exists()


def test_the_sidecar_records_what_the_report_needs(tmp_path: Path, word_tokenizer: None) -> None:
    write_mage(tmp_path, "test", mage_rows(4))
    out = tmp_path / "out"

    result = run_mage(tmp_path, out, workers=1)
    path = write_sidecar(result, out)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "mage.stats.json"
    assert payload["source"] == "mage"
    assert payload["docs"] == 4
    assert payload["tokenizer_repo"]
    assert payload["adapter_stats"]["by_domain"] == {"cmv": 4}
    assert payload["segment_stats"]["docs"] == 4
    assert isinstance(payload["path"], str), "Path must serialise for JSON"


# --------------------------------------------------------------------------- #
# The real pool
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_workers_do_not_change_the_bytes(tmp_path: Path) -> None:
    """The claim the whole module rests on: parallelism is not observable.

    Marked network because spawned workers load the real tokenizer -- a
    fixture patched in this process cannot reach them, which is also why every
    other test here runs in-process.
    """
    write_mage(tmp_path, "test", mage_rows(300))

    serial = run_mage(tmp_path, tmp_path / "serial", batch_size=32)
    parallel = run_mage(tmp_path, tmp_path / "parallel", workers=3, batch_size=32)

    assert parallel.workers == 3
    assert serial.path.read_bytes() == parallel.path.read_bytes()
    assert serial.segment_stats == parallel.segment_stats
    assert serial.adapter_stats == parallel.adapter_stats


# --------------------------------------------------------------------------- #
# Surviving a task that never returns
# --------------------------------------------------------------------------- #


class _StubHandle:
    """An `apply_async` handle that either returns or hangs, on demand."""

    def __init__(self, batch: Any, hang_ids: set[str]) -> None:
        self.batch = batch
        self.hangs = any(getattr(r, "id", None) in hang_ids for r in batch.rows)

    def get(self, timeout: float | None = None) -> Any:
        if self.hangs:
            raise multiprocessing.TimeoutError
        return ingest._Result([b'{"stub": 1}\n'] * len(self.batch.rows), SegmentStats(), {})


class _StubPool:
    def __init__(self, hang_ids: set[str], log: list[str]) -> None:
        self.hang_ids = hang_ids
        self.log = log
        self.terminated = False

    def apply_async(self, func: Any, args: tuple[Any, ...]) -> _StubHandle:
        (batch,) = args
        self.log.append(",".join(str(getattr(r, "id", "?")) for r in batch.rows))
        return _StubHandle(batch, self.hang_ids)

    def terminate(self) -> None:
        self.terminated = True


def raid_rows(ids: Sequence[str]) -> list[Any]:
    from aivhuman.sources.raid import RawRow

    return [RawRow(**clean_row(id=i)) for i in ids]


def stub_runner(
    monkeypatch: pytest.MonkeyPatch, hang_ids: set[str]
) -> tuple[Any, list[str], list[_StubPool]]:
    """A _PoolRunner whose pools are stubs, so a hang is deterministic."""
    runner = ingest._PoolRunner(workers=2, task_timeout=0.01)
    submitted: list[str] = []
    pools: list[_StubPool] = []

    def ensure(self: Any = runner) -> _StubPool:
        if self._pool is None:
            self._pool = _StubPool(hang_ids, submitted)
            pools.append(self._pool)
        return self._pool

    monkeypatch.setattr(runner, "_ensure_pool", ensure)
    return runner, submitted, pools


def test_a_hung_task_is_isolated_and_the_rest_still_run(
    monkeypatch: pytest.MonkeyPatch, word_tokenizer: None
) -> None:
    """One pathological document must not cost the batch it happens to share.

    A regex holds the GIL, so the worker cannot be interrupted -- the pool has
    to be terminated, which kills every task queued behind the culprit too.
    Those get resubmitted, the offending batch is retried one document at a
    time, and only the document that actually hangs skips pysbd.
    """
    window = [
        ingest._Batch("raid", raid_rows(["a1", "a2"])),
        ingest._Batch("raid", raid_rows(["b1", "BAD", "b3"])),
        ingest._Batch("raid", raid_rows(["c1", "c2"])),
    ]
    runner, submitted, pools = stub_runner(monkeypatch, {"BAD"})

    results = runner.run(window)

    assert runner.timeouts == 1
    assert runner.forced_fallback_docs == ["BAD"], "only the culprit falls back"
    assert pools[0].terminated, "a poisoned pool must not be reused"
    assert len(pools) > 1, "the pool is rebuilt to finish the window"
    # The whole window is submitted up front, so anything queued behind the
    # culprit dies with the terminated pool and is redone -- at the window level
    # ("c1,c2") and again inside the isolation pass ("b3"). That repeated work
    # is the cost of a hang, and it is bounded by the window size.
    assert submitted == [
        "a1,a2",
        "b1,BAD,b3",
        "c1,c2",
        "b1",
        "BAD",
        "b3",
        "b3",
        "c1,c2",
    ]
    # Four healthy stub batches plus the one real fallback document.
    assert sum(len(r.lines) for r in results) == 2 + 1 + 1 + 1 + 2


def test_two_hung_documents_in_one_window_both_resolve(
    monkeypatch: pytest.MonkeyPatch, word_tokenizer: None
) -> None:
    """RAID has two of these, and they could land in the same window."""
    window = [
        ingest._Batch("raid", raid_rows(["BAD1", "x"])),
        ingest._Batch("raid", raid_rows(["y", "BAD2"])),
    ]
    runner, _submitted, _pools = stub_runner(monkeypatch, {"BAD1", "BAD2"})

    runner.run(window)

    assert runner.timeouts == 2
    assert sorted(runner.forced_fallback_docs) == ["BAD1", "BAD2"]


def test_the_forced_fallback_produces_valid_documents(word_tokenizer: None) -> None:
    """The fallback is what actually ships for a pathological document."""
    batch = ingest._Batch("raid", raid_rows(["only"]))

    result = ingest._task(batch, use_pysbd=False)

    (line,) = result.lines
    doc = doc_from_json(line)
    assert doc.doc_id == "raid:only"
    assert doc.sentences, "a fallback document still needs spans"
    assert all(
        doc.text[s.start : s.end].strip() == doc.text[s.start : s.end] for s in doc.sentences
    )


def test_a_fallback_segmenter_never_calls_pysbd() -> None:
    """Belt and braces: the flag, not the input, decides."""
    from aivhuman.text.segment import Segmenter

    seg = Segmenter(use_pysbd=False)
    seg._seg = None  # any pysbd call would now raise AttributeError

    spans = seg.segment("One sentence here. And a second one.\nThird line.")

    assert len(spans) == 3
    assert seg.stats.is_healthy
