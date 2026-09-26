"""Segment the three corpora into JSONL, in parallel where appropriate"""

# ------------------------------NOTE-----------------------------------------
# I got really carried away with optimising this. I'm doing some d.e training atm and wanted
# to practice.....
# This code is pretty gnarly with the worker management and pooling, pretty proud of
# this so be careful editing it blesss....
import itertools
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO, Final, NamedTuple, Protocol

import orjson
from pydantic import BaseModel, ConfigDict

from aivhuman import config
from aivhuman.schema import Doc, doc_to_json
from aivhuman.sources import mage, raid, seqxgpt
from aivhuman.sources.raid_parquet import CLEAN_FILE
from aivhuman.text.segment import Segmenter, SegmentStats
from aivhuman.text.tokens import TOKENIZER_REPO, tokenizer

# NEEDED TO ADD THIS TO PREVENT PICKLING OVERHEAD >:( s
BATCH_SIZE: Final = 256

# -------------------NOTE-------------------------------------------
# Some notes (i learnt this the hard way, this was painful)

# Tasks submitted per worker per window. `Pool.map` is used rather than
# imap BECAUSE imap drains its input iterable ASAP. which
# pulls all 468k rows into memory as pending tasks :))))
# A window bounds that to workers x TASKS_PER_WORKER x BATCH_SIZE
# documents, and costs one task's worth of idle workers at each window boundary.
TASKS_PER_WORKER: Final = 4

PROGRESS_EVERY: Final = 20_000


# Wall clock allowed for one batch before its worker is presumed hung.
# From benchmarking, a 256-document batch takes a few seconds....


# NOTE this is not a performance knob, it actually makes the diff. betw. corpus that finishes
# or not, so it is set with two orders of magnitude of headroom
TASK_TIMEOUT_S: Final = 120.0

# Wall clock allowed for a single document while isolating a hung batch.
ISOLATION_TIMEOUT_S: Final = 15.0

OUTPUT_NAMES: Final = {
    "raid": "raid.jsonl",
    "mage": "mage.jsonl",
    "seqxgpt": "seqxgpt.jsonl",
}

# Written while a pass runs, renamed on success to prevent confusion between success / trunc. runs
PARTIAL_SUFFIX: Final = ".partial"


class IngestGateError(RuntimeError):
    """A precondition for writing output failed. Nothing was written."""


class IntegrityGateError(IngestGateError):
    """A corpus failed its metadata checks, before any text was segmented."""


class _TextCounters(Protocol):
    """The two adapter counters only a built document can supply."""

    docs: int
    styles: dict[str, int]


class IngestResult(BaseModel):
    """One corpus's pass, as recorded in its sidecar next to the JSONL."""

    model_config = ConfigDict(extra="forbid")

    source: str
    path: Path
    rows: int
    docs: int
    bytes_written: int
    elapsed_s: float
    workers: int
    batch_size: int
    tokenizer_repo: str
    integrity_ok: bool
    is_green: bool
    segment_healthy: bool
    timed_out_batches: int = 0
    forced_fallback_docs: list[str] = []
    """Documents that had to skip pysbd because it would not finish on them.

    Named rather than counted: there were two in RAID, both worth being able to
    look up, and a growing list is a signal about the corpus rather than noise.
    """

    adapter_stats: dict[str, Any]
    segment_stats: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# -----------------------------Worker side------------------------------------- #
#


_SEG: Segmenter | None = None
_FALLBACK_SEG: Segmenter | None = None

# Only pure row-to-document builders may run in a worker.
_BUILDERS: Final[dict[str, Callable[[Any, Segmenter], Doc | None]]] = {
    "mage": mage.to_doc,
    "raid": raid.to_doc,
}


class _Batch(NamedTuple):
    source: str
    rows: list[Any]


class _Result(NamedTuple):
    lines: list[bytes]
    segment_stats: SegmentStats
    styles: dict[str, int]


def _segmenter() -> Segmenter:
    """The per-process segmenter. Expensive to build, not thread-safe."""
    global _SEG
    if _SEG is None:
        _SEG = Segmenter()
    return _SEG


def _fallback_segmenter() -> Segmenter:
    """A segmenter that never calls pysbd, for documents pysbd cannot finish."""
    global _FALLBACK_SEG
    if _FALLBACK_SEG is None:
        _FALLBACK_SEG = Segmenter(use_pysbd=False)
    return _FALLBACK_SEG


def _init_worker() -> None:
    """Pay for the segmenter and the tokenizer once per worker, not once per task."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _segmenter()
    tokenizer()


def _task(batch: _Batch, use_pysbd: bool = True) -> _Result:
    """Segment one batch. Runs in a worker; returns lines and summable counters."""
    seg = _segmenter() if use_pysbd else _fallback_segmenter()
    seg.stats = SegmentStats()
    build = _BUILDERS[batch.source]

    lines: list[bytes] = []
    styles: dict[str, int] = {}
    for row in batch.rows:
        doc = build(row, seg)
        if doc is None:
            continue
        lines.append(doc_to_json(doc) + b"\n")
        style = str(doc.meta["detok_style"])
        styles[style] = styles.get(style, 0) + 1
    return _Result(lines, seg.stats, styles)


# -----------------------Parent side------------------------------- #


class _Progress:
    def __init__(self, source: str, enabled: bool) -> None:
        self.source = source
        self.enabled = enabled
        self.started = perf_counter()
        self.last = 0

    def update(self, docs: int) -> None:
        if not self.enabled or docs - self.last < PROGRESS_EVERY:
            return
        self.last = docs
        elapsed = perf_counter() - self.started
        print(f"  {self.source}: {docs:,} docs, {docs / elapsed:.0f}/s", flush=True)


@contextmanager
def _atomic_writer(path: Path) -> Iterator[BinaryIO]:
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("wb") as fh:
        yield fh
    partial.replace(path)


def _accounted(
    rows: Iterable[Any], account: Callable[[Any, Any], Any], stats: Any
) -> Iterator[Any]:
    """Run the parent-side accounting as rows stream past on their way to a batch."""
    for row in rows:
        account(row, stats)
        yield row


def _batches(rows: Iterable[Any], source: str, size: int) -> Iterator[_Batch]:
    batch: list[Any] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield _Batch(source, batch)
            batch = []
    if batch:
        yield _Batch(source, batch)


def _windows[T](items: Iterator[T], size: int) -> Iterator[list[T]]:
    while window := list(itertools.islice(items, size)):
        yield window


def _merge_segment_stats(total: SegmentStats, delta: SegmentStats) -> None:
    for name in SegmentStats.model_fields:
        running = getattr(total, name)
        if not isinstance(running, int):
            raise TypeError(
                f"SegmentStats.{name} is a {type(running).__name__}, not a counter; "
                "this merge only knows how to sum"
            )
        setattr(total, name, running + getattr(delta, name))


def _apply(counters: _TextCounters, result: _Result) -> None:
    counters.docs += len(result.lines)
    for style, count in result.styles.items():
        counters.styles[style] = counters.styles.get(style, 0) + count


class _PoolRunner:
    """A spawn pool that survives a task which never returns."""

    # ----------------NOTE-------------------------------------#
    # From testing, pysbd hung forever on bad doc. and regex held GIL,
    # nothing inside worker can interrupt
    # only way is wall clock & terminate ().....

    # prevent this with a timeout (see const att start of file)

    def __init__(self, workers: int, *, task_timeout: float) -> None:
        self.workers = workers
        self.task_timeout = task_timeout
        self.timeouts = 0
        self.forced_fallback_docs: list[str] = []
        self._ctx = multiprocessing.get_context("spawn")
        self._pool: Any = None

    def run(self, tasks: list[_Batch]) -> list[_Result]:
        """Segment every task, isolating any that overruns."""
        return self._drain(tasks, self.task_timeout, self._on_batch_timeout)

    def _on_batch_timeout(self, batch: _Batch) -> list[_Result]:
        """Count the expensive event, then go find which document caused it."""
        self.timeouts += 1
        return self._isolate(batch)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    def _ensure_pool(self) -> Any:
        if self._pool is None:
            # think this allows it to be identified on other OS
            self._pool = self._ctx.Pool(self.workers, initializer=_init_worker)
        return self._pool

    def _drain(
        self,
        tasks: list[_Batch],
        timeout: float,
        on_timeout: Callable[[_Batch], list[_Result]],
    ) -> list[_Result]:
        """Collect tasks in order, rebuilding the pool around any that hang."""
        out: list[_Result] = []
        pending = tasks
        while pending:
            pool = self._ensure_pool()
            handles = [pool.apply_async(_task, (batch,)) for batch in pending]
            for index, handle in enumerate(handles):
                try:
                    out.append(handle.get(timeout=timeout))
                except multiprocessing.TimeoutError:
                    self.close()
                    out.extend(on_timeout(pending[index]))
                    # Everything after the culprit died with the pool.
                    pending = pending[index + 1 :]
                    break
            else:
                pending = []
        return out

    def _isolate(self, batch: _Batch) -> list[_Result]:
        """Re-run one batch document by document, to find what actually hung."""
        singles = [_Batch(batch.source, [row]) for row in batch.rows]
        return self._drain(singles, ISOLATION_TIMEOUT_S, self._force_fallback)

    def _force_fallback(self, batch: _Batch) -> list[_Result]:
        """Segment without pysbd, in this process. Cannot hang, cannot be beaten."""
        for row in batch.rows:
            self.forced_fallback_docs.append(_row_label(row))
        return [_task(batch, use_pysbd=False)]


def _row_label(row: Any) -> str:
    """Best available identifier for a row, for the record of what fell back."""
    for attr in ("id", "doc_id"):
        value = getattr(row, attr, None)
        if value is not None:
            return str(value)
    return f"{getattr(row, 'split', '?')}:{getattr(row, 'row_index', '?')}"


def _run(
    source: str,
    rows: Iterable[Any],
    out_path: Path,
    counters: _TextCounters,
    *,
    workers: int,
    batch_size: int,
    progress: bool,
    task_timeout: float = TASK_TIMEOUT_S,
) -> tuple[int, int, SegmentStats, float, int, list[str]]:
    """Segment rows into out_path.

    Returns docs, bytes, stats, seconds, timed-out batches and the ids of any
    documents that had to skip pysbd.
    """
    batches = _batches(rows, source, batch_size)
    segment_stats = SegmentStats()
    prog = _Progress(source, progress)
    started = perf_counter()
    docs = written = timeouts = 0
    forced: list[str] = []

    def consume(fh: BinaryIO, result: _Result) -> None:
        nonlocal docs, written
        for line in result.lines:
            written += fh.write(line)
        docs += len(result.lines)
        _merge_segment_stats(segment_stats, result.segment_stats)
        _apply(counters, result)
        prog.update(docs)

    with _atomic_writer(out_path) as fh:
        if workers <= 1:
            # In-process, so a debugger works and tests can monkeypatch the
            # tokenizer -- a patch in the parent cannot reach a spawned child.
            _init_worker()
            for batch in batches:
                consume(fh, _task(batch))
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            runner = _PoolRunner(workers, task_timeout=task_timeout)
            try:
                for window in _windows(batches, workers * TASKS_PER_WORKER):
                    for result in runner.run(window):
                        consume(fh, result)
            finally:
                runner.close()
            timeouts = runner.timeouts
            forced = runner.forced_fallback_docs

    return docs, written, segment_stats, perf_counter() - started, timeouts, forced


def _result(
    source: str,
    out_path: Path,
    *,
    rows: int,
    docs: int,
    written: int,
    elapsed: float,
    workers: int,
    batch_size: int,
    integrity_ok: bool,
    is_green: bool,
    adapter_stats: dict[str, Any],
    segment_stats: SegmentStats,
    timeouts: int = 0,
    forced: list[str] | None = None,
) -> IngestResult:
    return IngestResult(
        source=source,
        path=out_path,
        rows=rows,
        docs=docs,
        bytes_written=written,
        elapsed_s=round(elapsed, 1),
        workers=workers,
        batch_size=batch_size,
        tokenizer_repo=TOKENIZER_REPO,
        integrity_ok=integrity_ok,
        is_green=is_green,
        segment_healthy=segment_stats.is_healthy,
        timed_out_batches=timeouts,
        forced_fallback_docs=forced or [],
        adapter_stats=adapter_stats,
        segment_stats=segment_stats.as_dict(),
    )


# -----------------------The three corpora------------------------------------------ #
# CORPORA SPECIFIC PROCESSING


def ingest_raid(
    clean_parquet: Path,
    out_dir: Path,
    *,
    workers: int | None = None,
    batch_size: int = BATCH_SIZE,
    progress: bool = False,
    check_integrity: bool = True,
) -> IngestResult:
    """Segment the derived clean RAID parquet."""
    n_workers = config.workers() if workers is None else workers

    if check_integrity:
        pre = raid.scan(clean_parquet)
        if not pre.integrity_ok:
            raise IntegrityGateError(f"RAID integrity check failed: {pre.as_dict()}")

    stats = raid.RaidStats()
    rows = _accounted(raid.load_rows(clean_parquet), raid.account, stats)
    out_path = out_dir / OUTPUT_NAMES["raid"]
    docs, written, segment_stats, elapsed, timeouts, forced = _run(
        "raid",
        rows,
        out_path,
        stats,
        workers=n_workers,
        batch_size=batch_size,
        progress=progress,
    )
    return _result(
        "raid",
        out_path,
        rows=stats.rows,
        docs=docs,
        written=written,
        elapsed=elapsed,
        workers=n_workers,
        batch_size=batch_size,
        integrity_ok=stats.integrity_ok,
        is_green=stats.is_green,
        adapter_stats=stats.as_dict(),
        segment_stats=segment_stats,
        timeouts=timeouts,
        forced=forced,
    )


def ingest_mage(
    directory: Path,
    out_dir: Path,
    *,
    splits: list[str] | None = None,
    workers: int | None = None,
    batch_size: int = BATCH_SIZE,
    progress: bool = False,
    check_integrity: bool = True,
) -> IngestResult:
    """Segment MAGE's five CSVs, in file order."""
    n_workers = config.workers() if workers is None else workers

    if check_integrity:
        pre = mage.scan(directory, splits=splits)
        if not pre.integrity_ok:
            raise IntegrityGateError(f"MAGE integrity check failed: {pre.as_dict()}")

    stats = mage.MageStats()
    rows = _accounted(mage.iter_rows(directory, splits), mage.account, stats)
    out_path = out_dir / OUTPUT_NAMES["mage"]
    docs, written, segment_stats, elapsed, timeouts, forced = _run(
        "mage",
        rows,
        out_path,
        stats,
        workers=n_workers,
        batch_size=batch_size,
        progress=progress,
    )
    return _result(
        "mage",
        out_path,
        rows=stats.rows,
        docs=docs,
        written=written,
        elapsed=elapsed,
        workers=n_workers,
        batch_size=batch_size,
        integrity_ok=stats.integrity_ok,
        is_green=stats.is_green,
        adapter_stats=stats.as_dict(),
        segment_stats=segment_stats,
        timeouts=timeouts,
        forced=forced,
    )


def ingest_seqxgpt(
    directory: Path,
    out_dir: Path,
    *,
    split_role: str = "calib_pool",
    progress: bool = False,
) -> IngestResult:
    """Segment SeqXGPT, sequentially and deliberately so."""
    stats = seqxgpt.SeqXGPTStats()
    segmenter = Segmenter()
    out_path = out_dir / OUTPUT_NAMES["seqxgpt"]
    prog = _Progress("seqxgpt", progress)
    started = perf_counter()
    docs = written = 0

    with _atomic_writer(out_path) as fh:
        for doc in seqxgpt.build_docs(directory, split_role, segmenter=segmenter, stats=stats):
            written += fh.write(doc_to_json(doc) + b"\n")
            docs += 1
            prog.update(docs)
    # check labels here
    quarantine_ok = stats.quarantined == 0 and stats.label_file_mismatches == 0
    return _result(
        "seqxgpt",
        out_path,
        rows=stats.records,
        docs=docs,
        written=written,
        elapsed=perf_counter() - started,
        workers=1,
        batch_size=0,
        integrity_ok=quarantine_ok,
        is_green=quarantine_ok and docs + stats.quarantined == stats.records,
        adapter_stats=stats.as_dict(),
        segment_stats=segmenter.stats,
    )


def ingest_all(
    *,
    raw_dir: Path | None = None,
    interim_dir: Path | None = None,
    out_dir: Path | None = None,
    workers: int | None = None,
    progress: bool = True,
) -> list[IngestResult]:
    """Segment all three corpora and write their JSONL and sidecars."""
    raw = raw_dir if raw_dir is not None else config.RAW_DIR
    interim = interim_dir if interim_dir is not None else config.INTERIM_DIR
    out = out_dir if out_dir is not None else config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    results = [
        ingest_seqxgpt(
            raw / "seqxgpt" / "bench",
            out,
            progress=progress,
        ),
        ingest_mage(
            raw / "mage",
            out,
            workers=workers,
            progress=progress,
        ),
        ingest_raid(
            interim / "raid" / CLEAN_FILE,
            out,
            workers=workers,
            progress=progress,
        ),
    ]
    for result in results:
        write_sidecar(result, out)
    return results


def write_sidecar(result: IngestResult, out_dir: Path) -> Path:
    """Write `{source}.stats.json` beside the JSONL, for report to format."""
    path = out_dir / f"{result.source}.stats.json"
    path.write_bytes(orjson.dumps(result.as_dict(), option=orjson.OPT_INDENT_2))
    return path
