"""segment the corpora into jsonl, parallel where it makes sense"""

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
from aivhuman.sources import daigt, mage, raid, raid_attacks, seqxgpt
from aivhuman.text.segment import Segmenter, SegmentStats
from aivhuman.text.tokens import TOKENIZER_REPO, tokenizer

# NEEDED TO ADD THIS TO PREVENT PICKLING OVERHEAD >:( s
BATCH_SIZE: Final = 256

# NOTE (i learnt this the hard way, this was painful)
# tasks per worker per window. not using imap BECAUSE imap drains its input ASAP
# which pulls all 468k rows into memory as pending tasks :))))
# window caps it at workers x TASKS_PER_WORKER x BATCH_SIZE docs, costs a bit of
# idle time at each window boundary
TASKS_PER_WORKER: Final = 4

PROGRESS_EVERY: Final = 20_000


# wall clock per batch before we assume the worker hung.
# from benchmarking a 256 doc batch takes a few seconds....

# NOTE this is not a performance knob, it actually makes the diff. betw. corpus that finishes
# or not, so it is set with two orders of magnitude of headroom
TASK_TIMEOUT_S: Final = 120.0

# per doc, when isolating a hung batch
ISOLATION_TIMEOUT_S: Final = 15.0

OUTPUT_NAMES: Final = {
    "raid": "raid.jsonl",
    "mage": "mage.jsonl",
    "seqxgpt": "seqxgpt.jsonl",
    "daigt": "daigt.jsonl",
    "raid-attacks": "raid-attacks.jsonl",
}

# written during a pass, renamed on success so trunc. runs dont look finished
PARTIAL_SUFFIX: Final = ".partial"


class IngestGateError(RuntimeError):
    """precondition failed, nothing written"""


class IntegrityGateError(IngestGateError):
    """upstream file failed its integrity scan (bad labels, missing cols)"""


class _TextCounters(Protocol):
    docs: int


class IngestResult(BaseModel):
    # what goes in the sidecar next to the jsonl
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
    # docs that had to skip pysbd. keep the ids so i can look them up (2 in RAID)
    forced_fallback_docs: list[str] = []

    adapter_stats: dict[str, Any]
    segment_stats: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# worker side


_SEG: Segmenter | None = None
_FALLBACK_SEG: Segmenter | None = None

# only pure row -> doc builders can go in a worker
_BUILDERS: Final[dict[str, Callable[[Any, Segmenter], Doc | None]]] = {
    "mage": mage.to_doc,
    "daigt": daigt.to_doc,
    "raid": raid.to_doc,
    "raid-attacks": raid_attacks.to_doc,
}


class _Batch(NamedTuple):
    source: str
    rows: list[Any]


class _Result(NamedTuple):
    lines: list[bytes]
    segment_stats: SegmentStats


def _segmenter() -> Segmenter:
    # one per process, slow to build + not thread safe
    global _SEG
    if _SEG is None:
        _SEG = Segmenter()
    return _SEG


def _fallback_segmenter() -> Segmenter:
    global _FALLBACK_SEG
    if _FALLBACK_SEG is None:
        _FALLBACK_SEG = Segmenter(use_pysbd=False)
    return _FALLBACK_SEG


def _init_worker() -> None:
    # warm up segmenter + tokenizer once per worker, not per batch
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _segmenter()
    tokenizer()


def _task(batch: _Batch, use_pysbd: bool = True) -> _Result:
    # runs in a worker. returns serialised lines not Docs, bytes pickle way cheaper
    seg = _segmenter() if use_pysbd else _fallback_segmenter()
    # fresh stats per batch, parent sums them
    seg.stats = SegmentStats()
    build = _BUILDERS[batch.source]

    lines: list[bytes] = []
    for row in batch.rows:
        doc = build(row, seg)
        if doc is None:
            continue
        lines.append(doc_to_json(doc) + b"\n")
    return _Result(lines, seg.stats)


# parent side


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


# counts rows into the adapter stats as they stream past, no second pass over the data
def _accounted(
    rows: Iterable[Any], account: Callable[[Any, Any], Any], stats: Any
) -> Iterator[Any]:
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
    # pulls at most `size` batches at a time, this is what bounds memory
    while window := list(itertools.islice(items, size)):
        yield window


def _merge_segment_stats(total: SegmentStats, delta: SegmentStats) -> None:
    # sum every counter, raises if a non counter field ever gets added
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


class _PoolRunner:
    """spawn pool that survives a task that never returns"""

    # NOTE from testing, pysbd hung forever on bad doc. and regex held GIL,
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
        return self._drain(tasks, self.task_timeout, self._on_batch_timeout)

    def _on_batch_timeout(self, batch: _Batch) -> list[_Result]:
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
        out: list[_Result] = []
        pending = tasks
        while pending:
            pool = self._ensure_pool()
            # results collected in submit order so the output jsonl order is stable
            handles = [pool.apply_async(_task, (batch,)) for batch in pending]
            for index, handle in enumerate(handles):
                try:
                    out.append(handle.get(timeout=timeout))
                except multiprocessing.TimeoutError:
                    # cant cancel one task, kill the whole pool and respawn
                    self.close()
                    out.extend(on_timeout(pending[index]))
                    # everything after the bad one died w/ the pool
                    pending = pending[index + 1 :]
                    break
            else:
                pending = []
        return out

    def _isolate(self, batch: _Batch) -> list[_Result]:
        # rerun doc by doc to find the one that hung
        singles = [_Batch(batch.source, [row]) for row in batch.rows]
        return self._drain(singles, ISOLATION_TIMEOUT_S, self._force_fallback)

    def _force_fallback(self, batch: _Batch) -> list[_Result]:
        # no pysbd, in process, cant hang
        for row in batch.rows:
            self.forced_fallback_docs.append(_row_label(row))
        return [_task(batch, use_pysbd=False)]


def _row_label(row: Any) -> str:
    # best effort id for logging a doc that had to skip pysbd
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
    """returns docs, bytes, stats, secs, timed out batches, forced fallback ids"""
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
            # in process so the debugger works + tests can monkeypatch the
            # tokenizer (patches dont reach spawned children)
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


# packs the run outputs into the sidecar model
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
    """clean RAID parquet -> raid.jsonl"""
    n_workers = config.workers() if workers is None else workers

    # cheap scan first, refuse to spend an hour ingesting a broken file
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


def ingest_raid_attacks(
    rows: Iterable[raid.RawRow],
    out_dir: Path,
    *,
    workers: int | None = None,
    batch_size: int = BATCH_SIZE,
    progress: bool = False,
) -> IngestResult:
    """attacked RAID rows -> raid-attacks.jsonl"""
    # no integrity gate here, the attacks are the whole point
    n_workers = config.workers() if workers is None else workers
    stats = raid.RaidStats()
    out_path = out_dir / OUTPUT_NAMES["raid-attacks"]
    docs, written, segment_stats, elapsed, timeouts, forced = _run(
        "raid-attacks",
        _accounted(rows, raid.account, stats),
        out_path,
        stats,
        workers=n_workers,
        batch_size=batch_size,
        progress=progress,
    )
    return _result(
        "raid-attacks",
        out_path,
        rows=stats.rows,
        docs=docs,
        written=written,
        elapsed=elapsed,
        workers=n_workers,
        batch_size=batch_size,
        integrity_ok=True,
        is_green=segment_stats.is_healthy,
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
    """MAGE csvs -> mage.jsonl"""
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


def ingest_daigt(
    path: Path,
    out_dir: Path,
    *,
    workers: int | None = None,
    batch_size: int = BATCH_SIZE,
    progress: bool = False,
) -> IngestResult:
    """DAIGT v2 csv -> daigt.jsonl"""
    n_workers = config.workers() if workers is None else workers

    pre = daigt.scan(path)
    if not pre.integrity_ok:
        raise IntegrityGateError(f"DAIGT integrity check failed: {pre.as_dict()}")

    stats = daigt.DaigtStats()
    rows = _accounted(daigt.iter_rows(path), daigt.account, stats)
    out_path = out_dir / OUTPUT_NAMES["daigt"]
    docs, written, segment_stats, elapsed, timeouts, forced = _run(
        "daigt",
        rows,
        out_path,
        stats,
        workers=n_workers,
        batch_size=batch_size,
        progress=progress,
    )
    return _result(
        "daigt",
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
    """SeqXGPT jsonl files -> seqxgpt.jsonl"""
    # sequential, no pool. small enough + group recovery needs every record at once
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
    # check labels here, quarantined = boundary or label file didnt line up
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


def write_sidecar(result: IngestResult, out_dir: Path) -> Path:
    """{source}.stats.json, report.py and verify.py read these"""
    path = out_dir / f"{result.source}.stats.json"
    path.write_bytes(orjson.dumps(result.as_dict(), option=orjson.OPT_INDENT_2))
    return path
