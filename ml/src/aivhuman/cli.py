"""Data command line: acquire, derive, peek, ingest, verify, report, split."""

import argparse
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import orjson

from aivhuman import acquire, config
from aivhuman import report as report_mod
from aivhuman import splits as splits_mod
from aivhuman import verify as verify_mod
from aivhuman.ingest import ingest_mage, ingest_raid, ingest_seqxgpt, write_sidecar
from aivhuman.labels import mage_label, raid_label
from aivhuman.schema import label_name
from aivhuman.sources import mage, raid, seqxgpt
from aivhuman.sources.raid_parquet import CLEAN_FILE, derive

# Rows printed per source by `peek`, and the minimum of each awkward kind.
PEEK_ROWS = 20
PEEK_SEED = 20240501


def main(argv: Sequence[str] | None = None) -> int:
    config.configure_stdio()
    parser = _parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    result: int = handler(args)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aivhuman-data", description=__doc__)
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser("acquire", help="download the raw corpora")
    p.add_argument("--source", choices=["raid", "mage", "seqxgpt", "all"], default="all")
    p.set_defaults(handler=_acquire)

    p = subparsers.add_parser("derive", help="scan RAID's 11.8 GB CSV into parquet, once")
    p.add_argument("--no-attacks", action="store_true", help="skip the Phase 5 partitions")
    p.set_defaults(handler=_derive)

    p = subparsers.add_parser("peek", help="print stratified rows for a human to read")
    p.add_argument("--rows", type=int, default=PEEK_ROWS)
    p.add_argument("--chars", type=int, default=160)
    p.set_defaults(handler=_peek)

    p = subparsers.add_parser("ingest", help="segment the corpora into JSONL")
    p.add_argument("--source", choices=["raid", "mage", "seqxgpt", "all"], default="all")
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(handler=_ingest)

    p = subparsers.add_parser("verify", help="re-read the JSONL and re-assert every invariant")
    p.set_defaults(handler=_verify)

    p = subparsers.add_parser("report", help="write the Phase 1 report CSVs")
    p.add_argument("--skip-verify", action="store_true")
    p.set_defaults(handler=_report)

    p = subparsers.add_parser("split", help="write the grouped split manifests")
    p.set_defaults(handler=_split)

    return parser


# ----------------------------Commands--------------------------------------- #


def _acquire(args: argparse.Namespace) -> int:
    config.ensure_dirs()
    wanted = ["raid", "mage", "seqxgpt"] if args.source == "all" else [args.source]
    fetchers: dict[str, Callable[[], list[Path]]] = {
        "raid": acquire.fetch_raid,
        "mage": acquire.fetch_mage,
        "seqxgpt": acquire.fetch_seqxgpt,
    }
    for name in wanted:
        print(f"{name}:", flush=True)
        for path in fetchers[name]():
            print(f"  {path.name} ({path.stat().st_size / 1e6:,.1f} MB)", flush=True)
    return 0


def _derive(args: argparse.Namespace) -> int:
    config.ensure_dirs()
    out = config.INTERIM_DIR / "raid"
    stats = derive(
        config.RAW_DIR / "raid" / acquire.RAID_FILE,
        out,
        include_attacks=not args.no_attacks,
        progress=True,
    )
    print(orjson.dumps(stats.as_dict(), option=orjson.OPT_INDENT_2).decode(), flush=True)
    return 0


def _peek(args: argparse.Namespace) -> int:
    """Print rows a person can check the polarity against."""

    # ----------------- NOTE -------------------------------------
    #  Stratified on purpose. The quotas force the rows that actually distinguish a
    # correct polarity from a flipped one: human rows, and MAGE's paraphrased
    # human text, which upstream labels machine.

    rng = random.Random(PEEK_SEED)

    print("=" * 78)
    print("RAID — label is derived from the `model` column; there is no label column")
    print("=" * 78)
    # Quotas, not a plain sample: human rows are 2.9% of clean RAID, so an
    # unstratified sample of twenty can easily contain one or none -- and the
    # human rows are the only ones that can disprove a flipped polarity.
    picked = _stratified(
        raid.load_rows(config.INTERIM_DIR / "raid" / CLEAN_FILE),
        {"human": (lambda r: r.model == "human", 5)},
        default_quota=args.rows - 5,
        rng=rng,
    )
    for row in picked["human"] + picked["_other"]:
        label = raid_label(row.model)
        print(f"\n[{label_name(label)}] model={row.model} domain={row.domain}")
        print(f"  {_clip(row.generation, args.chars)}")

    print()
    print("=" * 78)
    print('MAGE — polarity is INVERTED: label "1" is human, "0" is machine')
    print("=" * 78)
    # The paraphrase rows are only ~ 0.37%
    picked = _stratified(
        mage.iter_rows(config.RAW_DIR / "mage"),
        {
            "paraphrased": (lambda r: r.src.endswith("_para"), 3),
            "human": (lambda r: r.src.endswith("_human"), 5),
        },
        default_quota=args.rows - 8,
        rng=rng,
    )
    for kind in ("human", "paraphrased", "_other"):
        for row in picked[kind]:
            label = mage_label(row.label)
            print(f"\n[{label_name(label)}] ({kind}) raw={row.label!r} src={row.src}")
            print(f"  {_clip(row.text, args.chars)}")

    print()
    print("=" * 78)
    print("SeqXGPT — `prompt_len` chars are human, the rest is machine")
    print("=" * 78)
    records = _reservoir(
        iter(seqxgpt.load_records(config.RAW_DIR / "seqxgpt" / "bench")),
        args.rows * 20,
        rng,
    )
    for record in records[: args.rows]:
        cut = record.prompt_len
        print(f"\n[{record.label_raw}] prompt_len={cut} file={record.file_stem}")
        if cut:
            print(f"  human:   {_clip(record.text[:cut], args.chars // 2)}")
            print(f"  machine: {_clip(record.text[cut:], args.chars // 2)}")
        else:
            print(f"  human:   {_clip(record.text, args.chars)}")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    config.ensure_dirs()

    wanted = ["seqxgpt", "mage", "raid"] if args.source == "all" else [args.source]
    out = config.PROCESSED_DIR
    for name in wanted:
        if name == "seqxgpt":
            result = ingest_seqxgpt(
                config.RAW_DIR / "seqxgpt" / "bench",
                out,
                progress=True,
            )
        elif name == "mage":
            result = ingest_mage(
                config.RAW_DIR / "mage",
                out,
                workers=args.workers,
                progress=True,
            )
        else:
            result = ingest_raid(
                config.INTERIM_DIR / "raid" / CLEAN_FILE,
                out,
                workers=args.workers,
                progress=True,
            )
        write_sidecar(result, out)
        rate = result.docs / result.elapsed_s if result.elapsed_s else 0.0
        print(
            f"{result.source:8} {result.docs:>7,} docs  {result.elapsed_s / 60:>5.1f} min"
            f"  {rate:>5.0f} docs/s  {result.bytes_written / 1e6:>7.1f} MB"
            f"  green={result.is_green}",
            flush=True,
        )
        if result.forced_fallback_docs:
            print(f"         pysbd skipped for: {result.forced_fallback_docs}", flush=True)
    return 0


def _verify(_args: argparse.Namespace) -> int:
    reports = verify_mod.verify_all(config.PROCESSED_DIR)
    if not reports:
        print(f"nothing to verify in {config.PROCESSED_DIR}", file=sys.stderr)
        return 2
    for report in reports:
        print(
            f"{report.path.name:15} docs={report.docs:>7,} spans={report.spans:>9,}"
            f" human={report.human_docs:>7,} machine={report.machine_docs:>7,}"
            f" {'ok' if report.ok else 'FAILED'}"
        )
    problems = list(verify_mod.iter_problems(reports))
    for problem in problems:
        print(f"  ! {problem}", file=sys.stderr)
    if problems:
        print(f"{sum(r.n_problems for r in reports)} problems", file=sys.stderr)
        return 1
    return 0


def _report(args: argparse.Namespace) -> int:
    verify_data: list[dict[str, Any]] | None = None
    if not args.skip_verify:
        verify_data = [r.as_dict() for r in verify_mod.verify_all(config.PROCESSED_DIR)]
    path = report_mod.build(config.PROCESSED_DIR, config.REPORTS_DIR, verify=verify_data)
    print(f"wrote {path}")
    return 0


def _split(_args: argparse.Namespace) -> int:
    stats = splits_mod.build(config.PROCESSED_DIR, config.MANIFESTS_DIR, config.SPLITS_REPORT)
    for name, n in stats.docs.items():
        print(
            f"{name:>14}: {n:>8,} docs {stats.groups[name]:>8,} groups "
            f"{stats.human[name]:>7,} human {stats.machine[name]:>8,} machine"
        )
    for reason, n in stats.dropped.items():
        print(f"  dropped {reason}: {n:,}")
    print(f"wrote {config.MANIFESTS_DIR} and {config.SPLITS_REPORT}")
    return 0


# --------------------------------Helpers----------------------------------- #


def _stratified(
    rows: Any,
    quotas: dict[str, tuple[Callable[[Any], bool], int]],
    *,
    default_quota: int,
    rng: random.Random,
) -> dict[str, list[Any]]:
    """One streaming pass, one reservoir per category, quotas guaranteed."""
    reservoirs: dict[str, list[Any]] = {name: [] for name in quotas}
    reservoirs["_other"] = []
    seen: dict[str, int] = dict.fromkeys(reservoirs, 0)

    for row in rows:
        name = "_other"
        for candidate, (predicate, _quota) in quotas.items():
            if predicate(row):
                name = candidate
                break
        limit = quotas[name][1] if name in quotas else default_quota
        bucket = reservoirs[name]
        if len(bucket) < limit:
            bucket.append(row)
        else:
            j = rng.randrange(seen[name] + 1)
            if j < limit:
                bucket[j] = row
        seen[name] += 1

    for bucket in reservoirs.values():
        rng.shuffle(bucket)
    return reservoirs


def _reservoir(rows: Any, k: int, rng: random.Random) -> list[Any]:
    """Reservoir-sample `k` rows in one streaming pass."""
    out: list[Any] = []
    for i, row in enumerate(rows):
        if i < k:
            out.append(row)
        else:
            j = rng.randrange(i + 1)
            if j < k:
                out[j] = row
    rng.shuffle(out)
    return out


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}..."


if __name__ == "__main__":
    # --------------IMPORTANT -----------------------
    # Required, not decoration: the ingest pool spawns children that re-import
    raise SystemExit(main())
