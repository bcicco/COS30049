"""data cli: acquire, derive, peek, ingest, verify, report, split, attacks"""

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
from aivhuman.ingest import (
    ingest_daigt,
    ingest_mage,
    ingest_raid,
    ingest_raid_attacks,
    ingest_seqxgpt,
    write_sidecar,
)
from aivhuman.labels import mage_label, raid_label
from aivhuman.schema import label_name
from aivhuman.sources import mage, raid, seqxgpt
from aivhuman.sources.raid_parquet import CLEAN_FILE, derive

# rows per source for peek
PEEK_ROWS = 20
PEEK_SEED = 20240501


def main(argv: Sequence[str] | None = None) -> int:
    """entry point for aivhuman-data"""
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

    # download raw corpora into data/raw
    p = subparsers.add_parser("acquire")
    p.add_argument("--source", choices=["raid", "mage", "seqxgpt", "daigt", "all"], default="all")
    p.set_defaults(handler=_acquire)

    # stream the raid csv into parquet (clean rows + one partition per attack)
    p = subparsers.add_parser("derive")
    p.add_argument("--no-attacks", action="store_true")
    p.set_defaults(handler=_derive)

    # print sample rows from each source to sanity check label polarity
    p = subparsers.add_parser("peek")
    p.add_argument("--rows", type=int, default=PEEK_ROWS)
    p.add_argument("--chars", type=int, default=160)
    p.set_defaults(handler=_peek)

    # normalise + segment each source into data/processed/*.jsonl
    p = subparsers.add_parser("ingest")
    p.add_argument("--source", choices=["raid", "mage", "seqxgpt", "daigt", "all"], default="all")
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(handler=_ingest)

    # re-check offsets and labels of every processed doc
    p = subparsers.add_parser("verify")
    p.set_defaults(handler=_verify)

    p = subparsers.add_parser("report")
    p.add_argument("--skip-verify", action="store_true")
    p.set_defaults(handler=_report)

    # assign groups to splits, writes manifests/
    p = subparsers.add_parser("split")
    p.set_defaults(handler=_split)

    # ingest attacked raid docs whose clean parent is in a split
    p = subparsers.add_parser("attacks")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)  # small smoke run into data/interim
    p.set_defaults(handler=_attacks)

    return parser


def _acquire(args: argparse.Namespace) -> int:
    """fetch the requested sources and print what landed on disk"""
    config.ensure_dirs()
    wanted = ["raid", "mage", "seqxgpt", "daigt"] if args.source == "all" else [args.source]
    fetchers: dict[str, Callable[[], list[Path]]] = {
        "raid": acquire.fetch_raid,
        "mage": acquire.fetch_mage,
        "seqxgpt": acquire.fetch_seqxgpt,
        "daigt": acquire.fetch_daigt,
    }
    for name in wanted:
        print(f"{name}:", flush=True)
        for path in fetchers[name]():
            print(f"  {path.name} ({path.stat().st_size / 1e6:,.1f} MB)", flush=True)
    return 0


def _derive(args: argparse.Namespace) -> int:
    """raid csv -> parquet in data/interim/raid"""
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
    """print rows to eyeball the polarity against"""

    # NOTE stratified so we always get the rows that would show a flipped polarity:
    # human rows + MAGE paraphrased human text (which upstream labels machine)

    rng = random.Random(PEEK_SEED)

    print("=" * 78)
    print("RAID - label comes from the `model` column, there is no label column")
    print("=" * 78)
    # human rows are only 2.9% of clean RAID, a plain sample of 20 often has 0 or 1
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
    print('MAGE - polarity is INVERTED: label "1" is human, "0" is machine')
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
    print("SeqXGPT - first `prompt_len` chars are human, rest is machine")
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
    """normalise and segment each source, one jsonl per corpus + a stats sidecar"""
    config.ensure_dirs()

    wanted = ["seqxgpt", "mage", "raid", "daigt"] if args.source == "all" else [args.source]
    out = config.PROCESSED_DIR
    # raid reads the derived parquet, the others read the raw downloads
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
        elif name == "daigt":
            result = ingest_daigt(
                config.RAW_DIR / "daigt" / acquire.DAIGT_FILE,
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
        # segment stats saved next to the corpus, green = no text left outside a span
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
    """re-validate processed docs, non zero exit if anything fails"""
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
    """dataset summary (counts, balance, segment health) into reports/"""
    verify_data: list[dict[str, Any]] | None = None
    if not args.skip_verify:
        verify_data = [r.as_dict() for r in verify_mod.verify_all(config.PROCESSED_DIR)]
    path = report_mod.build(config.PROCESSED_DIR, config.REPORTS_DIR, verify=verify_data)
    print(f"wrote {path}")
    return 0


def _split(_args: argparse.Namespace) -> int:
    """build group disjoint split manifests and print their sizes"""
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


def _attacks(args: argparse.Namespace) -> int:
    """ingest attacked raid docs and write their manifests"""
    from aivhuman.evaluate import load_manifest
    from aivhuman.sources import raid_attacks

    # attacked docs inherit the group of their clean parent, so they stay in the same split
    groups = {
        parent: load_manifest(config.MANIFESTS_DIR, parent)
        for parent in raid_attacks.ADV_PARENT.values()
    }
    wanted = raid_attacks.select(config.FEATURES_DIR, groups, args.limit)
    smoke = config.INTERIM_DIR / "attacks-smoke"
    out = smoke if args.limit else config.PROCESSED_DIR
    manifest_dir = smoke if args.limit else config.MANIFESTS_DIR / "attacks"
    manifests: dict[str, dict[str, str]] = {adv: {} for adv in raid_attacks.ADV_PARENT}

    # records each row into its manifest as it streams past into ingest
    def tapped() -> Any:
        for row in raid_attacks.load_rows(config.INTERIM_DIR / "raid" / "by_attack", wanted):
            adv, group = wanted[(row.adv_source_id, row.attack)]
            manifests[adv][f"raid:{row.id}"] = group
            yield row

    out.mkdir(parents=True, exist_ok=True)
    result = ingest_raid_attacks(tapped(), out, workers=args.workers, progress=True)
    write_sidecar(result, out)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for adv, manifest in manifests.items():
        path = manifest_dir / f"{adv}.json"
        path.write_bytes(orjson.dumps(dict(sorted(manifest.items())), option=orjson.OPT_INDENT_2))
        print(f"{adv:>13}: {len(manifest):>7,} docs  -> {path}")
    # requested (source, attack) pairs that raid doesnt actually have
    missing = len(wanted) - sum(len(m) for m in manifests.values())
    print(
        f"{result.docs:,} docs in {result.elapsed_s / 60:.1f} min; "
        f"{missing:,} requested rows missing"
    )
    return 0


def _stratified(
    rows: Any,
    quotas: dict[str, tuple[Callable[[Any], bool], int]],
    *,
    default_quota: int,
    rng: random.Random,
) -> dict[str, list[Any]]:
    """reservoir sample with a fixed quota per category, rest goes to _other"""
    # one pass, a reservoir per category
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
    """uniform sample of k rows in one pass (algorithm R)"""
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
    # collapse whitespace so one row prints on one line
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}..."


if __name__ == "__main__":
    # IMPORTANT dont remove, the ingest pool spawns children that re-import
    raise SystemExit(main())
