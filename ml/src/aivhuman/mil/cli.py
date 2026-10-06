"""MIL command line: extract, splice, vet, train, predict, evaluate, faithfulness, robustness,
calibrate, cluster."""

import argparse
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aivhuman import config
from aivhuman.evaluate import SPLIT_SOURCE

FEATURE_SPLITS = tuple(SPLIT_SOURCE)
EVAL_MODELS = ("tfidf-lr", "modernbert-doc", "mil-linear", "mil-gam")
REPORT_BASELINES = ("tfidf-lr", "modernbert-doc", "mil-gam")


def main(argv: Sequence[str] | None = None) -> int:
    config.configure_stdio()
    parser = argparse.ArgumentParser(prog="aivhuman-mil", description=__doc__)
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser("extract")
    p.add_argument("--split", nargs="+", default=list(FEATURE_SPLITS), choices=FEATURE_SPLITS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--token-budget", type=int, default=2048)
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(handler=_extract)

    p = subparsers.add_parser("splice")
    p.add_argument("--adv", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--token-budget", type=int, default=2048)
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(handler=_splice)

    p = subparsers.add_parser("vet")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(handler=_vet)

    p = subparsers.add_parser("train")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--mix",
        choices=("base", "spliced", "all", "spliced-adv"),
        default="base",
    )
    p.add_argument(
        "--sentence-weight",
        type=float,
        nargs="+",
        default=[],
    )
    p.add_argument("--crf", action="store_true")
    p.add_argument("--kept", action="store_true")
    p.add_argument("--crf-lr", type=float, default=1e-2)
    p.add_argument("--run", default=None)
    p.set_defaults(handler=_train)

    p = subparsers.add_parser("predict")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run", default=None)
    p.set_defaults(handler=_predict)

    p = subparsers.add_parser("evaluate")
    p.add_argument("--models", nargs="+", default=list(EVAL_MODELS))
    p.add_argument("--report-dir", type=Path, default=config.MIL_REPORT_DIR)
    p.set_defaults(handler=_evaluate)

    p = subparsers.add_parser("faithfulness")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run", default=None)
    p.set_defaults(handler=_faithfulness)

    p = subparsers.add_parser("robustness")
    p.add_argument("--runs", nargs="+", default=["mil"])
    p.set_defaults(handler=_robustness)

    p = subparsers.add_parser("calibrate")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run", default=None)
    p.set_defaults(handler=_calibrate)

    p = subparsers.add_parser("cluster")
    p.add_argument("--split", default="dev")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run", default=None)
    p.set_defaults(handler=_cluster)

    p = subparsers.add_parser("report")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run", default=None)
    p.add_argument("--baselines", nargs="+", default=list(REPORT_BASELINES))
    p.set_defaults(handler=_report)

    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    result: int = handler(args)
    return result


def _extract(args: argparse.Namespace) -> int:
    from multiprocessing import Pool

    import torch

    from aivhuman.features import extract, lm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = lm.ReferenceLM(device, token_budget=args.token_budget)
    out_dir = config.FEATURES_DIR / ("smoke" if args.limit else "")
    with Pool(args.workers or config.workers(), initializer=extract.init_worker) as pool:
        for split in args.split:
            _extract_split(split, args.limit, out_dir, ref, pool)
    return 0


def _splice(args: argparse.Namespace) -> int:
    from multiprocessing import Pool

    import torch

    from aivhuman.features import extract, lm, splice
    from aivhuman.features.load import load_span_docs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = lm.ReferenceLM(device, token_budget=args.token_budget)
    out_dir = config.FEATURES_DIR / ("smoke" if args.limit else "")
    with Pool(args.workers or config.workers(), initializer=extract.init_worker) as pool:
        for split in ("train-adv", "dev-adv") if args.adv else ("train", "dev"):
            path = out_dir / f"{split}-spliced.parquet"
            if path.exists():
                print(f"{split}-spliced: exists, skipping", flush=True)
                continue
            docs = splice.build(load_span_docs(config.PROCESSED_DIR, config.MANIFESTS_DIR, split))
            if args.limit:
                docs = random.Random(0).sample(docs, min(args.limit, len(docs)))
            print(f"{split}-spliced: {len(docs):,} docs", flush=True)
            rows = extract.extract(docs, path, ref, pool)
            print(f"{split}-spliced: {rows:,} spans", flush=True)
    return 0


def _extract_split(split: str, limit: int | None, out_dir: Path, ref: Any, pool: Any) -> None:
    from aivhuman.features import extract
    from aivhuman.features.load import load_span_docs

    path = out_dir / f"{split}.parquet"
    if path.exists():
        print(f"{split}: exists, skipping", flush=True)
        return
    start = time.time()
    docs = load_span_docs(config.PROCESSED_DIR, config.MANIFESTS_DIR, split)
    if limit:
        docs = random.Random(0).sample(docs, min(limit, len(docs)))
    print(f"{split}: {len(docs):,} docs loaded in {time.time() - start:.0f}s", flush=True)
    rows = extract.extract(docs, path, ref, pool)
    print(f"{split}: {rows:,} spans in {time.time() - start:.0f}s", flush=True)


def _vet(args: argparse.Namespace) -> int:
    from aivhuman.features import vet

    features, _, _, reports = _dirs(args.smoke)
    results = vet.vet(features)
    path = vet.write_report(results, reports / "feature_vetting.json")
    for r in results:
        print(f"{r.name:>20}  {', '.join(r.flags) or '-'}")
    print(f"wrote {path}")
    return 0


def _dirs(smoke: bool, run: str | None = None) -> tuple[Path, Path, Path, Path]:
    """Features, predictions root, checkpoint and report locations.

    A named run gets its own checkpoint and a report folder under the robustness reports;
    otherwise the checkpoint is `mil/model.pt` and reports go to the MIL report directory.
    """
    sub = "smoke" if smoke else ""
    reports = config.ROBUSTNESS_REPORT_DIR / sub / run if run else config.MIL_REPORT_DIR / sub
    return (
        config.FEATURES_DIR / sub,
        config.PREDICTIONS_DIR / sub,
        config.CHECKPOINTS_DIR / sub / (run or "mil") / "model.pt",
        reports,
    )


def _training_bags(features: Path, mix: str) -> tuple[Any, Any, Any, Any]:
    """Standardised train, dev and sentence-validation bags for a training mix, and the
    standardizer, which is always fitted on clean train."""
    from aivhuman.features import EXCLUDED, FEATURE_NAMES
    from aivhuman.mil.data import Standardizer, load_bags

    names = [n for n in FEATURE_NAMES if n not in EXCLUDED]
    train_bags = load_bags(features / "train.parquet", names)
    std = Standardizer.fit(train_bags.x, names)
    train_bags = train_bags.standardised(std)
    if mix != "base":
        spliced = load_bags(features / "train-spliced.parquet", names).standardised(std)
        if mix != "all":
            train_bags = train_bags.subset(train_bags.labels == 0)
        train_bags = train_bags.concat(spliced)
    if mix == "spliced-adv":
        adv = load_bags(features / "train-adv.parquet", names).standardised(std)
        adv_spliced = load_bags(features / "train-adv-spliced.parquet", names).standardised(std)
        train_bags = train_bags.concat(adv.subset(adv.labels == 0)).concat(adv_spliced)
    dev_bags = load_bags(features / "dev.parquet", names).standardised(std)
    sent_bags = _calibration_slice(features, std, validation=True)
    return train_bags, dev_bags, sent_bags, std


def _train(args: argparse.Namespace) -> int:
    from aivhuman.mil import predict, train
    from aivhuman.mil.data import load_bags

    features, _, checkpoint, reports = _dirs(args.smoke, args.run)
    train_bags, dev_bags, sent_bags, std = _training_bags(features, args.mix)
    names = std.names
    print(
        f"train {len(train_bags):,} docs, dev {len(dev_bags):,}, "
        f"sentence validation {len(sent_bags):,}, {len(names)} features"
    )

    best = None
    results = []
    if args.kept:
        configs = [train.KEPT.model_copy(update={"crf": args.crf, "crf_lr": args.crf_lr})]
    else:
        configs = train.sweep_configs(tuple(args.sentence_weight), crf=args.crf)
    for cfg in configs:
        model, result = train.fit(cfg, train_bags, dev_bags, sent_bags)
        results.append(result)
        shape = f"tau={cfg.tau:g}" if cfg.pooling == "lse" else f"k={cfg.k}"
        if cfg.sentence_weight:
            shape += f" sw={cfg.sentence_weight:g}"
        if cfg.crf:
            shape += f" crf lr={cfg.crf_lr:g}"
        print(
            f"  {cfg.head} {cfg.pooling} {shape} l1={cfg.l1:g}: select {result.selection:.4f}"
            f" (doc pAUC {result.dev_pauc:.4f}, sent AUROC {result.sentence_auroc:.4f},"
            f" TPR {result.dev_tpr:.4f})"
            f" (epoch {result.best_epoch}, {result.seconds:.0f}s)",
            flush=True,
        )
        if best is None or result.selection > best[1].selection:
            best = (model, result)
    assert best is not None
    dev_spliced = features / "dev-spliced.parquet"
    raid_sent = None
    if dev_spliced.exists():
        raid_sent = train.sentence_auroc(best[0], load_bags(dev_spliced, names).standardised(std))
        print(f"best: sentence AUROC on dev-spliced (RAID) {raid_sent:.4f}", flush=True)
    train.save(checkpoint, best[0], std, best[1])
    _sweep_report(results, reports / "sweep.json", raid_sent)
    print(f"wrote {predict.weights_report(best[0], std, best[1], reports / 'weights.json')}")
    return 0


def _sweep_report(results: list[Any], path: Path, raid_sent: float | None) -> None:
    """Every sweep result, best selection first, and the selected model's RAID sentence AUROC."""
    import orjson

    report = {
        "results": [r.model_dump() for r in sorted(results, key=lambda r: -r.selection)],
        "raid_sentence_auroc": raid_sent,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))


def _calibration_slice(features: Path, std: Any, validation: bool) -> Any:
    """seqxgpt-calib bags: the sentence-validation slice, or the calibration remainder."""
    from aivhuman.evaluate import load_manifest
    from aivhuman.mil.data import in_sentence_validation, load_bags

    bags = load_bags(features / "seqxgpt-calib.parquet", std.names).standardised(std)
    groups = load_manifest(config.MANIFESTS_DIR, "seqxgpt-calib")
    in_val = in_sentence_validation(bags.doc_ids, groups)
    return bags.subset(in_val if validation else ~in_val)


def _predict(args: argparse.Namespace) -> int:
    from aivhuman.mil import predict, train

    features, predictions, checkpoint, _ = _dirs(args.smoke, args.run)
    model, std = train.load(checkpoint)
    out = predictions / (args.run or model.name)
    for split in FEATURE_SPLITS:
        path = features / f"{split}.parquet"
        if not split.startswith("train") and path.exists():
            predict.predict_split(model, std, path, out, split)
            print(f"  scored {split}", flush=True)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from aivhuman import evaluate as ev

    splits = ev.load_splits(config.PROCESSED_DIR, config.MANIFESTS_DIR, ev.EVAL_SPLITS)
    metrics = [
        m for name in args.models for m in ev.evaluate_model(name, config.PREDICTIONS_DIR, splits)
    ]
    path = ev.write_report(metrics, args.report_dir, stem="metrics")
    for m in metrics:
        print(f"{m.model:>15} {m.split:>10}  TPR@1%FPR {m.tpr_at_1pct_fpr:.4f}  {m.auroc:.4f}")
    print(f"wrote {path}")
    return 0


def _faithfulness(args: argparse.Namespace) -> int:
    from aivhuman.mil import predict, train

    features, _, checkpoint, reports = _dirs(args.smoke, args.run)
    model, std = train.load(checkpoint)
    calib = _calibration_slice(features, std, validation=False)
    path = predict.faithfulness_report(model, calib, std.names, reports / "faithfulness.json")
    print(path.read_text(encoding="utf-8"))
    return 0


def _robustness(args: argparse.Namespace) -> int:
    from aivhuman.mil import robustness, train

    features = config.FEATURES_DIR
    runs = {}
    for run in args.runs:
        model, std = train.load(config.CHECKPOINTS_DIR / run / "model.pt")
        runs[run] = robustness.Scorer(model, std, features)
    raid = robustness.raid_contrasts(config.PROCESSED_DIR, config.MANIFESTS_DIR)
    mage, commentary = robustness.mage_contrasts(config.PROCESSED_DIR, config.MANIFESTS_DIR)
    train_bags, dev_bags, sent_bags, std = _training_bags(features, "spliced")
    groups = robustness.group_only_models(train_bags, dev_bags, sent_bags, std.names)
    path = config.ROBUSTNESS_REPORT_DIR / "robustness.json"
    print(f"wrote {robustness.report(runs, raid, mage, commentary, groups, path)}")
    return 0


def _sentence_splits(features: Path, model: Any, std: Any) -> tuple[Any, Any, Any]:
    """Labelled spans of seqxgpt-calib outside the selection slice, of seqxgpt-test, and of
    dev-spliced if its features exist."""
    import pyarrow.parquet as pq

    from aivhuman.evaluate import load_manifest
    from aivhuman.mil import calibrate
    from aivhuman.mil.data import in_sentence_validation

    calib_path = features / "seqxgpt-calib.parquet"
    calib_groups = load_manifest(config.MANIFESTS_DIR, "seqxgpt-calib")
    doc_ids = sorted(set(pq.read_table(calib_path, columns=["doc_id"])["doc_id"].to_pylist()))
    in_val = in_sentence_validation(doc_ids, calib_groups)
    keep = [d for d, v in zip(doc_ids, in_val, strict=True) if not v]
    fit = calibrate.load_spans(
        "seqxgpt-calib", calib_path, model, std, calib_groups.__getitem__, keep
    )
    test_groups = load_manifest(config.MANIFESTS_DIR, "seqxgpt-test")
    test = calibrate.load_spans(
        "seqxgpt-test",
        features / "seqxgpt-test.parquet",
        model,
        std,
        test_groups.__getitem__,
    )
    path = features / "dev-spliced.parquet"
    dev = (
        calibrate.load_spans("dev-spliced", path, model, std, _splice_group)
        if path.exists()
        else None
    )
    return fit, test, dev


def _calibrate(args: argparse.Namespace) -> int:
    from aivhuman.mil import calibrate, train

    features, _, checkpoint, _ = _dirs(args.smoke, args.run)
    reports = config.CALIBRATION_REPORT_DIR / ("smoke" if args.smoke else "")
    model, std = train.load(checkpoint)

    fit, test, dev_spliced = _sentence_splits(features, model, std)
    cal = calibrate.Calibrator.fit(fit.logits, fit.n_tokens, fit.labels)
    cal.save(checkpoint.parent / "calibrator.json")
    print(f"fitted on {len(fit.labels):,} spans: {cal.n_fit}", flush=True)

    splits = [test] if dev_spliced is None else [test, dev_spliced]
    cells = [c for s in splits for c in calibrate.evaluate(cal, s)]
    style_cells = calibrate.evaluate(cal, test, by_style=True)
    short = calibrate.bucket_of(test.n_tokens, cal.edges) == 0
    logits, n = test.logits[short], test.n_tokens[short]
    moved = int((cal.apply(logits, n) != cal.apply(logits, n, cap=False)).sum())
    path = calibrate.report(cal, cells, style_cells, (moved, int(short.sum())), reports)
    for c in cells:
        print(f"{c.split:>13} {c.bucket:>6}  ECE {c.ece:.4f}  AUROC {c.auroc_raw:.4f}")
    print(f"wrote {path}")
    return 0


def _report(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    from aivhuman import evaluate as ev
    from aivhuman.mil import calibrate, report, sentences, train

    features, predictions, checkpoint, _ = _dirs(args.smoke, args.run)
    out_dir = config.EVALUATION_REPORT_DIR / ("smoke" if args.smoke else "")
    model, std = train.load(checkpoint)
    run = args.run or model.name
    cal = calibrate.Calibrator.load(checkpoint.parent / "calibrator.json")

    splits = ev.load_splits(config.PROCESSED_DIR, config.MANIFESTS_DIR, report.DOC_SPLITS)
    splits["mage-para"], commentary = ev.drop_commentary(splits["mage-para"])
    doc_metrics = []
    for name in [run, *args.baselines]:
        for split, docs in splits.items():
            path = predictions / name / f"{split}.parquet"
            if not path.exists():
                continue
            preds = ev.read_predictions(path)
            if args.smoke:
                docs = [d for d in docs if d.doc_id in preds]
            doc_metrics.append(ev.compute_metrics(name, split, docs, preds))
            print(f"  scored {name} {split}", flush=True)

    calib, test, dev_spliced = _sentence_splits(features, model, std)
    calib_probs = cal.apply(calib.logits, calib.n_tokens)
    thresholds = {"p = 0.5": sentences.EVEN_THRESHOLD} | {
        f"FPR {f:.0%}": sentences.threshold_at_fpr(calib.labels, calib_probs, f)
        for f in sentences.OPERATING_FPRS
    }
    test_probs = cal.apply(test.logits, test.n_tokens)
    sent = sentences.evaluate(test, test_probs, thresholds, cal.edges)
    sentences.plot_pr(test, test_probs, sent, cal.edges, out_dir / "pr_curves.png")
    straddles = pq.read_table(features / "seqxgpt-test.parquet", columns=["straddles"])
    straddling = int(straddles["straddles"].to_numpy(zero_copy_only=False).sum())

    ece_splits = [test] if dev_spliced is None else [test, dev_spliced]
    cells = [c for s in ece_splits for c in calibrate.evaluate(cal, s)]
    style_cells = calibrate.evaluate(cal, test, by_style=True)
    path = report.write(run, doc_metrics, commentary, sent, straddling, cells, style_cells, out_dir)
    for p in sent.points:
        c = p.cells[-1]
        print(
            f"{p.name:>9}  P {c.precision:.3f}  R {c.recall:.3f}  FPR {c.fpr:.4f}  "
            f"IoU {p.iou_mean:.3f}"
        )
    print(f"wrote {path}")
    return 0


def _cluster(args: argparse.Namespace) -> int:
    from aivhuman.mil import cluster, train

    features, _, checkpoint, reports = _dirs(args.smoke, args.run)
    model, std = train.load(checkpoint)
    rep = cluster.cluster(features, config.PROCESSED_DIR, model, std, args.split, args.k)
    path = cluster.write_report(rep, reports / "clusters")
    for c in rep.clusters:
        print(
            f"cluster {c.index}: {c.size:>8,}  human {c.human_share:.1%}  "
            f"{', '.join(c.top_features)}"
        )
    print(f"wrote {path}")
    return 0


def _splice_group(doc_id: str) -> str:
    """A spliced document's group: its id without the `:splice{k}` suffix."""
    return doc_id.rsplit(":splice", 1)[0]


if __name__ == "__main__":
    raise SystemExit(main())
