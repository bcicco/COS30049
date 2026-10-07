"""Baseline CLI: tfidf, encoder, evaluate."""

import argparse
import time
from collections.abc import Sequence

import orjson
import torch

from aivhuman import config
from aivhuman import evaluate as ev
from aivhuman.baselines import encoder

# prediction dirs under data/predictions/ that `evaluate` reads
MODELS = ("tfidf-lr", "modernbert-doc")


def main(argv: Sequence[str] | None = None) -> int:
    """entry point for aivhuman-baseline"""
    config.configure_stdio()
    parser = argparse.ArgumentParser(prog="aivhuman-baseline", description=__doc__)
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser("tfidf", help="fit tf-idf + LR on train and score the eval splits")
    p.set_defaults(handler=_tfidf)

    # fine tune ModernBERT on train, select on dev, then score every eval split
    p = subparsers.add_parser("encoder")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--predict-only", action="store_true")  # reuse the saved checkpoint
    p.set_defaults(handler=_encoder)

    # metrics for both baselines from their saved predictions
    p = subparsers.add_parser("evaluate")
    p.set_defaults(handler=_evaluate)

    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        return 2
    result: int = handler(args)
    return result


def _load(splits: Sequence[str]) -> dict[str, list[ev.EvalDoc]]:
    """load docs for the named splits from data/processed via the manifests"""
    start = time.time()
    loaded = ev.load_splits(config.PROCESSED_DIR, config.MANIFESTS_DIR, splits)
    sizes = ", ".join(f"{k} {len(v):,}" for k, v in loaded.items())
    print(f"loaded {sizes} in {time.time() - start:.0f}s", flush=True)
    return loaded


def _tfidf(_args: argparse.Namespace) -> int:
    """fit tf-idf + LR, grid searched on dev, and write predictions"""
    from aivhuman.baselines import tfidf

    splits = _load(["train", *ev.EVAL_SPLITS])
    grid = tfidf.run(splits, config.PREDICTIONS_DIR)
    print(f"dev grid: {grid}")
    return 0


def _encoder(args: argparse.Namespace) -> int:
    """train (or load) ModernBERT and write predictions per split"""
    # only pass flags that were set so EncoderConfig defaults win otherwise
    overrides = {
        k: v
        for k in ("epochs", "batch_size", "grad_accum", "eval_batch_size")
        if (v := getattr(args, k)) is not None
    }
    cfg = encoder.EncoderConfig(**overrides)
    checkpoint = config.CHECKPOINTS_DIR / encoder.MODEL_NAME / "model.pt"
    model = encoder.Encoder(cfg)

    if args.predict_only:
        model.model.load_state_dict(torch.load(checkpoint, map_location=model.device))
        splits = _load(ev.EVAL_SPLITS)
    else:
        splits = _load(["train", *ev.EVAL_SPLITS])
        history = model.fit(splits["train"], splits["dev"], checkpoint)
        # old preds from a previous ckpt would get skipped as already scored otherwise
        for stale in (config.PREDICTIONS_DIR / encoder.MODEL_NAME).glob("*.parquet"):
            stale.unlink()
        print(f"dev TPR@1%FPR per epoch: {history}", flush=True)
    for split in encoder.predict(model, splits, config.PREDICTIONS_DIR):
        print(f"  scored {split}", flush=True)
    return 0


def _evaluate(_args: argparse.Namespace) -> int:
    """score both baselines on the eval splits and write the report json"""
    splits = _load(ev.EVAL_SPLITS)
    metrics = [
        m for name in MODELS for m in ev.evaluate_model(name, config.PREDICTIONS_DIR, splits)
    ]
    path = ev.write_report(metrics, config.BASELINES_REPORT_DIR)
    for m in metrics:
        print(orjson.dumps(m.model_dump(exclude={"by_generator"})).decode())
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
