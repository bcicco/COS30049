"""feature extraction, one parquet row per span"""

# spacy n_process pickles every Doc back to the parent, was ~5x slower than 1 proc.
# so workers parse + compute the arrays themselves, and cpu work overlaps with the gpu

import time
from collections.abc import Sequence
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aivhuman.features import FEATURE_NAMES, lexical, lm, syntax
from aivhuman.features.load import SpanDoc

CHUNK: Final = 10_000  # also the parquet row group size

_NLP: Any = None


def init_worker() -> None:
    global _NLP
    _NLP = syntax.load()


def text_features(job: tuple[str, list[tuple[int, int]]]) -> np.ndarray:
    # lexical then syntax, shape [n_spans, 5 + 13]
    global _NLP
    if _NLP is None:
        init_worker()
    text, spans = job
    return np.hstack([lexical.span_features(text, spans), syntax.span_features(_NLP(text), spans)])


def to_table(docs: Sequence[SpanDoc], features: Sequence[np.ndarray]) -> pa.Table:
    n = [len(d.spans) for d in docs]
    stacked = np.vstack(features).astype(np.float32)
    if stacked.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"{stacked.shape[1]} feature columns, expected {len(FEATURE_NAMES)}")

    def per_span(values: list[Any]) -> list[Any]:
        return [v for v, k in zip(values, n, strict=True) for _ in range(k)]

    columns: dict[str, Any] = {
        "doc_id": per_span([d.doc_id for d in docs]),
        "span_idx": np.concatenate([np.arange(k, dtype=np.int32) for k in n]),
        "label": pa.array(per_span([d.label for d in docs]), pa.int8()),
        "domain": per_span([d.domain for d in docs]),
        "breakdown": per_span([d.breakdown for d in docs]),
        "span_label": pa.array(
            [
                lab
                for d in docs
                for lab in (d.span_labels if d.span_labels is not None else [None] * len(d.spans))
            ],
            pa.int8(),
        ),
        "straddles": [s for d in docs for s in d.straddles],
    }
    for j, name in enumerate(FEATURE_NAMES):
        columns[name] = pa.array(stacked[:, j], pa.float32())
    return pa.table(columns)


def extract(
    docs: Sequence[SpanDoc],
    path: Path,
    ref: lm.ReferenceLM,
    pool: Pool | None = None,
    chunk: int = CHUNK,
) -> int:
    """Write features for docs, returns n span rows. pool must use init_worker"""
    if not docs:
        raise ValueError(f"no documents for {path.stem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".parquet.partial")
    writer: pq.ParquetWriter | None = None
    rows, start = 0, time.time()
    try:
        for lo in range(0, len(docs), chunk):
            batch = docs[lo : lo + chunk]
            jobs = [(d.text, d.spans) for d in batch]
            pending = pool.map_async(text_features, jobs, chunksize=64) if pool else None
            ids, offsets = ref.encode([d.text for d in batch])
            scores = ref.token_scores(ids)
            text_feats = pending.get() if pending else [text_features(j) for j in jobs]
            feats = [
                np.hstack(
                    [
                        lm.span_features(offs, sc, d.spans),
                        tf,
                        np.asarray(d.span_tokens, dtype=np.float64)[:, None],
                    ]
                )
                for d, offs, sc, tf in zip(batch, offsets, scores, text_feats, strict=True)
            ]
            table = to_table(batch, feats)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema)
            writer.write_table(table)
            rows += table.num_rows
            done = lo + len(batch)
            rate = done / (time.time() - start)
            print(f"  {path.stem}: {done:,}/{len(docs):,} docs ({rate:.0f} docs/s)", flush=True)
    finally:
        if writer is not None:
            writer.close()
    partial.replace(path)
    return rows
