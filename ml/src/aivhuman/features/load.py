import random
from pathlib import Path
from typing import Final

import orjson

from aivhuman.evaluate import (
    SPLIT_SOURCE,
    EvalDoc,
    breakdown_name,
    load_manifest,
    sample_per_group,
)

TRAIN_MACHINE_PER_GROUP: Final = 4
SAMPLE_SEED: Final = 20240501


class SpanDoc(EvalDoc):
    spans: list[tuple[int, int]]
    span_tokens: list[int]
    span_labels: list[int] | None  # seqxgpt only
    straddles: list[bool]


def load_span_docs(processed_dir: Path, manifests_dir: Path, split: str) -> list[SpanDoc]:
    # train gets subsampled per group
    keep = load_manifest(manifests_dir, split)
    docs = list(_read(processed_dir / f"{SPLIT_SOURCE[split]}.jsonl", keep))
    if split == "train":
        chosen = sample_per_group(docs, TRAIN_MACHINE_PER_GROUP, random.Random(SAMPLE_SEED))
        docs = [docs[i] for i in sorted(chosen)]
    return docs


def _read(path: Path, keep: dict[str, str]) -> list[SpanDoc]:
    out = []
    with path.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = orjson.loads(line)
            if d["doc_id"] not in keep:
                continue
            sentences = d["sentences"]
            out.append(
                SpanDoc(
                    doc_id=d["doc_id"],
                    text=d["text"],
                    label=d["label"],
                    group_id=d["group_id"],
                    domain=d["domain"],
                    breakdown=breakdown_name(d),
                    spans=[(s["start"], s["end"]) for s in sentences],
                    span_tokens=[s["n_tokens"] for s in sentences],
                    span_labels=[s["label"] for s in sentences]
                    if d["source"] == "seqxgpt"
                    else None,
                    straddles=[s.get("straddles_boundary", False) for s in sentences],
                )
            )
    return out
