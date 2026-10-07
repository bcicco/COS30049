"""feature parquet -> bags of sentence vectors"""

from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict

from aivhuman.baselines.encoder import _batches
from aivhuman.text.normalize import stable_hash

CLIP: Final = 5.0  # heavy tailed features were blowing up logits
SENTENCE_VAL_BUCKETS: Final = 5  # seqxgpt-calib split 1:4, sent val vs calibration


class Standardizer(BaseModel):
    # fit on train only
    model_config = ConfigDict(frozen=True, extra="forbid")

    names: list[str]
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, x: np.ndarray, names: Sequence[str]) -> "Standardizer":
        std = np.nanstd(x, axis=0)
        return cls(
            names=list(names),
            mean=np.nanmean(x, axis=0).tolist(),
            std=np.where(std > 0, std, 1.0).tolist(),
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        # nan -> 0 i.e. the train mean
        z = (x - np.asarray(self.mean)) / np.asarray(self.std)
        return np.clip(np.nan_to_num(z, nan=0.0), -CLIP, CLIP).astype(np.float32)


class Bags(BaseModel):
    """all sentences flat in x, bags are slices of it (ragged, no padding)"""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    doc_ids: list[str]
    labels: np.ndarray  # doc label per bag
    x: np.ndarray  # [n_spans, n_features], raw until standardised
    offsets: np.ndarray  # bag i = rows offsets[i]:offsets[i+1]
    span_labels: np.ndarray  # -1 = unknown
    straddles: np.ndarray  # seqxgpt span contains the human/machine boundary

    def __len__(self) -> int:
        return len(self.doc_ids)

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def standardised(self, std: Standardizer) -> "Bags":
        return self.model_copy(update={"x": std.transform(self.x)})

    def subset(self, keep: np.ndarray) -> "Bags":
        # keep = bool mask over bags, rows and offsets rebuilt to match
        idx = np.flatnonzero(keep)
        rows = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx])
        return Bags(
            doc_ids=[self.doc_ids[i] for i in idx],
            labels=self.labels[idx],
            x=self.x[rows],
            offsets=np.r_[0, np.cumsum(self.sizes[idx])],
            span_labels=self.span_labels[rows],
            straddles=self.straddles[rows],
        )

    def concat(self, other: "Bags") -> "Bags":
        return Bags(
            doc_ids=[*self.doc_ids, *other.doc_ids],
            labels=np.r_[self.labels, other.labels],
            x=np.vstack([self.x, other.x]),
            offsets=np.r_[self.offsets, other.offsets[1:] + self.offsets[-1]],
            span_labels=np.r_[self.span_labels, other.span_labels],
            straddles=np.r_[self.straddles, other.straddles],
        )

    def sentence_labels(self) -> tuple[np.ndarray, np.ndarray]:
        # known label and not straddling
        keep = (self.span_labels >= 0) & ~self.straddles
        return keep, self.span_labels[keep].astype(int)


def in_sentence_validation(doc_ids: Sequence[str], groups: dict[str, str]) -> np.ndarray:
    # hashed on group id so a base doc cant end up on both sides
    return np.array([int(stable_hash(groups[d]), 16) % SENTENCE_VAL_BUCKETS == 0 for d in doc_ids])


def load_bags(path: Path, names: Sequence[str], keep: Collection[str] | None = None) -> Bags:
    """read features, optionally just the docs in keep"""
    cols = ["doc_id", "span_idx", "label", "span_label", "straddles", *names]
    # filter in pyarrow so unwanted rows never get loaded
    filters = [("doc_id", "in", list(keep))] if keep is not None else None
    table = pq.read_table(path, columns=cols, filters=filters)
    doc_col = table["doc_id"].to_numpy(zero_copy_only=False)
    # bag starts are where doc_id changes. needs each doc's rows contiguous + in order
    starts = np.flatnonzero(np.r_[True, doc_col[1:] != doc_col[:-1]])
    span_idx = table["span_idx"].to_numpy()
    if (span_idx[starts] != 0).any() or len(set(doc_col[starts])) != len(starts):
        raise ValueError(f"{path}: document rows are not contiguous")
    x = np.column_stack([table[n].to_numpy(zero_copy_only=False) for n in names])
    return Bags(
        doc_ids=doc_col[starts].tolist(),
        labels=table["label"].to_numpy()[starts].astype(np.float32),
        x=x.astype(np.float64),
        offsets=np.r_[starts, len(doc_col)],
        span_labels=table["span_label"].to_numpy(zero_copy_only=False).astype(np.float64),
        straddles=table["straddles"].to_numpy(zero_copy_only=False),
    )


def batches(
    bags: Bags, order: Sequence[int], batch_size: int
) -> Iterator[tuple[list[int], torch.Tensor, torch.Tensor]]:
    # (idx, x [B,S,F], mask [B,S]), padded per batch
    # length bucketed like the encoder so docs of similar sentence count share a batch
    sizes = bags.sizes
    for batch in _batches(list(order), sizes.tolist(), batch_size):
        width = int(sizes[batch].max())
        x = np.zeros((len(batch), width, bags.x.shape[1]), dtype=np.float32)
        mask = np.zeros((len(batch), width), dtype=bool)
        for row, i in enumerate(batch):
            lo, hi = bags.offsets[i], bags.offsets[i + 1]
            x[row, : hi - lo] = bags.x[lo:hi]
            mask[row, : hi - lo] = True
        yield batch, torch.from_numpy(x), torch.from_numpy(mask)
