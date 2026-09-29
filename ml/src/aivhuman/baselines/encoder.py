"""ModernBERT-base document classifier.... mean-pooled, one binary head, no MIL."""

import math
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from aivhuman.evaluate import EvalDoc, sample_per_group, tpr_at_fpr, write_predictions
from aivhuman.schema import LABEL_HUMAN

MODEL_NAME: Final = "modernbert-doc"
BACKBONE: Final = "answerdotai/ModernBERT-base"
CHUNK: Final = 20_000
"""Documents tokenised at a time."""


class EncoderConfig(BaseModel):
    """Training settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_length: int = 512
    batch_size: int = 8
    grad_accum: int = 4
    log_every: int = 50
    eval_batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    epochs: int = 2
    machine_per_group: int = 4
    """Machine docs sampled per group each epoch.....every human doc is always kept."""
    dev_machine_per_group: int = 2
    seed: int = 20240501


class DocClassifier(nn.Module):
    def __init__(self, backbone: str = BACKBONE) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone, attn_implementation="sdpa")
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        logits: torch.Tensor = self.head(pooled).squeeze(-1)
        return logits


def _batches(order: list[int], lengths: list[int], batch_size: int) -> list[list[int]]:
    """Batches of similar length, to cut padding. Order across batches follows `order`."""
    window = batch_size * 50
    out: list[list[int]] = []
    for start in range(0, len(order), window):
        chunk = sorted(order[start : start + window], key=lambda i: lengths[i])
        out.extend(chunk[j : j + batch_size] for j in range(0, len(chunk), batch_size))
    return out


def _collate(
    ids: list[np.ndarray], batch: list[int], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(ids[i]) for i in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, i in enumerate(batch):
        input_ids[row, : len(ids[i])] = torch.from_numpy(ids[i])
        mask[row, : len(ids[i])] = 1
    return input_ids.to(device), mask.to(device)


class Encoder:
    """Tokenisation, training and scoring around a DocClassifier."""

    def __init__(self, cfg: EncoderConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(BACKBONE)  # type: ignore[no-untyped-call, unused-ignore]
        self.model = DocClassifier().to(self.device)

    def tokenize(self, docs: list[EvalDoc]) -> list[np.ndarray]:
        """Token ids per document, as int32 arrays to keep large splits in memory."""
        out: list[np.ndarray] = []
        for start in range(0, len(docs), CHUNK):
            encoded = self.tokenizer(
                [d.text for d in docs[start : start + CHUNK]],
                truncation=True,
                max_length=self.cfg.max_length,
            )["input_ids"]
            out.extend(np.asarray(x, dtype=np.int32) for x in encoded)
        return out

    @torch.no_grad()
    def score(self, docs: list[EvalDoc]) -> np.ndarray:
        """Machine probability per document."""
        self.model.eval()
        out = np.empty(len(docs), dtype=np.float64)
        for start in range(0, len(docs), CHUNK):
            ids = self.tokenize(docs[start : start + CHUNK])
            lengths = [len(x) for x in ids]
            order = sorted(range(len(ids)), key=lambda i: lengths[i])
            for batch in _batches(order, lengths, self.cfg.eval_batch_size):
                input_ids, mask = _collate(ids, batch, self.tokenizer.pad_token_id, self.device)
                with torch.autocast(self.device.type, dtype=torch.bfloat16):
                    logits = self.model(input_ids, mask)
                out[[start + i for i in batch]] = torch.sigmoid(logits.float()).cpu().numpy()
        return out

    def fit(self, train: list[EvalDoc], dev: list[EvalDoc], checkpoint: Path) -> list[float]:
        """Train, keeping the epoch with the best dev TPR at 1% FPR. Returns dev TPR per epoch."""
        cfg = self.cfg
        rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)

        dev_sample = [dev[i] for i in sample_per_group(dev, cfg.dev_machine_per_group, rng)]
        dev_labels = np.array([d.label for d in dev_sample])
        ids = self.tokenize(train)
        lengths = [len(x) for x in ids]
        labels = torch.tensor([d.label for d in train], dtype=torch.float32)

        n_epoch = len(sample_per_group(train, cfg.machine_per_group, random.Random(0)))
        steps = math.ceil(n_epoch / (cfg.batch_size * cfg.grad_accum)) * cfg.epochs
        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        schedule = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call, unused-ignore]
            optimiser, int(steps * cfg.warmup_frac), steps
        )

        history: list[float] = []
        for epoch in range(cfg.epochs):
            chosen = sample_per_group(train, cfg.machine_per_group, rng)
            rng.shuffle(chosen)
            n_human = sum(train[i].label == LABEL_HUMAN for i in chosen)
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(n_human / (len(chosen) - n_human), device=self.device)
            )
            batches = _batches(chosen, lengths, cfg.batch_size)
            rng.shuffle(batches)

            self.model.train()
            start, running = time.time(), 0.0
            for step, batch in enumerate(batches, start=1):
                input_ids, mask = _collate(ids, batch, self.tokenizer.pad_token_id, self.device)
                with torch.autocast(self.device.type, dtype=torch.bfloat16):
                    logits = self.model(input_ids, mask)
                loss = loss_fn(logits.float(), labels[batch].to(self.device)) / cfg.grad_accum
                loss.backward()
                running += loss.item() * cfg.grad_accum
                if step % cfg.grad_accum == 0 or step == len(batches):
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimiser.step()
                    schedule.step()
                    optimiser.zero_grad(set_to_none=True)
                if step % cfg.log_every == 0:
                    rate = step * cfg.batch_size / (time.time() - start)
                    print(
                        f"  epoch {epoch} step {step}/{len(batches)}"
                        f" loss {running / cfg.log_every:.4f} ({rate:.0f} docs/s)",
                        flush=True,
                    )
                    running = 0.0

            tpr, _ = tpr_at_fpr(dev_labels, self.score(dev_sample), 0.01)
            print(f"  epoch {epoch}: dev TPR@1%FPR {tpr:.4f}", flush=True)
            if not history or tpr > max(history):
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint)
            history.append(tpr)

        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        return history


def predict(
    encoder: Encoder, splits: dict[str, list[EvalDoc]], predictions_dir: Path
) -> Iterator[str]:
    """Score every non-train split, skipping any already written. Yields each split done."""
    for split, docs in splits.items():
        path = predictions_dir / MODEL_NAME / f"{split}.parquet"
        if split == "train" or path.exists():
            continue
        write_predictions(path, [d.doc_id for d in docs], encoder.score(docs))
        yield split
