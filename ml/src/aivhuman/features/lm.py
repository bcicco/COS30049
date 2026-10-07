"""GPT-2 small reference LM features"""

from typing import Any, Final

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aivhuman.text.tokens import assign_to_spans

REFERENCE_LM: Final = "gpt2"
MAX_CONTEXT: Final = 1024  # GPT-2 max positions
CARRIED: Final = 512  # left context carried into each window after the 1st
TOP_K: Final = 10

# cols of the per token score matrix
LOGPROB, LOGRANK, TOP10, ENTROPY = range(4)


def windows(
    n_tokens: int, max_context: int = MAX_CONTEXT, carried: int = CARRIED
) -> list[tuple[int, int, int]]:
    """(start, end, score_from) windows over [BOS] + tokens"""
    # long docs dont fit in one pass. each window after the 1st re-reads `carried`
    # tokens as context but only scores from score_from, so each token is scored once
    if not 0 < carried < max_context:
        raise ValueError(f"carried={carried} must lie in (0, max_context={max_context})")
    n = n_tokens + 1
    out = []
    start, score_from = 0, 1
    while True:
        end = min(n, start + max_context)
        out.append((start, end, score_from))
        if end == n:
            return out
        score_from = end
        start = end - carried


def span_features(
    offsets: list[tuple[int, int]], scores: np.ndarray, spans: list[tuple[int, int]]
) -> np.ndarray:
    """[n_spans, 6]: mean logprob, logrank, top10, entropy, logprob - doc mean, burstiness"""
    # map each LM token to the span it falls in (-1 = none) then average per span
    owner = np.array(assign_to_spans(offsets, spans), dtype=np.int64)
    out = np.full((len(spans), 6), np.nan, dtype=np.float64)
    inside = owner >= 0
    if not inside.any():
        return out
    counts = np.bincount(owner[inside], minlength=len(spans))
    has = counts > 0
    for col in (LOGPROB, LOGRANK, TOP10, ENTROPY):
        sums = np.bincount(owner[inside], weights=scores[inside, col], minlength=len(spans))
        out[has, col] = sums[has] / counts[has]
    # context features: how far a span sits from its doc, and the spread across the doc
    doc_mean = scores[inside, LOGPROB].mean()
    out[:, 4] = out[:, LOGPROB] - doc_mean
    out[:, 5] = np.nanstd(out[:, LOGPROB]) if has.any() else np.nan
    return out


class ReferenceLM:
    """batched GPT-2 scoring of every token in a doc"""

    # token_budget = tokens per forward pass. ~1MB/token for the fp32 vocab dists,
    # 2048 ok on my 6GB card, 16384 for an 80GB one

    def __init__(
        self,
        device: torch.device,
        token_budget: int = 2048,
        model: Any = None,
        tokenizer: Any = None,
        max_context: int = MAX_CONTEXT,
        carried: int = CARRIED,
    ) -> None:
        self.device = device
        self.token_budget = token_budget
        self.max_context = max_context
        self.carried = carried
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(REFERENCE_LM)  # type: ignore[no-untyped-call, unused-ignore]
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(REFERENCE_LM)
        self.model: Any = model
        self.model.to(device)
        self.model.eval()
        self.bos = self.tokenizer.bos_token_id

    def encode(self, texts: list[str]) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
        # char offsets are kept so tokens can be mapped back to sentence spans
        enc = self.tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
        return enc["input_ids"], [[tuple(o) for o in offs] for offs in enc["offset_mapping"]]

    @torch.no_grad()
    def token_scores(self, ids: list[list[int]]) -> list[np.ndarray]:
        # per doc [n_tokens, 4]: logprob, logrank, in top10, entropy
        seqs = [[self.bos, *doc] for doc in ids]
        out = [np.empty((len(doc), 4), dtype=np.float32) for doc in ids]
        jobs = [
            (d, s, e, f)
            for d, doc in enumerate(ids)
            for s, e, f in windows(len(doc), self.max_context, self.carried)
            if f < e
        ]
        jobs.sort(key=lambda j: j[2] - j[1])
        # sorted by length so the newest job is always the widest one in the batch
        # pack windows until batch size * width would go over the token budget
        batch: list[tuple[int, int, int, int]] = []
        for job in jobs:
            if batch and (len(batch) + 1) * (job[2] - job[1]) > self.token_budget:
                self._score_batch(seqs, batch, out)
                batch = []
            batch.append(job)
        if batch:
            self._score_batch(seqs, batch, out)
        return out

    def _score_batch(
        self,
        seqs: list[list[int]],
        batch: list[tuple[int, int, int, int]],
        out: list[np.ndarray],
    ) -> None:
        # right pad to the widest window, mask hides the padding
        width = max(e - s for _, s, e, _ in batch)
        input_ids = torch.full((len(batch), width), self.bos, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (d, s, e, _) in enumerate(batch):
            input_ids[row, : e - s] = torch.tensor(seqs[d][s:e])
            mask[row, : e - s] = 1
        input_ids, mask = input_ids.to(self.device), mask.to(self.device)
        with torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            logits = self.model(input_ids=input_ids, attention_mask=mask).logits
        logp = logits[:, :-1].float().log_softmax(-1)
        target = input_ids[:, 1:]
        tok = logp.gather(-1, target.unsqueeze(-1))
        # rank 1 = the top guess. entropy = how spread out the next token dist is
        rank = (logp > tok).sum(-1) + 1
        entropy = -(logp.exp() * logp).sum(-1)
        stats = (
            torch.stack(
                [tok.squeeze(-1), rank.float().log(), (rank <= TOP_K).float(), entropy],
                dim=-1,
            )
            .cpu()
            .numpy()
        )
        for row, (d, s, e, f) in enumerate(batch):
            # logits at pos j predict seq pos s+j+1, which is token s+j
            out[d][f - 1 : e - 1] = stats[row, f - s - 1 : e - s - 1]
