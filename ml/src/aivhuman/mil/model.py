"""linear / spline sentence heads pooled to a doc score"""

import math
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn


class MILConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    head: Literal["linear", "gam"] = "linear"
    n_knots: int = 8  # gam only
    pooling: Literal["lse", "topk"] = "lse"
    tau: float = 2.0
    k: int = 3
    sentence_weight: float = 0.0  # sentence bce, only spliced docs have span labels
    l1: float = 1e-4
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 60
    patience: int = 8
    seed: int = 20240501


def pool_lse(logits: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
    # tau * log(mean exp(l / tau)), mean not sum so bag size doesnt matter
    z = (logits / tau).masked_fill(~mask, -math.inf)
    n = mask.sum(1).clamp(min=1)
    return tau * (torch.logsumexp(z, 1) - n.log())


def pool_topk(logits: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    kk = min(k, logits.shape[1])
    top = logits.masked_fill(~mask, -math.inf).topk(kk, dim=1).values
    count = mask.sum(1).clamp(min=1, max=kk)
    return torch.where(torch.isfinite(top), top, 0.0).sum(1) / count


def coverage(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits) * mask
    return probs.sum(1) / mask.sum(1).clamp(min=1)


class SplineHead(nn.Module):
    # term_f(z) = a_f z + sum_j b_fj (relu(z - k_fj) - relu(-k_fj))
    # centred so z = 0 gives 0

    def __init__(self, n_features: int, n_knots: int) -> None:
        super().__init__()
        self.linear = nn.Parameter(torch.zeros(n_features))
        self.hinge = nn.Parameter(torch.zeros(n_features, n_knots))
        self.bias = nn.Parameter(torch.zeros(()))
        self.knots: torch.Tensor
        self.register_buffer("knots", torch.zeros(n_features, n_knots))

    def terms(self, x: torch.Tensor) -> torch.Tensor:
        hinges = torch.relu(x.unsqueeze(-1) - self.knots) - torch.relu(-self.knots)
        return x * self.linear + (hinges * self.hinge).sum(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias + self.terms(x).sum(-1)


class MILModel(nn.Module):
    def __init__(self, n_features: int, cfg: MILConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features
        self.head: SplineHead | nn.Linear = (
            SplineHead(n_features, cfg.n_knots) if cfg.head == "gam" else nn.Linear(n_features, 1)
        )

    @property
    def name(self) -> str:
        return f"mil-{self.cfg.head}"

    def set_knots(self, x: torch.Tensor) -> None:
        if isinstance(self.head, SplineHead):
            q = torch.linspace(0.1, 0.9, self.cfg.n_knots)
            self.head.knots.copy_(torch.quantile(x, q, dim=0).T)

    def sentence_logits(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(x)
        return out if isinstance(self.head, SplineHead) else out.squeeze(-1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """returns doc logit, coverage, sentence logits"""
        logits = self.sentence_logits(x)
        if self.cfg.pooling == "lse":
            doc = pool_lse(logits, mask, self.cfg.tau)
        else:
            doc = pool_topk(logits, mask, self.cfg.k)
        return doc, coverage(logits, mask), logits

    def contributions(self, x: torch.Tensor) -> torch.Tensor:
        # sums to logit - bias
        if isinstance(self.head, SplineHead):
            return self.head.terms(x)
        return x * self.head.weight.squeeze(0)

    @property
    def bias(self) -> float:
        return float(self.head.bias.item())

    def l1_penalty(self) -> torch.Tensor:
        if isinstance(self.head, SplineHead):
            return self.head.linear.abs().sum() + self.head.hinge.abs().sum()
        return self.head.weight.abs().sum()

    @torch.no_grad()
    def slopes(self) -> torch.Tensor:
        # half the change from z=-1 to z=+1
        eye = torch.eye(self.n_features)
        return (self.contributions(eye).diagonal() - self.contributions(-eye).diagonal()) / 2
