"""Additive sentence heads (linear or spline), an optional CRF, pooled to a document score."""

import math
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn


class MILConfig(BaseModel):
    """Model and training settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    head: Literal["linear", "gam"] = "linear"
    n_knots: int = 8
    """Spline knots per feature for the GAM head, at train quantiles."""
    pooling: Literal["lse", "topk"] = "lse"
    tau: float = 2.0
    """LSE temperature."""
    k: int = 3
    """Top-k size."""
    coverage_weight: float = 0.0
    """Weight of the coverage BCE; measured to cost ~9 dev points at 0.3."""
    sentence_weight: float = 0.0
    """Weight of a sentence BCE on spans with known labels (spliced documents only)."""
    crf: bool = False
    """Replace sentence logits with the marginal log-odds of a symmetric two-state CRF."""
    crf_lr: float = 1e-2
    """Learning rate of the CRF stickiness; the head's rate barely moves it before early
    stopping."""
    l1: float = 1e-4
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 60
    patience: int = 8
    seed: int = 20240501


def pool_lse(logits: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
    """`tau * log(mean_i exp(l_i / tau))` over unmasked sentences; invariant to bag size."""
    z = (logits / tau).masked_fill(~mask, -math.inf)
    n = mask.sum(1).clamp(min=1)
    return tau * (torch.logsumexp(z, 1) - n.log())


def pool_topk(logits: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of the `k` highest unmasked logits, or of all of them in a shorter bag."""
    kk = min(k, logits.shape[1])
    top = logits.masked_fill(~mask, -math.inf).topk(kk, dim=1).values
    count = mask.sum(1).clamp(min=1, max=kk)
    return torch.where(torch.isfinite(top), top, 0.0).sum(1) / count


def coverage(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean sentence probability over unmasked sentences."""
    probs = torch.sigmoid(logits) * mask
    return probs.sum(1) / mask.sum(1).clamp(min=1)


def crf_marginals(
    emissions: torch.Tensor, mask: torch.Tensor, transitions: torch.Tensor, start: torch.Tensor
) -> torch.Tensor:
    """Posterior log-odds of the machine state per sentence, by forward-backward.

    State 0 is human with emission 0, state 1 machine with emission `emissions`, so zero
    `transitions [2, 2]` and `start [2]` return the emissions unchanged. `mask` must be a
    prefix per row (right padding); padded steps carry the recursions through unchanged.
    """
    n = emissions.shape[1]
    emit = torch.stack([torch.zeros_like(emissions), emissions], -1)
    alpha = [start + emit[:, 0]]
    for t in range(1, n):
        step = torch.logsumexp(alpha[-1].unsqueeze(-1) + transitions, 1) + emit[:, t]
        alpha.append(torch.where(mask[:, t, None], step, alpha[-1]))
    beta = [torch.zeros_like(alpha[0])]
    for t in range(n - 2, -1, -1):
        step = torch.logsumexp(transitions + (emit[:, t + 1] + beta[0]).unsqueeze(1), 2)
        beta.insert(0, torch.where(mask[:, t + 1, None], step, beta[0]))
    post = torch.stack(alpha, 1) + torch.stack(beta, 1)
    return post[..., 1] - post[..., 0]


class SplineHead(nn.Module):
    """One piecewise-linear function per feature, summed: still additive and plottable.

    `term_f(z) = a_f z + sum_j b_fj (relu(z - k_fj) - relu(-k_fj))`, centred so that a feature
    at its train mean (z = 0) contributes nothing.
    """

    def __init__(self, n_features: int, n_knots: int) -> None:
        super().__init__()
        self.linear = nn.Parameter(torch.zeros(n_features))
        self.hinge = nn.Parameter(torch.zeros(n_features, n_knots))
        self.bias = nn.Parameter(torch.zeros(()))
        self.knots: torch.Tensor
        self.register_buffer("knots", torch.zeros(n_features, n_knots))

    def terms(self, x: torch.Tensor) -> torch.Tensor:
        """Per-feature contribution, same shape as `x`."""
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
        if cfg.crf:
            # One log-potential for staying in either state. Tied and with no start preference,
            # the CRF cannot learn which author comes first or how often authorship switches
            # in one direction; a free 2x2 matrix learned SeqXGPT's human-then-machine order.
            self.stickiness = nn.Parameter(torch.zeros(()))

    @property
    def transitions(self) -> torch.Tensor:
        """`[from, to]` log-potentials implied by the stickiness; 0 = human, 1 = machine."""
        return self.stickiness * torch.eye(2)

    @property
    def name(self) -> str:
        return f"mil-{self.cfg.head}" + ("-crf" if self.cfg.crf else "")

    def set_knots(self, x: torch.Tensor) -> None:
        """Place GAM knots at evenly spaced quantiles of standardised train features."""
        if isinstance(self.head, SplineHead):
            q = torch.linspace(0.1, 0.9, self.cfg.n_knots)
            self.head.knots.copy_(torch.quantile(x, q, dim=0).T)

    def sentence_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Emission logits from the head alone, before any CRF."""
        out: torch.Tensor = self.head(x)
        return out if isinstance(self.head, SplineHead) else out.squeeze(-1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Document logit, coverage, and final sentence logits (CRF marginals if enabled)."""
        logits = self.sentence_logits(x)
        if self.cfg.crf:
            logits = crf_marginals(logits, mask, self.transitions, torch.zeros(2))
        if self.cfg.pooling == "lse":
            doc = pool_lse(logits, mask, self.cfg.tau)
        else:
            doc = pool_topk(logits, mask, self.cfg.k)
        return doc, coverage(logits, mask), logits

    def contributions(self, x: torch.Tensor) -> torch.Tensor:
        """Per-feature contribution to each sentence logit; they sum to logit minus bias."""
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
        """Effective direction per feature: half the change in its term from z = -1 to +1."""
        eye = torch.eye(self.n_features)
        return (self.contributions(eye).diagonal() - self.contributions(-eye).diagonal()) / 2
