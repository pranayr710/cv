"""A small Mamba (selective state-space) model for window attention scoring.

Why a hand-written block rather than ``mamba_ssm``: the reference package ships
a fused CUDA kernel that has no Windows build. The kernel is a speed
optimisation, not part of the method -- the recurrence below is the same
selective scan, written in plain PyTorch. At 179 sequences of 48 steps the
sequential scan costs milliseconds, so nothing is lost but throughput.

Why it is small. The published vision Mamba models are 7-30M parameters,
trained on ImageNet-scale corpora. Fitted to a few hundred labelled windows
they would memorise rather than generalise, and that is not a prediction: a
10k-parameter MLP already scored *worse* here than a 300-parameter one. This
block is sized in the low thousands so its capacity is comparable to the
models it is being measured against, which is the only way the comparison
says anything about the architecture rather than about the parameter count.

What is genuinely different from the aggregate model: this reads the window as
an ordered sequence. "Looked away and came back" and "drifted away and stayed"
have the same means and fractions but different trajectories, and only a
sequence model can separate them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MambaConfig:
    """Shape of the model. Defaults are sized for a few hundred labels."""

    n_features: int = 13
    d_model: int = 16
    d_state: int = 8
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 1
    dropout: float = 0.3


class SelectiveSSM(nn.Module):
    """The selective scan at the heart of Mamba.

    The step that separates Mamba from earlier state-space models (S4) is that
    the timestep and the input/output projections are *functions of the input*
    rather than fixed. That is what lets the model decide, per timestep,
    whether to write something into state or let it pass -- the "selectivity"
    that matters here, where one frame of looking away is noise and twenty
    consecutive ones are not.
    """

    def __init__(self, d_inner: int, d_state: int) -> None:
        super().__init__()
        self.d_inner, self.d_state = d_inner, d_state
        # delta, B and C are produced from the input: the selective part.
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        # A is kept in log space and negated on use, so the recurrence is
        # guaranteed to decay rather than blow up over a long window.
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: Tensor) -> Tensor:
        """Args: x of shape (batch, length, d_inner). Returns the same shape."""
        b, length, _ = x.shape
        A = -torch.exp(self.A_log)                     # (d_inner, d_state)
        parts = self.x_proj(x)
        dt = torch.nn.functional.softplus(
            self.dt_proj(parts[..., :1]))              # (b, l, d_inner)
        B = parts[..., 1:1 + self.d_state]             # (b, l, d_state)
        C = parts[..., 1 + self.d_state:]              # (b, l, d_state)

        # Zero-order hold discretisation, the standard Mamba form.
        dA = torch.exp(dt.unsqueeze(-1) * A)           # (b, l, d_in, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)         # (b, l, d_in, d_state)

        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device,
                        dtype=x.dtype)
        outs = []
        for t in range(length):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            outs.append(torch.einsum("bds,bs->bd", h, C[:, t]))
        y = torch.stack(outs, dim=1)
        return y + x * self.D


class MambaBlock(nn.Module):
    """One Mamba layer: gated selective SSM with a causal depthwise conv."""

    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        d_inner = cfg.d_model * cfg.expand
        self.norm = nn.LayerNorm(cfg.d_model)
        self.in_proj = nn.Linear(cfg.d_model, d_inner * 2, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, cfg.d_conv,
                              groups=d_inner, padding=cfg.d_conv - 1)
        self.ssm = SelectiveSSM(d_inner, cfg.d_state)
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        # Causal: trim the right padding so no timestep sees the future.
        x = self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        x = torch.nn.functional.silu(x)
        x = self.ssm(x)
        x = x * torch.nn.functional.silu(gate)
        return residual + self.out_proj(x)


class MambaAttentionScorer(nn.Module):
    """Sequence of per-frame features in, one engagement probability out.

    Pooling is masked-mean over real timesteps only. Padding a short window to
    a fixed length and then averaging over the padding too would make coverage
    leak into the prediction as an artefact of the padding scheme.
    """

    def __init__(self, cfg: MambaConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MambaConfig()
        self.embed = nn.Linear(self.cfg.n_features, self.cfg.d_model)
        self.blocks = nn.ModuleList(
            MambaBlock(self.cfg) for _ in range(self.cfg.n_layers))
        self.norm = nn.LayerNorm(self.cfg.d_model)
        self.drop = nn.Dropout(self.cfg.dropout)
        self.head = nn.Linear(self.cfg.d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x of shape (batch, length, n_features). Returns logits (batch,)."""
        # Column 0 of the per-frame vector is the presence flag.
        mask = x[..., 0:1]
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(self.drop(pooled)).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def probability(model: MambaAttentionScorer, steps: list[list[float]],
                device: str = "cpu") -> float:
    """One window's probability of being on task."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor([steps], dtype=torch.float32, device=device)
        return float(torch.sigmoid(model(x))[0])


def to_score_10(prob: float) -> int:
    """A 1-10 score, matching backend.attention_score.to_score_10."""
    return max(1, min(10, math.floor(prob * 10) + 1))
