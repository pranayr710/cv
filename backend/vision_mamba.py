"""A small Vision Mamba for still engagement images.

An image is not a sequence, but Vision Mamba (Vim) makes it one: the picture is
cut into patches, flattened in raster order, and scanned by the same selective
state-space recurrence used on video windows in :mod:`backend.mamba_model`.
That is what makes "train the Mamba on the image set" a real thing to do rather
than a category error -- the video model consumes 48 timesteps of behavioural
features and cannot read pixels at all.

Two departures from the published Vim, both because of the data:

**Bidirectional scanning.** A video window has a direction and the model must
not see the future. An image does not: the patch above matters as much as the
patch below, and scanning raster order alone would make the top-left corner
systematically less informed than the bottom-right. Each block therefore scans
forwards and backwards and sums, which is what Vim does and what the video
model deliberately does not.

**Size.** Published Vim-Ti is about 7M parameters trained on ImageNet's 1.28M
images. Here there are 12,582 images of which 83.7% were near-duplicates of
something else before deduplication, so the effective sample is far smaller
than the count suggests. This is built at a few hundred thousand parameters,
and whether even that is too many is measured rather than assumed -- the
comparison against frozen ResNet18 features is in
``tools/train_vision_mamba.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from backend.mamba_model import SelectiveSSM


@dataclass(frozen=True)
class VisionMambaConfig:
    """Shape of the image model."""

    image_size: int = 128
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 96
    d_state: int = 8
    expand: int = 2
    n_layers: int = 4
    dropout: float = 0.2
    n_classes: int = 1

    @property
    def n_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side


class BiMambaBlock(nn.Module):
    """A Mamba block that scans the patch sequence in both directions.

    The forward and backward passes share nothing but the block's projections,
    so each learns its own state dynamics; summing rather than concatenating
    keeps the width constant and the parameter count down.
    """

    def __init__(self, cfg: VisionMambaConfig) -> None:
        super().__init__()
        d_inner = cfg.d_model * cfg.expand
        self.norm = nn.LayerNorm(cfg.d_model)
        self.in_proj = nn.Linear(cfg.d_model, d_inner * 2, bias=False)
        # Non-causal padding: an image patch may legitimately see its
        # neighbours on both sides.
        self.conv = nn.Conv1d(d_inner, d_inner, 3, groups=d_inner, padding=1)
        self.ssm_fwd = SelectiveSSM(d_inner, cfg.d_state)
        self.ssm_bwd = SelectiveSSM(d_inner, cfg.d_state)
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = torch.nn.functional.silu(x)
        y = self.ssm_fwd(x) + self.ssm_bwd(x.flip(1)).flip(1)
        y = y * torch.nn.functional.silu(gate)
        return residual + self.out_proj(y)


class VisionMamba(nn.Module):
    """Patch-embed, scan, pool, classify."""

    def __init__(self, cfg: VisionMambaConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VisionMambaConfig()
        self.patch = nn.Conv2d(self.cfg.in_channels, self.cfg.d_model,
                               self.cfg.patch_size, stride=self.cfg.patch_size)
        # Learned position embedding: the selective scan carries order, but
        # raster order alone does not tell a patch which row it is on.
        self.pos = nn.Parameter(
            torch.zeros(1, self.cfg.n_patches, self.cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            BiMambaBlock(self.cfg) for _ in range(self.cfg.n_layers))
        self.norm = nn.LayerNorm(self.cfg.d_model)
        self.drop = nn.Dropout(self.cfg.dropout)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x of shape (batch, 3, H, W). Returns logits (batch,)."""
        h = self.patch(x).flatten(2).transpose(1, 2)
        h = h + self.pos
        for block in self.blocks:
            h = block(h)
        h = self.norm(h).mean(dim=1)
        return self.head(self.drop(h)).squeeze(-1)

    def features(self, x: Tensor) -> Tensor:
        """Pooled representation before the head, for reuse elsewhere."""
        h = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        for block in self.blocks:
            h = block(h)
        return self.norm(h).mean(dim=1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
