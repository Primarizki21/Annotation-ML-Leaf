"""Shared freeze/unfreeze helpers for the 2-phase fine-tuning pattern
used by Scripts 2-5.

Each pretrained backbone exposes a top-level feature/classifier pair; we
freeze everything except the head for phase 1, then unfreeze all in
phase 2. The exact attribute names differ per architecture, so the
helpers accept the parent module and a list of "backbone" attribute
names to freeze.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def freeze_backbone(model: nn.Module, backbone_attrs: list[str]) -> None:
    """Set requires_grad=False on all params inside the listed submodules."""
    for attr in backbone_attrs:
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def trainable_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_optimizer_for_phase(
    model: nn.Module, lr: float
) -> torch.optim.Adam:
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
