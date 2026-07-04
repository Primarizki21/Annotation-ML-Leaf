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
from torch.optim import Optimizer


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


def print_phase_state(
    model: nn.Module,
    optimizer: Optimizer,
    early_stop,
    lr: float,
) -> None:
    """Print trainable count + optimizer info + early-stop state.

    Call once per phase, AFTER freeze/unfreeze + make_optimizer_for_phase
    + EarlyStopping have all run, so the numbers reflect the live state.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    n_tensors = sum(1 for _ in optimizer.param_groups[0]["params"])
    print(f"Trainable:   {trainable:,} / total: {total:,}")
    print(f"Optimizer:   Adam(lr={lr:.0e}, {n_tensors} tensors, 1 param group)")
    print(
        f"EarlyStop:   patience={early_stop.patience}, "
        f"best={early_stop.best:.4f}, counter={early_stop.counter}"
    )


def print_phase_footer(
    phase_name: str,
    n_epochs_ran: int,
    best_val_loss: float,
    best_val_acc: float,
) -> None:
    """Print a one-line summary at the end of a phase."""
    print(
        f"{phase_name} done: {n_epochs_ran} epochs, "
        f"best val_loss={best_val_loss:.4f}, best val_acc={best_val_acc:.4f}"
    )


def print_phase_skipped(
    phase_name: str,
    ckpt_epoch: int,
    threshold: int,
) -> None:
    """Print when a resume checkpoint already finished the phase."""
    print(
        f"{phase_name} already complete in checkpoint "
        f"(epoch {ckpt_epoch} >= {threshold}), skipping to next phase"
    )
