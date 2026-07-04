"""Shared training-loop, eval, ONNX export, and quantization helpers.

Used by all train_*.py scripts. Each training script owns its model
definition and CLI parsing; this module owns the parts that must stay
identical across the 5 architectures so that the comparison table is
meaningful.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_output_dirs(output_root: Path, model_name: str) -> dict[str, Path]:
    base = output_root / model_name
    paths = {
        "base": base,
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
        "eval": base / "eval",
        "exported": base / "exported",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def print_config(
    model_name: str,
    args,
    device: torch.device,
    use_amp: bool,
    paths: dict[str, Path],
    **extras,
) -> None:
    print("\n" + "=" * 60)
    print(f"  {model_name} — Configuration")
    print("=" * 60)
    cfg = {
        "csv_path": str(args.csv_path),
        "patches_root": str(args.patches_root),
        "output_dir": str(paths["base"]),
        "device": f"{device} (AMP={'on' if use_amp else 'off'})",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "up_sample_size": getattr(args, "up_sample_size", "n/a"),
        "num_workers": args.num_workers,
        "patience": args.patience,
        "seed": args.seed,
    }
    cfg.update(extras)
    for k, v in cfg.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, float]:
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    autocast = torch.amp.autocast("cuda", enabled=use_amp)
    bar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            logits = model(x)
            loss = criterion(logits, y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_m.update(loss.item(), x.size(0))
        acc_m.update((logits.argmax(1) == y).float().mean().item(), x.size(0))
        bar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.4f}")
    bar.close()
    return loss_m.avg, acc_m.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, float]:
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    autocast = torch.amp.autocast("cuda", enabled=use_amp)
    bar = tqdm(loader, desc="val  ", leave=False, dynamic_ncols=True)
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast:
            logits = model(x)
            loss = criterion(logits, y)
        loss_m.update(loss.item(), x.size(0))
        acc_m.update((logits.argmax(1) == y).float().mean().item(), x.size(0))
        bar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.4f}")
    bar.close()
    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    autocast = torch.amp.autocast("cuda", enabled=use_amp)
    bar = tqdm(loader, desc="test ", leave=False, dynamic_ncols=True)
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        with autocast:
            logits = model(x)
        y_true.extend(y.tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    bar.close()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)


def load_checkpoint(
    model: nn.Module, path: Path, device: torch.device, strict: bool = True
) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    model.to(device)


def save_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    early_stop: "EarlyStopping",
    phase: int,
    path: Path,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Save full training state for resume (model + optimizer + scaler +
    scheduler + epoch + best_val_loss + early-stop state + phase)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "early_stop_counter": early_stop.counter,
        "early_stop_best": early_stop.best,
        "phase": phase,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def load_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    early_stop: "EarlyStopping",
    path: Path,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    """Load full training state from a periodic checkpoint. Restores
    model + optimizer + scaler (if AMP) + scheduler (if provided) +
    early-stop counter/best. Returns the full checkpoint dict so the
    caller can read `epoch` and `phase`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler.is_enabled() and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    early_stop.counter = ckpt.get("early_stop_counter", 0)
    early_stop.best = ckpt.get("early_stop_best", float("inf"))
    return ckpt


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the highest-epoch last_NNNN.pt in checkpoint_dir, or None."""
    if not checkpoint_dir.exists():
        return None
    files = sorted(checkpoint_dir.glob("last_*.pt"))
    return files[-1] if files else None


def cleanup_old_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    """Delete oldest last_*.pt files, keep only `keep` most recent (FIFO)."""
    files = sorted(checkpoint_dir.glob("last_*.pt"))
    while len(files) > keep:
        files[0].unlink()
        files.pop(0)


def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    up_sample_size: int,
    device: torch.device,
    opset: int = 17,
) -> None:
    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, up_sample_size, up_sample_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )


def quantize_onnx(fp32_path: Path, int8_path: Path) -> dict:
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    return {
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "fp32_size_mb": round(fp32_size / 1024 / 1024, 3),
        "int8_size_mb": round(int8_size / 1024 / 1024, 3),
        "compression_ratio": round(fp32_size / int8_size, 3) if int8_size else None,
        "quant_type": "dynamic_int8",
    }


def write_training_log(log_path: Path, rows: list[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"],
        )
        w.writeheader()
        w.writerows(rows)


def plot_learning_curve(log_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with log_path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    epochs = [int(r["epoch"]) for r in rows]
    tl = [float(r["train_loss"]) for r in rows]
    vl = [float(r["val_loss"]) for r in rows]
    ta = [float(r["train_acc"]) for r in rows]
    va = [float(r["val_acc"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, tl, label="train")
    ax1.plot(epochs, vl, label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.set_title("Loss")
    ax2.plot(epochs, ta, label="train")
    ax2.plot(epochs, va, label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def write_metrics_json(metrics_path: Path, metrics: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))


def plot_confusion_matrix(
    cm: list[list[int]], classes: list[str], png_path: Path
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            color = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "black"
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self) -> None:
        self.counter = 0
        self.best = float("inf")


def print_summary(
    model_name: str,
    trainable_params: int,
    total_params: int,
    best_val_acc: float,
    test_metrics: dict,
    onnx_fp32_mb: float | None,
    onnx_int8_mb: float | None,
    total_time_s: float,
    extras: dict | None = None,
) -> None:
    print("\n" + "=" * 60)
    print(f"  {model_name} — SUMMARY")
    print("=" * 60)
    print(f"  Parameters     : {trainable_params:,} trainable / {total_params:,} total")
    print(f"  Best val_acc   : {best_val_acc:.4f}")
    print(f"  Test accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Test precision : {test_metrics['precision']:.4f}")
    print(f"  Test recall    : {test_metrics['recall']:.4f}")
    print(f"  Test F1        : {test_metrics['f1_score']:.4f}")
    print(f"  Test cm        : {test_metrics['confusion_matrix']}")
    if onnx_fp32_mb is not None:
        print(f"  ONNX fp32 size : {onnx_fp32_mb:.3f} MB")
    if onnx_int8_mb is not None:
        print(f"  ONNX int8 size : {onnx_int8_mb:.3f} MB")
    print(f"  Total time     : {total_time_s:.1f} s ({total_time_s/60:.1f} min)")
    if extras:
        for k, v in extras.items():
            print(f"  {k:14s}: {v}")
    print("=" * 60)
