"""Script 1 — SmallInception (custom CNN, trained from scratch).

What it does:
  Trains the SmallInception architecture for binary classification
  (healthy vs needs_annotation) on native 32x32 patches. 1-phase
  training from scratch with Adam (lr=1e-3, 300 epochs max, early
  stop at patience 5). Evaluates on the held-out test split,
  exports ONNX (fp32 only — no int8 per the paper's baseline
  replication).

How to run:
  uv run train_model_script/train_small_inception.py

  Custom paths / hyperparameters:
  uv run train_model_script/train_small_inception.py \\
      --csv_path ./output_dataset_final/dataset_binary.csv \\
      --patches_root ./dataset_patches \\
      --output_dir ./outputs/small_inception \\
      --epochs 300 --batch_size 32 --lr 1e-3

  Resume from last periodic checkpoint (auto-detected on startup):
  uv run train_model_script/train_small_inception.py
  Force fresh start (ignore existing checkpoints):
  uv run train_model_script/train_small_inception.py --no_resume

  Periodic save controls:
  uv run train_model_script/train_small_inception.py --save_every 5 --max_checkpoints 3

Output structure:
  outputs/small_inception/
  |-- checkpoints/
  |   |-- best_model.pt           # model with best val_loss (for final eval)
  |   |-- last_NNNN.pt            # full state, every --save_every epochs
  |   `-- ...                     # (FIFO: max --max_checkpoints kept)
  |-- logs/
  |   |-- training_log.csv        # per-epoch loss/acc/lr
  |   `-- learning_curve.png
  |-- eval/
  |   |-- metrics.json            # test accuracy/precision/recall/f1 + cm
  |   `-- confusion_matrix.png
  `-- exported/
      `-- model.onnx              # fp32 only (no int8)

Dependencies: PyTorch + torchvision, onnx, onnxruntime, onnxscript, tqdm,
              scikit-learn, matplotlib, pandas, numpy, Pillow.

GPU: recommended (CUDA + AMP). Falls back to CPU but will be slow.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from _train_common import (
    EarlyStopping,
    cleanup_old_checkpoints,
    evaluate_test,
    export_onnx,
    find_latest_checkpoint,
    load_checkpoint,
    load_state,
    make_output_dirs,
    plot_confusion_matrix,
    plot_learning_curve,
    print_config,
    print_summary,
    save_checkpoint,
    save_state,
    train_one_epoch,
    validate,
    write_metrics_json,
    write_training_log,
)
from preprocessing import (
    count_parameters,
    get_dataloaders,
    get_transforms,
    set_seed,
)


# ── Model: per spec BAGIAN 3 ────────────────────────────────────────────────
class ConvModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2 if stride == 1 else 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class InceptionModule(nn.Module):
    """1x1 + 3x3 parallel branches, concat on channel axis."""

    def __init__(self, in_ch: int, half_out: int) -> None:
        super().__init__()
        self.b1 = ConvModule(in_ch, half_out, 1)
        self.b2 = ConvModule(in_ch, half_out, 3)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x)], dim=1)


class DownsampleModule(nn.Module):
    """3x3 stride-2 conv (preserves channels) + 3x3 stride-2 maxpool
    (preserves channels), concat doubles channels. No padding so
    32x32 -> 15x15, 15x15 -> 7x7."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return torch.cat([self.conv(x), self.pool(x)], dim=1)


class SmallInception(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = ConvModule(3, 96, 3)            # 32x32x3   -> 32x32x96
        self.inc1 = InceptionModule(96, 32)         # -> 32x32x64
        self.inc2 = InceptionModule(64, 40)         # -> 32x32x80
        self.down1 = DownsampleModule(80)           # -> 15x15x160
        self.inc3 = InceptionModule(160, 80)        # -> 15x15x160
        self.inc4 = InceptionModule(160, 80)        # -> 15x15x160
        self.inc5 = InceptionModule(160, 80)        # -> 15x15x160
        self.inc6 = InceptionModule(160, 84)        # -> 15x15x168
        self.down2 = DownsampleModule(168)          # -> 7x7x336
        self.inc7 = InceptionModule(336, 168)       # -> 7x7x336
        self.inc8 = InceptionModule(336, 168)       # -> 7x7x336
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(336, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x); x = self.inc2(x)
        x = self.down1(x)
        x = self.inc3(x); x = self.inc4(x); x = self.inc5(x); x = self.inc6(x)
        x = self.down2(x)
        x = self.inc7(x); x = self.inc8(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SmallInception from scratch on plant-disease patches",
        epilog=(
            "Examples:\n"
            "  uv run train_model_script/train_small_inception.py\n"
            "  uv run train_model_script/train_small_inception.py --epochs 10 --batch_size 16\n"
            "  uv run train_model_script/train_small_inception.py --no_resume"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv_path", type=Path, default=Path("./output_dataset_final/dataset_binary.csv"))
    p.add_argument("--patches_root", type=Path, default=Path("./dataset_patches"))
    p.add_argument("--output_dir", type=Path, default=Path("./outputs"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--max_checkpoints", type=int, default=3)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    paths = make_output_dirs(args.output_dir, "small_inception")
    print_config("small_inception", args, device, use_amp, paths)

    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=args.csv_path,
        patches_root=args.patches_root,
        batch_size=args.batch_size,
        up_sample_size=32,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = SmallInception(num_classes=2).to(device)
    trainable, total = count_parameters(model)
    print(f"SmallInception — trainable: {trainable:,} / total: {total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_stop = EarlyStopping(patience=args.patience)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    log_rows: list[dict] = []
    best_ckpt = paths["checkpoints"] / "best_model.pt"
    t_start = time.time()

    start_epoch = 0
    latest_ckpt = None if args.no_resume else find_latest_checkpoint(paths["checkpoints"])
    if latest_ckpt:
        ckpt = load_state(model, optimizer, scaler, early_stop, latest_ckpt, device, scheduler=None)
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {latest_ckpt.name} at epoch {start_epoch} (val_loss={best_val_loss:.4f})")

    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        vl, va = validate(model, val_loader, criterion, device, use_amp)
        log_rows.append({"epoch": epoch, "train_loss": round(tl, 6), "train_acc": round(ta, 6),
                         "val_loss": round(vl, 6), "val_acc": round(va, 6), "lr": lr})
        if vl < best_val_loss:
            best_val_acc = va
            best_val_loss = vl
            save_checkpoint(model, best_ckpt)
            print(f"  saved best (val_loss={vl:.4f})")
        print(f"epoch {epoch:3d} | tl {tl:.4f} ta {ta:.4f} | vl {vl:.4f} va {va:.4f} | lr {lr:.2e}")
        if (epoch + 1) % args.save_every == 0:
            state_path = paths["checkpoints"] / f"last_{(epoch+1):04d}.pt"
            save_state(model, optimizer, scaler, epoch + 1, best_val_loss, early_stop, 1, state_path, scheduler=None)
            cleanup_old_checkpoints(paths["checkpoints"], args.max_checkpoints)
            print(f"  saved periodic checkpoint: {state_path.name}")
        if early_stop(vl):
            print(f"Early stop at epoch {epoch} (patience {args.patience})")
            break

    train_time = time.time() - t_start
    print("Writing training log...")
    write_training_log(paths["logs"] / "training_log.csv", log_rows)
    print("Generating learning curve...")
    plot_learning_curve(paths["logs"] / "training_log.csv", paths["logs"] / "learning_curve.png")

    load_checkpoint(model, best_ckpt, device)
    print("Evaluating on test set...")
    test_metrics = evaluate_test(model, test_loader, device, use_amp)
    print("Writing metrics.json...")
    write_metrics_json(paths["eval"] / "metrics.json", test_metrics)
    print("Generating confusion matrix...")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        ["healthy (0)", "unhealthy (1)"],
        paths["eval"] / "confusion_matrix.png",
    )

    onnx_path = paths["exported"] / "model.onnx"
    print(f"Exporting ONNX (fp32) to {onnx_path}...")
    export_onnx(model, onnx_path, up_sample_size=32, device=device)
    onnx_mb = onnx_path.stat().st_size / 1024 / 1024

    print_summary(
        model_name="small_inception",
        trainable_params=trainable,
        total_params=total,
        best_val_acc=best_val_acc,
        test_metrics=test_metrics,
        onnx_fp32_mb=onnx_mb,
        onnx_int8_mb=None,
        total_time_s=train_time,
        extras={"note": "fp32 ONNX only (paper baseline replication)"},
    )


if __name__ == "__main__":
    main()
