"""Script 5 — SqueezeNet 1.1 (ImageNet pretrained, 2-phase FT, + size comparison).

What it does:
  Trains SqueezeNet 1.1 for binary classification (healthy vs
  needs_annotation) on 128x128 patches. 2-phase fine-tune: phase 1
  freezes the backbone and trains the new 1x1 conv head for 5 epochs
  (lr=1e-3), phase 2 unfreezes all layers with lr=1e-4 and uses
  ReduceLROnPlateau (factor 0.5, patience 5). Evaluates on the
  held-out test split, exports ONNX (fp32), quantizes to dynamic
  int8, and writes model_size_comparison.json showing the .pt /
  fp32 ONNX / int8 ONNX sizes + compression ratio.

  SqueezeNet's classifier is a 1x1 Conv2d (not Linear); the head is
  replaced with nn.Conv2d(512, 2, kernel_size=1) and model.num_classes
  is set to 2.

How to run:
  uv run train_model_script/train_squeezenet.py

  Custom paths / hyperparameters:
  uv run train_model_script/train_squeezenet.py \\
      --csv_path ./output_dataset_final/dataset_binary.csv \\
      --patches_root ./dataset_patches \\
      --output_dir ./outputs/squeezenet1_1 \\
      --epochs 40 --phase1_epochs 5 --batch_size 32 --lr 1e-4

Output structure:
  outputs/squeezenet1_1/
  |-- checkpoints/
  |   `-- best_model.pt           # model with best val_loss (for final eval)
  |-- logs/
  |   |-- training_log.csv        # per-epoch loss/acc/lr
  |   `-- learning_curve.png
  |-- eval/
  |   |-- metrics.json            # test accuracy/precision/recall/f1 + cm
  |   |-- confusion_matrix.png
  |   `-- model_size_comparison.json  # .pt vs fp32 vs int8 ONNX sizes
  `-- exported/
      |-- model.onnx              # fp32
      |-- model_int8.onnx         # dynamic int8
      `-- quantization_meta.json

Dependencies: PyTorch + torchvision, onnx, onnxruntime, onnxscript, tqdm,
              scikit-learn, matplotlib, pandas, numpy, Pillow.

GPU: recommended (CUDA + AMP). First run downloads SqueezeNet1_1
     ImageNet weights (~5 MB, cached in ~/.cache/torch/hub/checkpoints/).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

from _finetune import (
    freeze_backbone,
    make_optimizer_for_phase,
    trainable_count,
    unfreeze_all,
)
from _train_common import (
    EarlyStopping,
    evaluate_test,
    export_onnx,
    load_checkpoint,
    make_output_dirs,
    plot_confusion_matrix,
    plot_learning_curve,
    print_config,
    print_summary,
    quantize_onnx,
    save_checkpoint,
    train_one_epoch,
    validate,
    write_metrics_json,
    write_training_log,
)
from preprocessing import count_parameters, get_dataloaders, set_seed


def build_model() -> nn.Module:
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    final_conv = nn.Conv2d(512, 2, kernel_size=1)
    model.classifier = final_conv
    model.num_classes = 2
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SqueezeNet 1.1 (ImageNet pretrained, 2-phase FT, + size comparison)",
        epilog=(
            "Examples:\n"
            "  uv run train_model_script/train_squeezenet.py\n"
            "  uv run train_model_script/train_squeezenet.py --epochs 10 --batch_size 16"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv_path", type=Path, default=Path("./output_dataset_final/dataset_binary.csv"))
    p.add_argument("--patches_root", type=Path, default=Path("./dataset_patches"))
    p.add_argument("--output_dir", type=Path, default=Path("./outputs"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--phase1_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--phase1_lr", type=float, default=1e-3)
    p.add_argument("--up_sample_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    paths = make_output_dirs(args.output_dir, "squeezenet1_1")
    print_config(
        "squeezenet1_1", args, device, use_amp, paths,
        phase1_epochs=args.phase1_epochs, phase1_lr=args.phase1_lr,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=args.csv_path,
        patches_root=args.patches_root,
        batch_size=args.batch_size,
        up_sample_size=args.up_sample_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print("Loading pretrained weights (first run downloads from PyTorch)...")
    model = build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SqueezeNet 1.1 — trainable (init): {trainable_count(model):,} / total: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_ckpt = paths["checkpoints"] / "best_model.pt"
    best_val_loss = float("inf")
    best_val_acc = 0.0
    log_rows: list[dict] = []
    t_start = time.time()

    print("\n=== Phase 1: head only (frozen backbone) ===")
    freeze_backbone(model, ["features"])
    optimizer = make_optimizer_for_phase(model, args.phase1_lr)
    early_stop = EarlyStopping(patience=args.patience)
    for epoch in range(min(args.phase1_epochs, args.epochs)):
        lr = optimizer.param_groups[0]["lr"]
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        vl, va = validate(model, val_loader, criterion, device, use_amp)
        log_rows.append({"epoch": epoch, "train_loss": round(tl, 6), "train_acc": round(ta, 6),
                         "val_loss": round(vl, 6), "val_acc": round(va, 6), "lr": lr})
        if vl < best_val_loss:
            best_val_loss, best_val_acc = vl, va
            save_checkpoint(model, best_ckpt)
            print(f"  saved best (val_loss={vl:.4f})")
        print(f"phase1 ep {epoch:3d} | tl {tl:.4f} ta {ta:.4f} | vl {vl:.4f} va {va:.4f} | lr {lr:.2e}")

    print("\n=== Phase 2: full fine-tune (ReduceLROnPlateau) ===")
    unfreeze_all(model)
    optimizer = make_optimizer_for_phase(model, args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stop.reset()
    for epoch in range(args.phase1_epochs, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        vl, va = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step(vl)
        log_rows.append({"epoch": epoch, "train_loss": round(tl, 6), "train_acc": round(ta, 6),
                         "val_loss": round(vl, 6), "val_acc": round(va, 6), "lr": lr})
        if vl < best_val_loss:
            best_val_loss, best_val_acc = vl, va
            save_checkpoint(model, best_ckpt)
            print(f"  saved best (val_loss={vl:.4f})")
        print(f"phase2 ep {epoch:3d} | tl {tl:.4f} ta {ta:.4f} | vl {vl:.4f} va {va:.4f} | lr {lr:.2e}")
        if early_stop(vl):
            print(f"Early stop at epoch {epoch}")
            break

    train_time = time.time() - t_start
    trainable, _ = count_parameters(model)
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
    int8_path = paths["exported"] / "model_int8.onnx"
    print(f"Exporting ONNX (fp32) to {onnx_path}...")
    export_onnx(model, onnx_path, args.up_sample_size, device)
    print(f"Quantizing to int8 -> {int8_path}...")
    qmeta = quantize_onnx(onnx_path, int8_path)
    (paths["exported"] / "quantization_meta.json").write_text(json.dumps(qmeta, indent=2))

    size_comparison = {
        "model_name": "squeezenet1_1",
        "pt_size_mb": round(best_ckpt.stat().st_size / 1024 / 1024, 3),
        "onnx_fp32_size_mb": qmeta["fp32_size_mb"],
        "onnx_int8_size_mb": qmeta["int8_size_mb"],
        "fp32_to_int8_ratio": qmeta["compression_ratio"],
        "rationale": "SqueezeNet chosen for smallest deployment footprint; int8 further shrinks it for edge/CPU serving.",
    }
    print("Writing model_size_comparison.json...")
    (paths["eval"] / "model_size_comparison.json").write_text(json.dumps(size_comparison, indent=2))

    print_summary(
        model_name="squeezenet1_1",
        trainable_params=trainable,
        total_params=total_params,
        best_val_acc=best_val_acc,
        test_metrics=test_metrics,
        onnx_fp32_mb=qmeta["fp32_size_mb"],
        onnx_int8_mb=qmeta["int8_size_mb"],
        total_time_s=train_time,
        extras={"pt_size_mb": size_comparison["pt_size_mb"]},
    )


if __name__ == "__main__":
    main()
