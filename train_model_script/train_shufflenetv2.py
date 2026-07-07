"""Script 4 — ShuffleNetV2-x1.0 (ImageNet pretrained, 2-phase FT, + benchmark).

What it does:
  Trains ShuffleNetV2-x1.0 for binary classification (healthy vs
  needs_annotation) on 128x128 patches. 2-phase fine-tune: phase 1
  freezes the backbone and trains the new linear head for 5 epochs
  (lr=1e-3), phase 2 unfreezes all layers with lr=1e-4 and uses
  ReduceLROnPlateau (factor 0.5, patience 5). Evaluates on the
  held-out test split, exports ONNX (fp32), quantizes to dynamic
  int8, and runs a 50-iteration CPU inference-speed benchmark on
  the exported ONNX model.

How to run:
  uv run train_model_script/train_shufflenetv2.py

  Custom paths / hyperparameters:
  uv run train_model_script/train_shufflenetv2.py \\
      --csv_path ./output_dataset_final/dataset_binary.csv \\
      --patches_root ./dataset_patches \\
      --output_dir ./outputs/shufflenetv2_x1_0 \\
      --epochs 40 --phase1_epochs 5 --batch_size 32 --lr 1e-4

  Inference benchmark controls:
  uv run train_model_script/train_shufflenetv2.py \\
      --bench_runs 100 --bench_batch 32

  Resume from last periodic checkpoint (auto-detected on startup):
  uv run train_model_script/train_shufflenetv2.py
  Force fresh start (ignore existing checkpoints):
  uv run train_model_script/train_shufflenetv2.py --no_resume

  Periodic save controls:
  uv run train_model_script/train_shufflenetv2.py --save_every 5 --max_checkpoints 3

Output structure:
  outputs/shufflenetv2_x1_0/
  |-- checkpoints/
  |   |-- best_model.pt           # model with best val_loss (for final eval)
  |   |-- last_NNNN.pt            # full state, every --save_every epochs
  |   `-- ...                     # (FIFO: max --max_checkpoints kept)
  |-- logs/
  |   |-- training_log.csv        # per-epoch loss/acc/lr
  |   `-- learning_curve.png
  |-- eval/
  |   |-- metrics.json            # test accuracy/precision/recall/f1 + cm
  |   |-- confusion_matrix.png
  |   `-- inference_benchmark.json  # mean/std/p50 ms over 50 runs
  `-- exported/
      |-- model.onnx              # fp32
      |-- model_int8.onnx         # dynamic int8
      `-- quantization_meta.json

Dependencies: PyTorch + torchvision, onnx, onnxruntime, onnxscript, tqdm,
              scikit-learn, matplotlib, pandas, numpy, Pillow.

GPU: recommended (CUDA + AMP). First run downloads ShuffleNet_V2_X1_0
     ImageNet weights (~5 MB, cached in ~/.cache/torch/hub/checkpoints/).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0

from _finetune import (
    freeze_backbone,
    make_optimizer_for_phase,
    print_phase_footer,
    print_phase_skipped,
    print_phase_state,
    trainable_count,
    unfreeze_all,
)
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
    quantize_onnx,
    save_checkpoint,
    save_state,
    train_one_epoch,
    validate,
    write_metrics_json,
    write_training_log,
)
from preprocessing import count_parameters, get_dataloaders, set_seed


def build_model() -> nn.Module:
    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ShuffleNetV2-x1.0 (ImageNet pretrained, 2-phase FT, + inference benchmark)",
        epilog=(
            "Examples:\n"
            "  uv run train_model_script/train_shufflenetv2.py\n"
            "  uv run train_model_script/train_shufflenetv2.py --epochs 10 --batch_size 16\n"
            "  uv run train_model_script/train_shufflenetv2.py --no_resume"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv_path", type=Path, default=Path("./output_dataset_final/dataset_binary.csv"))
    p.add_argument("--patches_root", type=Path, default=Path("./dataset_patches"))
    p.add_argument("--output_dir", type=Path, default=Path("./outputs"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--phase1_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--phase1_lr", type=float, default=1e-3)
    p.add_argument("--up_sample_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--max_checkpoints", type=int, default=3)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bench_runs", type=int, default=50)
    p.add_argument("--bench_batch", type=int, default=64)
    return p.parse_args()


def run_inference_benchmark(
    onnx_path: Path, up_sample_size: int, batch_size: int, runs: int
) -> dict:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(batch_size, 3, up_sample_size, up_sample_size).astype(np.float32)
    # warmup
    for _ in range(3):
        sess.run(None, {"input": x})
    times_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "onnx_path": str(onnx_path),
        "batch_size": batch_size,
        "up_sample_size": up_sample_size,
        "runs": runs,
        "mean_ms": round(float(np.mean(times_ms)), 3),
        "std_ms": round(float(np.std(times_ms)), 3),
        "min_ms": round(float(np.min(times_ms)), 3),
        "max_ms": round(float(np.max(times_ms)), 3),
        "p50_ms": round(float(np.percentile(times_ms, 50)), 3),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    paths = make_output_dirs(args.output_dir, "shufflenetv2_x1_0")
    print_config(
        "shufflenetv2_x1_0", args, device, use_amp, paths,
        phase1_epochs=args.phase1_epochs, phase1_lr=args.phase1_lr,
        bench_runs=args.bench_runs, bench_batch=args.bench_batch,
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
    print(f"ShuffleNetV2-x1.0 — trainable (init): {trainable_count(model):,} / total: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_ckpt = paths["checkpoints"] / "best_model.pt"
    best_val_loss = float("inf")
    best_val_acc = 0.0
    log_rows: list[dict] = []
    t_start = time.time()

    latest_ckpt = None if args.no_resume else find_latest_checkpoint(paths["checkpoints"])
    ckpt_meta = None
    if latest_ckpt:
        ckpt_meta = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_meta["model_state_dict"])
        best_val_loss = ckpt_meta.get("best_val_loss", float("inf"))
        print(
            f"Found existing checkpoint: {latest_ckpt.name} "
            f"(phase {ckpt_meta.get('phase', 1)}, epoch {ckpt_meta.get('epoch', 0)}, "
            f"val_loss={best_val_loss:.4f})"
        )

    phase1_start = args.phase1_epochs
    if ckpt_meta and ckpt_meta.get("phase", 1) == 1 and ckpt_meta.get("epoch", 0) < args.phase1_epochs:
        phase1_start = ckpt_meta.get("epoch", 0)
    phase1_end = min(args.phase1_epochs, args.epochs)
    n_phase1 = max(phase1_end - phase1_start, 0)
    print(
        f"\n=== Phase 1: head only (frozen backbone), "
        f"epochs {phase1_start}-{phase1_end - 1} ({n_phase1} total) ==="
    )
    freeze_backbone(model, ["conv1", "stage2", "stage3", "stage4"])
    optimizer = make_optimizer_for_phase(model, args.phase1_lr)
    early_stop = EarlyStopping(patience=args.patience)
    print_phase_state(model, optimizer, early_stop, args.phase1_lr)

    phase1_early_stopped = False
    if n_phase1 == 0:
        print_phase_skipped("Phase 1", phase1_start, phase1_end)
    elif ckpt_meta and ckpt_meta.get("phase", 1) == 1:
        load_state(model, optimizer, scaler, early_stop, latest_ckpt, device, scheduler=None)
        print(f"Resuming phase 1 at epoch {phase1_start}")

    phase1_epochs_ran = 0
    for epoch in range(phase1_start, phase1_end):
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
        if (epoch + 1) % args.save_every == 0:
            state_path = paths["checkpoints"] / f"last_{(epoch+1):04d}.pt"
            save_state(model, optimizer, scaler, epoch + 1, best_val_loss, early_stop, 1, state_path, scheduler=None)
            cleanup_old_checkpoints(paths["checkpoints"], args.max_checkpoints)
            print(f"  saved periodic checkpoint: {state_path.name}")
        if early_stop(vl):
            print(f"Early stop at phase 1 epoch {epoch}")
            phase1_early_stopped = True
            break
        phase1_epochs_ran += 1
    print_phase_footer("Phase 1", phase1_epochs_ran, best_val_loss, best_val_acc)

    if not phase1_early_stopped and args.phase1_epochs < args.epochs:
        phase2_start = args.phase1_epochs
        if ckpt_meta and ckpt_meta.get("phase", 1) == 2 and ckpt_meta.get("epoch", 0) >= args.phase1_epochs:
            phase2_start = ckpt_meta.get("epoch", 0)
        n_phase2 = max(args.epochs - phase2_start, 0)
        print(
            f"\n=== Phase 2: full fine-tune (ReduceLROnPlateau), "
            f"epochs {phase2_start}-{args.epochs - 1} ({n_phase2} total) ==="
        )
        unfreeze_all(model)
        optimizer = make_optimizer_for_phase(model, args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        early_stop = EarlyStopping(patience=args.patience)
        print_phase_state(model, optimizer, early_stop, args.lr)

        if n_phase2 == 0:
            print_phase_skipped("Phase 2", phase2_start, args.epochs)
        elif ckpt_meta and ckpt_meta.get("phase", 1) == 2:
            load_state(model, optimizer, scaler, early_stop, latest_ckpt, device, scheduler=scheduler)
            print(f"Resuming phase 2 at epoch {phase2_start}")

        phase2_epochs_ran = 0
        for epoch in range(phase2_start, args.epochs):
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
            if (epoch + 1) % args.save_every == 0:
                state_path = paths["checkpoints"] / f"last_{(epoch+1):04d}.pt"
                save_state(model, optimizer, scaler, epoch + 1, best_val_loss, early_stop, 2, state_path, scheduler=scheduler)
                cleanup_old_checkpoints(paths["checkpoints"], args.max_checkpoints)
                print(f"  saved periodic checkpoint: {state_path.name}")
            if early_stop(vl):
                print(f"Early stop at phase 2 epoch {epoch}")
                break
            phase2_epochs_ran += 1
        print_phase_footer("Phase 2", phase2_epochs_ran, best_val_loss, best_val_acc)

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

    print(f"\nRunning inference benchmark ({args.bench_runs} runs, batch={args.bench_batch})...")
    bench = run_inference_benchmark(
        paths["exported"] / "model.onnx",
        args.up_sample_size,
        args.bench_batch,
        args.bench_runs,
    )
    (paths["eval"] / "inference_benchmark.json").write_text(json.dumps(bench, indent=2))

    print_summary(
        model_name="shufflenetv2_x1_0",
        trainable_params=trainable,
        total_params=total_params,
        best_val_acc=best_val_acc,
        test_metrics=test_metrics,
        onnx_fp32_mb=qmeta["fp32_size_mb"],
        onnx_int8_mb=qmeta["int8_size_mb"],
        total_time_s=train_time,
        extras={"inference_mean_ms": bench["mean_ms"], "inference_std_ms": bench["std_ms"]},
    )


if __name__ == "__main__":
    main()
