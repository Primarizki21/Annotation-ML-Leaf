"""Auto-find max GPU batch size for each pretrained model.

Sweeps --batch_sizes (default [16, 32, 64, 128, 256, 512]) for each model
in REGISTRY. For each batch size, runs --iters forward + backward + step
passes (default 2 for warmup) and records peak GPU memory + step time.
On OOM, stops the sweep for that model and marks the previous successful
batch as max.

Phase 2 only (full unfreeze, Adam optimizer) — the bottleneck of real
training. Phase 1 (frozen backbone) is always much smaller.

Output:
  - Console: progress per (model x batch), summary table at end
  - outputs/batch_size_finder/batch_size_results.csv
      columns: model, batch_size, status, peak_mem_mb, step_time_ms
  - outputs/batch_size_finder/max_batch_per_model.json
      {model_name: max_batch_size}  — easy to load from train scripts

To add a new model:
  1. Import or define its build_model() function (returns nn.Module).
  2. Add one tuple to REGISTRY: (name, build_fn, up_sample_size).
  3. Done.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from train_efficientnet_b0 import build_model as build_efficientnet
from train_mobilenetv3_small import build_model as build_mobilenet
from train_shufflenetv2 import build_model as build_shufflenet
from train_squeezenet import build_model as build_squeezenet

# === EXTENSIBILITY: add new pretrained models here ===
# Each tuple: (display_name, build_fn_returning_nn.Module, up_sample_size)
REGISTRY: list[tuple[str, Callable[[], nn.Module], int]] = [
    ("mobilenetv3_small", build_mobilenet, 128),
    ("efficientnet_b0", build_efficientnet, 160),
    ("shufflenetv2_x1_0", build_shufflenet, 128),
    ("squeezenet1_1", build_squeezenet, 128),
]

DEFAULT_BATCH_SIZES = [16, 32, 64, 128, 256, 512]
PHASE2_LR = 1e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-find max GPU batch size per pretrained model",
        epilog=(
            "Examples:\n"
            "  uv run train_model_script/find_batch_size.py\n"
            "  uv run train_model_script/find_batch_size.py --models efficientnet_b0\n"
            "  uv run train_model_script/find_batch_size.py --batch_sizes 16 32 64 --no_amp"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names to test (default: all in REGISTRY)",
    )
    p.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Candidate batch sizes to sweep (ascending)",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=2,
        help="Forward+backward iters per batch (>=2 recommended for warmup)",
    )
    p.add_argument("--no_amp", action="store_true", help="Disable AMP (test fp32)")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./outputs/batch_size_finder"),
    )
    return p.parse_args()


def test_one_batch(
    model: nn.Module,
    batch_size: int,
    upsample_size: int,
    use_amp: bool,
    iters: int,
    device: torch.device,
) -> tuple[str, float | None, float | None]:
    """Run `iters` forward+backward+step. Return (status, peak_mb, step_ms)."""
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=PHASE2_LR
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats()
    status, peak_mb, step_ms = "OOM", None, None
    x = y = out = loss = None
    try:
        x = torch.randn(batch_size, 3, upsample_size, upsample_size, device=device)
        y = torch.randint(0, 2, (batch_size,), device=device)
        for _ in range(iters):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        status = "OK"
    except torch.cuda.OutOfMemoryError:
        pass
    finally:
        del x, y, out, loss, optimizer, scaler, criterion
        gc.collect()
        torch.cuda.empty_cache()
    return status, peak_mb, step_ms


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script requires a CUDA GPU.")

    use_amp = not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        wanted = set(m.strip() for m in args.models.split(","))
        registry = [r for r in REGISTRY if r[0] in wanted]
        if not registry:
            raise ValueError(
                f"No models matched --models={args.models!r}. "
                f"Available: {[r[0] for r in REGISTRY]}"
            )
    else:
        registry = REGISTRY

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print("=" * 70)
    print(
        f"Batch size finder | device={gpu_name} ({gpu_mem_gb:.1f} GB) "
        f"| AMP={'on' if use_amp else 'off'} | iters={args.iters}"
    )
    print(f"Candidates: {args.batch_sizes}")
    print(f"Models: {[r[0] for r in registry]}")
    print("=" * 70)

    results: list[dict] = []
    max_per_model: dict[str, int | None] = {}

    for name, build_fn, upsample in registry:
        print(f"\n--- {name} (up_sample={upsample}) ---")
        torch.cuda.empty_cache()
        model = build_fn().to(device)
        for p in model.parameters():
            p.requires_grad = True
        prev_ok: tuple[int, float, float] | None = None
        for bs in args.batch_sizes:
            status, peak_mb, step_ms = test_one_batch(
                model, bs, upsample, use_amp, args.iters, device
            )
            if status == "OK":
                print(
                    f"  bs={bs:4d}  OK    peak={peak_mb:7.1f} MB  step={step_ms:7.2f} ms"
                )
                prev_ok = (bs, peak_mb, step_ms)
                results.append(
                    {
                        "model": name,
                        "batch_size": bs,
                        "status": "OK",
                        "peak_mem_mb": round(peak_mb, 1),
                        "step_time_ms": round(step_ms, 2),
                    }
                )
            else:
                print(
                    f"  bs={bs:4d}  OOM   (max stays at "
                    f"{prev_ok[0] if prev_ok else 'N/A'})"
                )
                results.append(
                    {
                        "model": name,
                        "batch_size": bs,
                        "status": "OOM",
                        "peak_mem_mb": None,
                        "step_time_ms": None,
                    }
                )
                break
        max_per_model[name] = prev_ok[0] if prev_ok else None
        if prev_ok:
            print(
                f"  -> max batch_size = {prev_ok[0]} "
                f"(peak {prev_ok[1]:.1f} MB, step {prev_ok[2]:.2f} ms)"
            )
        else:
            print(
                f"  -> no batch size in {args.batch_sizes} fit "
                f"(OOM at smallest {args.batch_sizes[0]})"
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    csv_path = args.output_dir / "batch_size_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "batch_size",
                "status",
                "peak_mem_mb",
                "step_time_ms",
            ],
        )
        w.writeheader()
        w.writerows(results)

    json_path = args.output_dir / "max_batch_per_model.json"
    json_path.write_text(json.dumps(max_per_model, indent=2))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, max_bs in max_per_model.items():
        if max_bs is not None:
            peak = next(
                r["peak_mem_mb"]
                for r in results
                if r["model"] == name and r["batch_size"] == max_bs
            )
            print(
                f"  {name:25s}  max_batch_size = {max_bs:4d}   peak_mem = {peak:.1f} MB"
            )
        else:
            print(f"  {name:25s}  no fit (all OOM)")
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
