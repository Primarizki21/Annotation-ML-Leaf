"""Benchmark pure data-loading throughput (no model forward/backward).

Iterates a DataLoader for --max_batches batches and reports samples/sec,
ms/batch, and peak RAM delta. Use before/after applying the I/O fix in
preprocessing.py to compare numbers directly.

  uv run train_model_script/benchmark_dataloader.py
  uv run train_model_script/benchmark_dataloader.py --batch_size 64 --num_workers 8
  uv run train_model_script/benchmark_dataloader.py --eval_mode --max_batches 200

Default mode uses train transforms (with augmentation) to reflect the
real training cost. --eval_mode skips augmentation for a fair compare
against val/test loops.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessing import PatchDataset, get_transforms, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark DataLoader throughput (no model)",
        epilog=(
            "Examples:\n"
            "  uv run train_model_script/benchmark_dataloader.py\n"
            "  uv run train_model_script/benchmark_dataloader.py --batch_size 64 --num_workers 8\n"
            "  uv run train_model_script/benchmark_dataloader.py --eval_mode --max_batches 200"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv_path", type=Path, default=Path("./output_dataset_final/dataset_binary.csv"))
    p.add_argument("--patches_root", type=Path, default=Path("./dataset_patches"))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--up_sample_size", type=int, default=128)
    p.add_argument("--max_batches", type=int, default=100)
    p.add_argument("--eval_mode", action="store_true",
                   help="Use val/test transforms (no augmentation) instead of train")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    mode = "val/test" if args.eval_mode else "train"
    print("=" * 60)
    print(f"DataLoader benchmark | mode={mode} | batch={args.batch_size} "
          f"| workers={args.num_workers} | up_sample={args.up_sample_size}")
    print("=" * 60)

    ds = PatchDataset(
        args.csv_path, args.patches_root, split="train",
        transform=get_transforms(args.up_sample_size, train=not args.eval_mode),
    )
    print(f"Dataset: {len(ds):,} train rows (single loader, no shuffle)")

    loader_kw: dict = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 4
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    # Warmup: 1 batch to spin up workers + filesystem cache, not timed.
    it = iter(loader)
    _ = next(it, None)

    # Measure: iterate --max_batches, real wall clock
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    n_samples = 0
    n_batches = 0
    for batch in it:
        x, _y = batch
        n_samples += x.size(0)
        n_batches += 1
        if n_batches >= args.max_batches:
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    samples_per_s = n_samples / elapsed if elapsed > 0 else float("nan")
    ms_per_batch = (elapsed * 1000.0) / n_batches if n_batches > 0 else float("nan")
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available() else None
    )

    print(f"  batches run    : {n_batches}")
    print(f"  samples loaded : {n_samples:,}")
    print(f"  total time     : {elapsed:.3f} s")
    print(f"  throughput     : {samples_per_s:,.1f} samples/sec")
    print(f"  per batch      : {ms_per_batch:.2f} ms")
    if peak_mb is not None:
        print(f"  peak GPU mem   : {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
