"""Shared training-side utilities for 5-architecture plant disease patch
classification.

Reads output_dataset_final/dataset_binary.csv (already filtered + 70/15/15
leak-free split by source leaf) and lazily opens patches from
dataset_patches/. No black-pixel re-filtering here -- the patches on disk
have already been pre-filtered upstream by patch_splitting.py.

Inference-side helpers (full-image splitting, prevalence rate) live in
preprocess_inference.py.

I/O optimization note (perf commit):
  PatchDataset.__init__ precomputes `self.paths` (numpy array of full
  path strings) and `self.labels` (numpy int64) once. __getitem__ then
  does O(1) array indexing instead of `self.df.iloc[idx]`, which avoids
  the per-row pandas overhead that bottlenecks the loader when called
  millions of times per epoch. No image is cached -- only the path/label
  metadata, so RAM cost is ~50 MB for 1 M rows. get_dataloaders also sets
  `prefetch_factor=4` (when num_workers > 0) so each worker keeps 4
  batches ready, smoothing GPU bursts.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(up_sample_size: int, train: bool) -> v2.Compose:
    ops: list = [
        v2.Resize(
            (up_sample_size, up_sample_size),
            antialias=True,
            interpolation=v2.InterpolationMode.BILINEAR,
        ),
        v2.ToImage(),
    ]
    if train:
        ops += [
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ]
    ops += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
    return v2.Compose(ops)


class PatchDataset(Dataset):
    """Lazy CSV-driven dataset.

    Filters dataset_binary.csv to a single split and opens each patch JPG
    on-the-fly in __getitem__. No black-pixel re-filter (patches are
    pre-filtered on disk).
    """

    def __init__(
        self,
        csv_path: str | Path,
        patches_root: str | Path,
        split: str,
        transform: v2.Compose | None = None,
    ) -> None:
        self.patches_root = Path(patches_root)
        df = pd.read_csv(csv_path)
        if "split" not in df.columns:
            raise ValueError(f"{csv_path} has no 'split' column")
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows with split='{split}' in {csv_path}")
        sample = self.patches_root / self.df.iloc[0]["patch_path"]
        if not sample.exists():
            raise FileNotFoundError(
                f"Sample patch not found: {sample} "
                f"(patches_root={self.patches_root})"
            )
        # Precompute numpy arrays (one-time cost in __init__, not per __getitem__).
        # Avoids the per-row pandas overhead of self.df.iloc[idx] when called
        # millions of times per epoch. Only path/label metadata is cached;
        # images are still read lazily from disk per item.
        self.paths = np.array(
            [str(self.patches_root / p) for p in self.df["patch_path"]],
            dtype=object,
        )
        self.labels = self.df["label"].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def get_dataloaders(
    csv_path: str | Path,
    patches_root: str | Path,
    batch_size: int,
    up_sample_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train, val, test) DataLoaders from the CSV's existing split column."""
    print(f"Loading dataset from {csv_path}...")
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = PatchDataset(
        csv_path, patches_root, split="train",
        transform=get_transforms(up_sample_size, train=True),
    )
    val_ds = PatchDataset(
        csv_path, patches_root, split="val",
        transform=get_transforms(up_sample_size, train=False),
    )
    test_ds = PatchDataset(
        csv_path, patches_root, split="test",
        transform=get_transforms(up_sample_size, train=False),
    )
    print(
        f"Dataset sizes — train: {len(train_ds):,} | "
        f"val: {len(val_ds):,} | test: {len(test_ds):,}"
    )

    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    # prefetch_factor only valid when num_workers > 0; default 2 -> 4 buffers
    # per worker so the GPU is less likely to wait for the next batch.
    if num_workers > 0:
        loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, generator=g, **loader_kw,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kw)
    return train_loader, val_loader, test_loader


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
