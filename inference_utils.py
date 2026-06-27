#!/usr/bin/env python3
"""
inference_utils.py

Shared inference helpers for the predict_unlabeled_patches.py and
active_learning_round.py pipelines. This module is the single source
of truth for: model loading, dataset construction, transforms, patch
path collection, consensus-set exclusion, and chunked CSV writing.

Intentionally contains no plotting, no CLI, no main() — those are
script-specific concerns.
"""

import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import efficientnet_b0
from torchvision.transforms import v2

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "round1" / "model.pt"
DEFAULT_CONSENSUS_CSV = BASE_DIR / "consensus_round1_master.csv"
DEFAULT_PATCHES_DIR = BASE_DIR / "dataset_patches"
PREDICTIONS_DIR = BASE_DIR / "predictions"

CHUNK_SIZE = 50_000


def build_consensus_set(consensus_csv):
    df = pd.read_csv(consensus_csv)
    paths = set(df["patch_path"].tolist())
    return paths


def collect_patch_paths(patches_dir, consensus_set):
    """Scan dataset_patches/ for healthy + needs_annotation subdirs.

    Returns (train_healthy, train_unlabeled, test_healthy, test_unlabeled).
    Healthy patches are collected with agreement_level="auto_healthy" so the
    CSV has a full picture, but they are NOT sent through the model.
    """
    results = {
        "train_healthy": [], "train_unlabeled": [],
        "test_healthy": [], "test_unlabeled": [],
    }

    for split in ["train", "test"]:
        split_dir = patches_dir / split
        if not split_dir.exists():
            continue

        for subdir_name, key_label in [("healthy", "auto_healthy"), ("needs_annotation", "unlabeled")]:
            sub_dir = split_dir / subdir_name
            if not sub_dir.exists():
                continue

            for img in sorted(sub_dir.rglob("*.jpg")):
                rel_path = str(img.relative_to(patches_dir))
                class_name = img.parent.name

                if subdir_name == "healthy":
                    results[f"{split}_healthy"].append({
                        "abs_path": str(img),
                        "rel_path": rel_path, "class_name": class_name,
                        "split": split, "agreement_level": "auto_healthy",
                        "is_consensus": False,
                    })
                else:
                    if rel_path in consensus_set:
                        continue
                    results[f"{split}_unlabeled"].append({
                        "abs_path": str(img),
                        "rel_path": rel_path, "class_name": class_name,
                        "split": split, "agreement_level": "unlabeled",
                        "is_consensus": False,
                    })

    return (
        results["train_healthy"], results["train_unlabeled"],
        results["test_healthy"], results["test_unlabeled"],
    )


def build_transform():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


class InferenceDataset(Dataset):
    def __init__(self, path_entries, transform):
        self.entries = path_entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path = self.entries[idx]["abs_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx


def load_model(checkpoint_path, device):
    """Rebuild EfficientNet-B0 with the same head as train_consensus_model.py
    (Dropout 0.2 -> Linear(1280, 2)) and load the checkpoint.
    """
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.classifier[1].in_features, 2),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def write_csv_in_chunks(output_path, headers, data_rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_path.exists()
    df_chunk = pd.DataFrame(data_rows, columns=headers)
    df_chunk.to_csv(output_path, mode="a", header=write_header, index=False)

    return len(data_rows)
