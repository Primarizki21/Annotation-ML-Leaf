#!/usr/bin/env python3
"""
predict_unlabeled_patches.py

Loads a trained EfficientNet-B0 checkpoint and predicts labels for all
patches in dataset_patches/ that are NOT part of the consensus set.

Output:
  predictions/master_predictions.csv   — per-patch predictions
  predictions/plots/                    — distribution visualizations
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0
from torchvision.transforms import v2
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams["figure.dpi"] = 500
plt.rcParams["savefig.bbox"] = "tight"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "efficientnet_b0_consensus" / "efficientnet_b0_consensus.pt"
DEFAULT_CONSENSUS_CSV = BASE_DIR / "consensus_review_master.csv"
DEFAULT_CONSENSUS_DIR = BASE_DIR / "dataset_consensus_only"
DEFAULT_PATCHES_DIR = BASE_DIR / "dataset_patches"
PREDICTIONS_DIR = BASE_DIR / "predictions"

CHUNK_SIZE = 50_000


def parse_args():
    parser = argparse.ArgumentParser(description="Predict labels for unlabeled patches")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--patches-dir", type=Path, default=DEFAULT_PATCHES_DIR)
    parser.add_argument("--consensus-csv", type=Path, default=DEFAULT_CONSENSUS_CSV)
    parser.add_argument("--consensus-dir", type=Path, default=DEFAULT_CONSENSUS_DIR)
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR / "master_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_consensus_set(consensus_csv, consensus_dir):
    df = pd.read_csv(consensus_csv)
    paths = set(df["patch_path"].tolist())
    return paths


def collect_patch_paths(patches_dir, consensus_set):
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


@torch.no_grad()
def run_inference(model, loader, device):
    all_preds = []
    all_confs = []

    for images, indices in tqdm(loader, desc="Inference", unit="batch"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

    return np.array(all_preds), np.array(all_confs)


def write_csv_in_chunks(output_path, headers, data_rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_path.exists()
    df_chunk = pd.DataFrame(data_rows, columns=headers)
    df_chunk.to_csv(output_path, mode="a", header=write_header, index=False)

    return len(data_rows)


def plot_confidence_distribution(df, output_dir, threshold):
    fig, ax = plt.subplots(figsize=(10, 5))

    for split_name, color in [("train", "#3498db"), ("test", "#e74c3c")]:
        subset = df[df["split"] == split_name]["confidence"]
        if len(subset) > 0:
            ax.hist(subset, bins=80, alpha=0.55, density=True,
                    label=f"{split_name} (n={len(subset):,})", color=color)

    ax.axvline(threshold, color="gray", linestyle="--",
               linewidth=1.2, label=f"threshold ({threshold})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "confidence_distribution.png")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'confidence_distribution.png'}")


def plot_per_class_label_distribution(df, output_dir):
    pivot = df.groupby(["class_name", "predicted_label"]).size().unstack(fill_value=0)
    pivot.columns = ["healthy", "unhealthy"]
    pivot = pivot.sort_values("unhealthy", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.32)))
    y_pos = range(len(pivot))
    ax.barh(y_pos, pivot["healthy"], label="predicted healthy", color="#3498db", alpha=0.85)
    ax.barh(y_pos, pivot["unhealthy"], left=pivot["healthy"],
            label="predicted unhealthy", color="#e74c3c", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Patch count")
    ax.set_title("Predicted Label Distribution per Class")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "per_class_label_distribution.png")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'per_class_label_distribution.png'}")


def plot_needs_review_summary(df, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    counts = df["needs_review"].value_counts()
    labels = ["auto-accepted", "needs review"]
    colors = ["#2ecc71", "#e74c3c"]
    values = [counts.get(False, 0), counts.get(True, 0)]
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90, textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax1.set_title("Overall Needs Review", fontsize=10)

    review_counts = (
        df[df["needs_review"]]
        .groupby("class_name")
        .size()
        .sort_values(ascending=True)
    )
    if len(review_counts) > 0:
        ax2.barh(range(len(review_counts)), review_counts.values,
                 color="#e74c3c", alpha=0.8)
        ax2.set_yticks(range(len(review_counts)))
        ax2.set_yticklabels(review_counts.index, fontsize=7)
        ax2.invert_yaxis()
        ax2.set_xlabel("Patches needing review")
        ax2.set_title("Needs Review per Class", fontsize=10)
        ax2.grid(True, axis="x", alpha=0.3)

        for bar, val in zip(ax2.containers[0], review_counts.values):
            ax2.text(val + max(review_counts.values) * 0.005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=5)

    fig.tight_layout()
    fig.savefig(output_dir / "needs_review_summary.png")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'needs_review_summary.png'}")


def print_summary(df):
    total = len(df)
    n_review = df["needs_review"].sum()
    n_auto = total - n_review
    label_dist = df["predicted_label"].value_counts()

    print(f"\n{'=' * 55}")
    print(f"  PREDICTION SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Total patches predicted:  {total:,}")
    print(f"  Predicted healthy:        {label_dist.get(0, 0):,} ({label_dist.get(0, 0) / total * 100:.1f}%)")
    print(f"  Predicted unhealthy:      {label_dist.get(1, 0):,} ({label_dist.get(1, 0) / total * 100:.1f}%)")
    print(f"  Auto-accepted:            {n_auto:,} ({n_auto / total * 100:.1f}%)")
    print(f"  Needs review:             {n_review:,} ({n_review / total * 100:.1f}%)")

    print(f"\n  By split:")
    for split in ["train", "test"]:
        for subdir_name in ["healthy", "needs_annotation"]:
            key_label = "auto_healthy" if subdir_name == "healthy" else "unlabeled"
            subset = df[(df["split"] == split) & (df["agreement_level"] == key_label)]
            if len(subset) > 0:
                n_r = subset["needs_review"].sum()
                print(f"    {split}/{subdir_name}/: {len(subset):,} predicted, "
                      f"{n_r:,} needs review ({n_r / len(subset) * 100:.1f}%)")

    review_class = (
        df[df["needs_review"]]
        .groupby("class_name")
        .size()
        .sort_values(ascending=False)
    )
    if len(review_class) > 0:
        print(f"\n  Top 5 classes needing most review:")
        for i, (cls, cnt) in enumerate(review_class.head(5).items(), 1):
            print(f"    {i}. {cls:<50} {cnt:,} patches")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.output.suffix != ".csv":
        args.output = args.output / "master_predictions.csv"

    if args.output.exists():
        args.output.unlink()

    print("Building consensus set...")
    consensus_set = build_consensus_set(args.consensus_csv, args.consensus_dir)
    print(f"  Consensus patches: {len(consensus_set):,}")

    print("Scanning dataset_patches directory...")
    train_healthy, train_unlabeled, test_healthy, test_unlabeled = \
        collect_patch_paths(args.patches_dir, consensus_set)

    label_map = {
        "train_healthy": train_healthy,
        "train_unlabeled": train_unlabeled,
        "test_healthy": test_healthy,
        "test_unlabeled": test_unlabeled,
    }
    all_entries = train_healthy + train_unlabeled + test_healthy + test_unlabeled
    total_patches = len(all_entries)
    print(f"  Total patches to predict: {total_patches:,}")
    for key, entries in label_map.items():
        print(f"    {key}: {len(entries):,}")

    if total_patches == 0:
        print("No patches to predict. Exiting.")
        return

    print("\nLoading model...")
    model = load_model(args.model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {args.model.name} ({n_params / 1e6:.1f}M params)")

    transform = build_transform()
    ds = InferenceDataset(all_entries, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pred_labels, confidences = run_inference(model, loader, device)

    csv_headers = [
        "patch_path", "class_name", "split", "agreement_level",
        "is_consensus", "predicted_label", "confidence", "needs_review",
    ]

    written = 0
    buffer = []

    for i, entry in enumerate(tqdm(all_entries, desc="Writing CSV", unit="row")):
        is_review = confidences[i] < args.confidence_threshold
        row = {
            "patch_path": entry["rel_path"],
            "class_name": entry["class_name"],
            "split": entry["split"],
            "agreement_level": entry["agreement_level"],
            "is_consensus": entry["is_consensus"],
            "predicted_label": int(pred_labels[i]),
            "confidence": round(float(confidences[i]), 6),
            "needs_review": is_review,
        }
        buffer.append(row)

        if len(buffer) >= CHUNK_SIZE:
            written += write_csv_in_chunks(args.output, csv_headers, buffer)
            buffer.clear()

    if buffer:
        written += write_csv_in_chunks(args.output, csv_headers, buffer)
        buffer.clear()

    print(f"\nPredictions saved: {args.output} ({written:,} rows)")

    print("\nGenerating plots...")
    df = pd.read_csv(args.output)
    plot_dir = args.output.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_confidence_distribution(df, plot_dir, args.confidence_threshold)
    plot_per_class_label_distribution(df, plot_dir)
    plot_needs_review_summary(df, plot_dir)

    print_summary(df)

    print(f"\nDone. All outputs in: {args.output.parent}")


if __name__ == "__main__":
    main()
