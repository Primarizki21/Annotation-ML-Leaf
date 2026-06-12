#!/usr/bin/env python3
"""
train_consensus_model.py

Trains EfficientNet-B0 on 5/5 + 4/5 consensus patches for binary
classification (healthy/unhealthy). Saves the best checkpoint,
training metrics, and evaluation plots to models/.
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONSENSUS_CSV = BASE_DIR / "consensus_review_master.csv"
DEFAULT_PATCHES_DIR = BASE_DIR / "dataset_consensus_only"
MODELS_DIR = BASE_DIR / "models"

CLASS_NAMES_BINARY = ["healthy", "unhealthy"]


class ConsensusDataset(Dataset):
    def __init__(self, df, patches_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.patches_dir = patches_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.patches_dir / row["patch_path"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["suggested_numeric_label"])

        if self.transform:
            image = self.transform(image)

        return image, label


class EvalDataset(Dataset):
    def __init__(self, df, patches_dir, transform):
        self.df = df.reset_index(drop=True)
        self.patches_dir = patches_dir
        self.transform = transform
        self.class_names = sorted(df["class_name"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.patches_dir / row["patch_path"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["suggested_numeric_label"])
        if self.transform:
            image = self.transform(image)
        return image, label, self.class_to_idx[row["class_name"]]


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on consensus data")
    parser.add_argument("--consensus-csv", type=Path, default=DEFAULT_CONSENSUS_CSV)
    parser.add_argument("--patches-dir", type=Path, default=DEFAULT_PATCHES_DIR)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "efficientnet_b0_consensus" / "efficientnet_b0_consensus.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_transforms(train=True):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    base = [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]

    if train:
        augment = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        return v2.Compose(augment + base)

    return v2.Compose(base)


def compute_class_weights(labels):
    classes, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    weights = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    class_weight_tensor = torch.tensor(
        [weights.get(0, 1.0), weights.get(1, 1.0)], dtype=torch.float32
    )
    return class_weight_tensor


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"loss": avg_loss, "acc": acc, "f1_macro": f1_macro}


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=[0, 1], zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1_per_class, f1_macro


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_acc,
        },
        path,
    )


def plot_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="train", markersize=3)
    axes[0].plot(epochs, history["val_loss"], "s-", label="val", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", label="train", markersize=3)
    axes[1].plot(epochs, history["val_acc"], "s-", label="val", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1-macro
    axes[2].plot(epochs, history["train_f1_macro"], "o-", label="train", markersize=3)
    axes[2].plot(epochs, history["val_f1_macro"], "s-", label="val", markersize=3)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1-macro")
    axes[2].set_title("F1-macro")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'training_curves.png'}")


def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix (val set)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_per_class_f1(per_class, output_path):
    valid = {k: v for k, v in per_class.items() if v is not None}
    if not valid:
        print("  Skipping per-class F1 plot: all classes have single label in val set")
        return

    names = sorted(valid, key=valid.get, reverse=True)
    scores = [valid[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.3)))
    bars = ax.barh(range(len(names)), scores, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("F1 (unhealthy = positive class)")
    ax.set_title("Per-Class F1 on Validation Set")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


@torch.no_grad()
def final_evaluation(model, val_df, args):
    model.eval()
    eval_ds = EvalDataset(val_df, args.patches_dir, build_transforms(train=False))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size * 2,
                             shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    all_class_ids = []

    for images, labels, class_ids in tqdm(eval_loader, desc="Final eval", leave=False):
        images = images.to(next(model.parameters()).device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_class_ids.extend(class_ids.cpu().numpy())

    idx_to_class = {v: k for k, v in eval_ds.class_to_idx.items()}

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    class_ids_arr = np.array(all_class_ids)

    per_class_f1 = {}
    for ci in sorted(set(all_class_ids)):
        mask = class_ids_arr == ci
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(np.unique(yt)) < 2:
            per_class_f1[idx_to_class[ci]] = None
        else:
            per_class_f1[idx_to_class[ci]] = float(
                f1_score(yt, yp, pos_label=1, zero_division=0)
            )

    return y_true.tolist(), y_pred.tolist(), per_class_f1


def main():
    args = parse_args()

    if args.output.suffix != ".pt":
        args.output = args.output / "model.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    df = pd.read_csv(args.consensus_csv)
    df = df[df["agreement_level"].isin(["4/5", "5/5"])].copy()
    print(f"Consensus patches (5/5 + 4/5): {len(df):,}")

    label_dist = df["suggested_numeric_label"].value_counts().to_dict()
    print(f"Label distribution: healthy={label_dist.get(0, 0):,} unhealthy={label_dist.get(1, 0):,}")

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_split,
        stratify=df["class_name"],
        random_state=args.seed,
    )
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}")

    train_ds = ConsensusDataset(train_df, args.patches_dir, transform=build_transforms(train=True))
    val_ds = ConsensusDataset(val_df, args.patches_dir, transform=build_transforms(train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.classifier[1].in_features, 2),
    )
    model = model.to(device)

    class_weights = compute_class_weights(train_df["suggested_numeric_label"].values)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda")

    best_val_acc = 0.0
    patience_counter = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1_macro": [], "val_f1_macro": [],
        "val_f1_healthy": [], "val_f1_unhealthy": [],
    }

    print(f"\nTraining {args.epochs} epochs (patience={args.patience})")
    print(f"Classes: {model.classifier[1].out_features}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, f1_per_class, f1_macro = validate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(f1_macro)
        history["val_f1_healthy"].append(f1_per_class[0])
        history["val_f1_unhealthy"].append(f1_per_class[1])

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"trn_loss: {train_metrics['loss']:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"trn_acc: {train_metrics['acc']:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"trn_f1: {train_metrics['f1_macro']:.4f} | "
            f"val_f1: {f1_macro:.4f} "
            f"(hth: {f1_per_class[0]:.4f}, unh: {f1_per_class[1]:.4f})"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_acc, args.output)
            print(f"  -> Best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(history, output_dir)

    print(f"\nRunning final evaluation on best checkpoint...")
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred, per_class_f1 = final_evaluation(model, val_df, args)

    save_confusion_matrix(y_true, y_pred, CLASS_NAMES_BINARY, output_dir / "confusion_matrix.png")

    valid_f1 = {k: round(v, 4) for k, v in per_class_f1.items() if v is not None}
    skip_f1 = [k for k, v in per_class_f1.items() if v is None]
    if valid_f1:
        worst_class = min(valid_f1, key=valid_f1.get)
        best_class = max(valid_f1, key=valid_f1.get)
        print(f"\nPer-class F1 (unhealthy):")
        print(f"  Best:  {best_class:<50} {valid_f1[best_class]:.4f}")
        print(f"  Worst: {worst_class:<50} {valid_f1[worst_class]:.4f}")
        avg_f1 = float(np.mean(list(valid_f1.values())))
        print(f"  Average across {len(valid_f1)} classes: {avg_f1:.4f}")
        if skip_f1:
            print(f"  Skipped ({len(skip_f1)} classes with single label): {skip_f1}")

    save_per_class_f1(per_class_f1, output_dir / "per_class_f1.png")

    best_epoch = checkpoint["epoch"]
    metrics_path = args.output.with_suffix(".json")
    metrics = {
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": best_epoch,
        "total_epochs_trained": epoch,
        "class_weights": {str(k): float(v) for k, v in enumerate(class_weights.cpu())},
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "per_class_f1": valid_f1,
        "avg_per_class_f1": avg_f1 if valid_f1 else None,
        "history": {k: [float(x) for x in v] for k, v in history.items()},
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model:  {args.output}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
