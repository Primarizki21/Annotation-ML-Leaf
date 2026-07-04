#!/usr/bin/env python3
"""build_dataset_binary.py

Build dataset_binary.csv from master_predictions_round4.csv + consensus_round1_master.csv.
Binary classification (0=healthy, 1=unhealthy). Train/test split follows dataset_filtered/splits/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR    = Path(__file__).resolve().parent
MASTER_CSV  = BASE_DIR / "predictions" / "master_predictions_round4.csv"
CONSENSUS_CSV = BASE_DIR / "consensus_round1_master.csv"
SPLITS_DIR  = BASE_DIR / "dataset_filtered" / "splits"
PATCHES_DIR = BASE_DIR / "dataset_patches"

OUT_CSV         = BASE_DIR / "dataset_binary.csv"
OUT_CLASS_INDEX = BASE_DIR / "class_index.json"
OUT_STATS       = BASE_DIR / "stats.json"

CONSENSUS_LEVELS = {"4/5", "5/5"}
OUT_COLS = ["patch_path", "split", "class_name", "label", "source",
            "agreement_level", "confidence", "margin", "needs_review"]


def load_leaf_splits() -> tuple[set[str], set[str]]:
    train = {ln.strip() for ln in (SPLITS_DIR / "train.txt").read_text().splitlines() if ln.strip()}
    test  = {ln.strip() for ln in (SPLITS_DIR / "test.txt").read_text().splitlines() if ln.strip()}
    overlap = train & test
    if overlap:
        raise SystemExit(f"FATAL: train/test leaf overlap = {len(overlap)}")
    return train, test


def load_overrides() -> tuple[pd.Series, pd.Series, int]:
    """Return (label Series, source Series, count of 4/5+5/5 rows)."""
    c = pd.read_csv(CONSENSUS_CSV)
    c = c[c["agreement_level"].isin(CONSENSUS_LEVELS)].copy()
    c["label"]  = c["suggested_numeric_label"].astype(int)
    c["source"] = "consensus_" + c["agreement_level"].str.replace("/", "of")
    n = len(c)
    return (c.set_index("patch_path")["label"],
            c.set_index("patch_path")["source"],
            n)


def main() -> int:
    print(f"Loading leaf splits from {SPLITS_DIR}...")
    train_leaves, test_leaves = load_leaf_splits()
    print(f"  train leaves: {len(train_leaves):,}  test leaves: {len(test_leaves):,}")

    print(f"Loading {MASTER_CSV.name}...")
    mp = pd.read_csv(MASTER_CSV)
    print(f"  rows: {len(mp):,}")

    print(f"Loading consensus overrides from {CONSENSUS_CSV.name}...")
    override_label, override_source, n_overrides = load_overrides()
    print(f"  4/5 + 5/5 rows: {n_overrides:,}")

    mapped_label  = mp["patch_path"].map(override_label)
    mapped_source = mp["patch_path"].map(override_source)
    override_mask = mapped_label.notna().to_numpy()

    auto_healthy_mask = mp["class_name"].str.endswith("___healthy").to_numpy()
    predicted_int = pd.to_numeric(mp["predicted_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    default_label  = np.where(auto_healthy_mask, 0, predicted_int)
    default_source = np.where(auto_healthy_mask, "auto_healthy", "unlabeled")
    default_agree  = np.where(auto_healthy_mask, "auto_healthy", "unlabeled")

    final_label  = np.where(override_mask, mapped_label.fillna(0).astype(int).to_numpy(), default_label)
    final_source = np.where(override_mask, mapped_source.fillna("").to_numpy(), default_source)
    final_agree  = np.where(
        override_mask,
        np.where(final_source == "consensus_5of5", "5/5", "4/5"),
        default_agree,
    )
    final_conf   = np.where(override_mask, np.nan, mp["confidence"].to_numpy(dtype=float))
    final_margin = np.where(override_mask, np.nan, mp["margin"].to_numpy(dtype=float))
    needs_review_str = mp["needs_review"].astype(str).str.strip()
    needs_review_bool = needs_review_str.isin(["True", "true", "1"]).to_numpy()
    final_nr = np.where(override_mask, False, needs_review_bool)

    out = pd.DataFrame({
        "patch_path":      mp["patch_path"].to_numpy(),
        "split":           mp["split"].to_numpy(),
        "class_name":      mp["class_name"].to_numpy(),
        "label":           final_label.astype(int),
        "source":          final_source,
        "agreement_level": final_agree,
        "confidence":      final_conf,
        "margin":          final_margin,
        "needs_review":    final_nr.astype(bool),
    })

    n_applied = int(override_mask.sum())
    if n_applied != n_overrides:
        print(f"WARN: {n_applied} of {n_overrides} consensus overrides applied "
              f"({n_overrides - n_applied} patch_paths not found in master_predictions)")

    print("Verifying patch paths on disk (1000-row sample)...")
    sample = out["patch_path"].sample(min(1000, len(out)), random_state=42)
    missing = [p for p in sample if not (PATCHES_DIR / p).exists()]
    if missing:
        print(f"WARN: {len(missing)} sampled patches missing on disk, e.g. {missing[0]}")

    class_names = sorted(out["class_name"].unique().tolist())
    class_index = {str(i): n for i, n in enumerate(class_names)}
    OUT_CLASS_INDEX.write_text(json.dumps(class_index, indent=2))
    print(f"Wrote {OUT_CLASS_INDEX.name}")

    def flat(d: dict) -> dict:
        return {f"{k[0]}_{k[1]}": int(v) for k, v in d.items()}

    stats = {
        "total_rows": int(len(out)),
        "split":  out["split"].value_counts().to_dict(),
        "label":  out["label"].value_counts().to_dict(),
        "source": out["source"].value_counts().to_dict(),
        "label_per_split":  flat(out.groupby(["split", "label"]).size().to_dict()),
        "source_per_split": flat(out.groupby(["split", "source"]).size().to_dict()),
        "class_count": len(class_names),
        "sanity": {
            "train_test_leaf_overlap": 0,
            "label_values": sorted(out["label"].unique().tolist()),
            "consensus_override_count": n_applied,
            "consensus_override_expected": n_overrides,
        },
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {OUT_STATS.name}")

    print(f"Writing {OUT_CSV.name}...")
    out[OUT_COLS].to_csv(OUT_CSV, index=False)

    n = len(out)
    print("\n=== Summary ===")
    print(f"Total rows:    {n:,}")
    print(f"Train / Test:  {(out.split=='train').sum():,} / {(out.split=='test').sum():,}")
    print(f"Healthy (0):   {(out.label==0).sum():,}  ({(out.label==0).mean()*100:.1f}%)")
    print(f"Unhealthy (1): {(out.label==1).sum():,}  ({(out.label==1).mean()*100:.1f}%)")
    print(f"Sources:       {out.source.value_counts().to_dict()}")
    print(f"Classes:       {len(class_names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
