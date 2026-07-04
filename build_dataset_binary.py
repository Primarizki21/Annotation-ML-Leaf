#!/usr/bin/env python3
"""build_dataset_binary.py

Build output_dataset_final/dataset_binary.csv from
predictions/master_predictions_round4.csv (model pseudo-labels) +
consensus_round1_master.csv (4/5 + 5/5 human labels).
Binary classification: 0 = healthy, 1 = unhealthy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR       = Path(__file__).resolve().parent
MASTER_CSV     = BASE_DIR / "predictions" / "master_predictions_round4.csv"
CONSENSUS_CSV  = BASE_DIR / "consensus_round1_master.csv"
PATCHES_DIR    = BASE_DIR / "dataset_patches"
OUTPUT_DIR     = BASE_DIR / "output_dataset_final"

CONSENSUS_LEVELS = {"4/5", "5/5"}
OUT_COLS = ["patch_path", "split", "class_name", "label", "source",
            "agreement_level", "confidence", "margin", "needs_review"]


def main() -> int:
    print(f"Loading {MASTER_CSV.name}...")
    mp = pd.read_csv(MASTER_CSV)
    print(f"  rows: {len(mp):,}")

    print(f"Loading {CONSENSUS_CSV.name} (4/5 + 5/5 only)...")
    c = pd.read_csv(CONSENSUS_CSV)
    c = c[c["agreement_level"].isin(CONSENSUS_LEVELS)].copy()
    c["source"] = "consensus_" + c["agreement_level"].str.replace("/", "of")
    c["split"]  = c["patch_path"].str.split("/").str[0]
    print(f"  rows: {len(c):,}")

    auto_healthy = mp["class_name"].str.endswith("___healthy").to_numpy()
    predicted    = pd.to_numeric(mp["predicted_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    needs_review = mp["needs_review"].astype(str).str.strip().isin(["True", "true", "1"]).to_numpy()

    master_df = pd.DataFrame({
        "patch_path":      mp["patch_path"].to_numpy(),
        "split":           mp["split"].to_numpy(),
        "class_name":      mp["class_name"].to_numpy(),
        "label":           np.where(auto_healthy, 0, predicted).astype(int),
        "source":          np.where(auto_healthy, "auto_healthy", "unlabeled"),
        "agreement_level": np.where(auto_healthy, "auto_healthy", "unlabeled"),
        "confidence":      mp["confidence"].to_numpy(dtype=float),
        "margin":          mp["margin"].to_numpy(dtype=float),
        "needs_review":    needs_review,
    })

    n_c = len(c)
    consensus_df = pd.DataFrame({
        "patch_path":      c["patch_path"].to_numpy(),
        "split":           c["split"].to_numpy(),
        "class_name":      c["class_name"].to_numpy(),
        "label":           c["suggested_numeric_label"].astype(int).to_numpy(),
        "source":          c["source"].to_numpy(),
        "agreement_level": c["agreement_level"].to_numpy(),
        "confidence":      np.full(n_c, np.nan),
        "margin":          np.full(n_c, np.nan),
        "needs_review":    np.zeros(n_c, dtype=bool),
    })

    out = pd.concat([master_df, consensus_df], ignore_index=True)
    assert len(out) == len(master_df) + len(consensus_df), "concat row count mismatch"

    print("Verifying consensus patches on disk (100%)...")
    missing_c = [p for p in consensus_df["patch_path"] if not (PATCHES_DIR / p).exists()]
    if missing_c:
        print(f"WARN: {len(missing_c)} consensus patches missing on disk, e.g. {missing_c[0]}")

    print("Verifying master patch sample on disk (1000-row sample)...")
    sample = master_df["patch_path"].sample(min(1000, len(master_df)), random_state=42)
    missing_m = [p for p in sample if not (PATCHES_DIR / p).exists()]
    if missing_m:
        print(f"WARN: {len(missing_m)} sampled master patches missing on disk, e.g. {missing_m[0]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv   = OUTPUT_DIR / "dataset_binary.csv"
    out_idx   = OUTPUT_DIR / "class_index.json"
    out_stats = OUTPUT_DIR / "stats.json"

    class_names = sorted(out["class_name"].unique().tolist())
    out_idx.write_text(json.dumps({str(i): n for i, n in enumerate(class_names)}, indent=2))
    print(f"Wrote {out_idx.name}")

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
            "label_values": sorted(out["label"].unique().tolist()),
            "master_count": int(len(master_df)),
            "consensus_count": int(len(consensus_df)),
            "consensus_missing_on_disk": len(missing_c),
        },
    }
    out_stats.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_stats.name}")

    print(f"Writing {out_csv.name}...")
    out[OUT_COLS].to_csv(out_csv, index=False)

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
