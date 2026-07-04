#!/usr/bin/env python3
"""
active_learning_round.py

Active learning loop for Plant Disease Detection (PlantVillage, EfficientNet-B0).

============================================================================
USAGE — Full Round 2 from scratch
============================================================================
# Prereq: Phase 1 done (predictions/master_predictions_round2.csv exists).
#         Rename existing predictions/master_predictions.csv to add the
#         _round2 suffix if needed.
# Prereq: Round 1 model checkpoint at models/efficientnet_b0_consensus/ (or
#         models/round1/model.pt -- both are the same model).

# 1. Generate Phase 2 verification pool (overlap, all 5 annotators see all)
uv run active_learning_round.py --phase 2 --subcommand generate --round 2

# 2. Select HITL pool via KMeans on 1280-d embeddings.
#    Phase 3 takes no --subcommand (single operation).
#    This DIRECTLY UPDATES al_assignments_round2.json (adds label_hitl patches).
#    K is auto-derived from --budget via compute_k() (override with --k).
#    --margin-threshold 0.7 recommended for confident models (median margin > 0.95).
uv run active_learning_round.py --phase 3 --round 2 --budget 100 --margin-threshold 0.7

# 3. Annotators complete their session via the web app (mode "al")
#    -> writes annotations/annotations_{name}_al_round2.csv for each of 5 annotators

# 4. Verify votes, compute per-class accuracy, output pseudo + HITL sets
uv run active_learning_round.py --phase 2 --subcommand verify --round 2

# 5. Compose Round 2 dataset
uv run active_learning_round.py --phase 4 --subcommand compose --round 2

# 6. Train Round 2 model (wraps train_consensus_model.py)
uv run active_learning_round.py --phase 5 --subcommand train --round 2
#    -> saves to models/round2/model.pt
#    -> by default fine-tunes from models/round1/model.pt
#       (or legacy models/efficientnet_b0_consensus/efficientnet_b0_consensus.pt)

============================================================================
USAGE — Round 3 (and beyond) reusing the same code
============================================================================
# Bump --round to 3 — output paths auto-update to master_predictions_round3.csv
# No need to pass --predictions-csv or --output; defaults follow --round.
uv run active_learning_round.py --phase 1 --round 3 \\
    --model models/round2/model.pt

uv run active_learning_round.py --phase 2 --subcommand generate --round 3
uv run active_learning_round.py --phase 3 --round 3
# ... annotators do their thing ...
uv run active_learning_round.py --phase 2 --subcommand verify --round 3
uv run active_learning_round.py --phase 4 --subcommand compose --round 3
uv run active_learning_round.py --phase 5 --subcommand train --round 3
#    -> fine-tunes from models/round2/model.pt

============================================================================
Cross-round design: clean-slate training (no data accumulation)
============================================================================
Each round's training set is composed independently from three sources:
  - initial: consensus_round1_master.csv (the only consensus file; READ-ONLY,
    never modified across rounds)
  - pseudo:  predictions/pseudo_labeled_set_round{N}.csv (THIS round only)
  - hitl:    predictions/hitl_annotated_round{N}.csv (THIS round only)

Pseudo-labels and HITL annotations from previous rounds are INTENTIONALLY
NOT included in the next round's training set. Knowledge from previous
rounds is transferred via the model weights (init checkpoint =
models/round{N-1}/model.pt), not via accumulated data. This avoids
pseudo-label error compounding across rounds.

To inspect round-N training composition:
  pd.read_csv('round{N}_dataset.csv')['source'].value_counts()
Expected breakdown:
  initial:  ~30,357  (from consensus_round1)
  pseudo:   variable (per-class capped, from THIS round's verify)
  hitl:     variable (from THIS round's verify, 4/5+ agreement only)

============================================================================
USAGE — Common flags
============================================================================
  --phase N              1..5 (required)
  --subcommand STR       For --phase 2: "generate" or "verify" (required)
  --round N              Round number (default: 2)
  --model PATH           Model checkpoint (default: Round 1 consensus model)
  --predictions-csv PATH Input master predictions CSV (Phase 2/3/verify input,
                             output of Phase 1). Default follows --round.
  --patches-dir PATH     dataset_patches directory
  --consensus-csv PATH   consensus_review_master.csv
  --output PATH          Phase 1 output CSV. Default follows --round.
  --batch-size N         Inference batch size (default: 128)
  --confidence-threshold F  Default conf threshold (default: 0.9)
  --margin-threshold F   Margin cutoff for "uncertain" pool (default: 0.2)
  --device STR           cuda or cpu (default: cuda)
  --class-thresholds-json PATH  Optional per-class threshold overrides
  --k N                  K-Means k override. Default: auto from compute_k()
  --n-classes N          Number of output classes for K floor (default: 2)
  --min-cluster-size N   Min patches per cluster (default: 5)
  --budget N             HITL hard cap (Phase 3, default: 100)

Outputs per round N land in:
  predictions/al_assignments_round{N}.json
  predictions/embeddings_round{N}.npy
  predictions/pseudo_labeled_set_round{N}.csv
  predictions/hitl_annotated_round{N}.csv
  predictions/per_class_accuracy_round{N}.json
  predictions/cluster_representatives_round{N}.json
  annotations/annotations_{name}_al_round{N}.csv   (× 5 annotators)

============================================================================
Phases:
============================================================================
  1. Inference & Split        — run model on unlabeled disease patches only,
                                compute margin (P(top-1) - P(top-2)),
                                write master_predictions.csv with margin column.
  2. Pseudo-label Quality     — stratified 3-10 per class manual check,
                                per-class threshold bumps to 0.95/0.98.
                                Subcommands: generate, verify.
  3. Smart HITL Selection     — extract 1280-d embeddings, KMeans(k),
                                pick cluster representative with lowest margin.
  4. Dataset Composition      — (TBD) merge initial + pseudo + HITL, per-class cap.
  5. Training Round 2         — (TBD) fine-tune from round 1 checkpoint.

Phase 1 scope:
  - GPU inference ONLY on train/needs_annotation/**/<DiseaseClass>/*.jpg
    and test/needs_annotation/**/<DiseaseClass>/*.jpg (no GPU on healthy/).
  - Auto-healthy rows included in CSV with hardcoded label=0, conf=1.0, margin=1.0.
  - Consensus-set patches excluded (already labeled in consensus_review_master.csv).
  - Output: master_predictions.csv with margin, top2, effective_threshold columns.
  - Plots: 7 PNGs in predictions/plots_round{N}/ (3 carried over + 4 new AL-specific).

Phase 2 generate scope (Step 1 of active learning):
  - Reads: predictions/master_predictions.csv (from Phase 1)
  - Samples 3-10 verify_pseudo patches per class (stratified, high_conf only)
  - All 5 annotators see all verify_pseudo patches (overlap, no split)
  - Attaches model_prediction, model_confidence, model_margin to each verify_pseudo patch
  - If predictions/al_assignments_round{N}.json already exists, merges any
    existing label_hitl patches (added by a prior --phase 3 run)
  - Output: predictions/al_assignments_round{N}.json

Phase 3 select scope (uses compute_k() to derive K from budget):
  - Reads: predictions/master_predictions.csv (uncertain pool = margin < threshold)
  - Extracts 1280-d embeddings via forward hook on model.avgpool
  - K-Means k derived via compute_k(budget, pool, n_classes, min_cluster_size):
      Rule 1: ceiling = max(1, n_uncertain // min_cluster_size)
      Rule 2: k = min(budget, ceiling)
      Rule 3: k = max(k, min(n_classes, ceiling))
    Override: --k N (capped at pool size)
  - Picks argmin(margin) representative per cluster (skips empty clusters)
  - Caps at --budget (sort representatives by margin ascending, take top N)
  - Updates predictions/al_assignments_round{N}.json to add label_hitl patches
  - Saves embeddings_round{N}.npy + cluster_representatives_round{N}.json
"""

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference_utils import (
    BASE_DIR,
    CHUNK_SIZE,
    DEFAULT_CONSENSUS_CSV,
    DEFAULT_MODEL,
    DEFAULT_PATCHES_DIR,
    PREDICTIONS_DIR,
    InferenceDataset,
    build_consensus_set,
    build_transform,
    collect_patch_paths,
    load_model,
    write_csv_in_chunks,
)

ANNOTATIONS_DIR = BASE_DIR / "annotations"

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams["figure.dpi"] = 500
plt.rcParams["savefig.bbox"] = "tight"

DEFAULT_MARGIN_THRESHOLD = 0.2

# Active learning defaults (Phase 2 + 3)
AL_MIN_VERIFY_PER_CLASS = 3       # floor on verification pool size per class
AL_MAX_VERIFY_PER_CLASS = 10      # cap on verification pool size per class
AL_VERIFY_FRACTION = 0.05         # 5% of high_conf per class (clamped by min/max)
AL_DEFAULT_VERIFY_SEED = 42       # reproducibility for stratified sampling
AL_DEFAULT_ROUND = 2              # default round number
AL_DEFAULT_BUDGET = 100           # HITL hard cap (1:1 with K when K=compute_k(budget,...))
AL_CLUSTER_SANITY_THRESHOLD = 0.30  # warn if any cluster > 30% of pool
AL_SCATTER_MAX_POINTS = 50_000     # max points to render as scatter (downsampled)
AL_DEFAULT_N_CLASSES = 2          # K-Means K floor — binary task (healthy vs unhealthy)
AL_MIN_CLUSTER_SIZE = 5           # each cluster needs at least this many patches
AL_DEFAULT_ANNOTATORS = ["Cinta", "Diaz", "Muna", "Oki", "Sarah"]


def _resolve_default_model(round_n: int) -> Path:
    """Default model for Phase 1 (inference) and Phase 3 (embeddings) of round N.

      - Round 1 or 2: round 1 model (the consensus model that round 2 fine-tunes from)
      - Round N >= 3: round (N-1) model (the freshly-trained model from the prior round)

    Falls back to round 1 model if the expected checkpoint doesn't exist
    (e.g. running Phase 3 of round 3 before Phase 5 of round 3 finishes).
    """
    if round_n <= 2:
        candidate = BASE_DIR / "models" / "round1" / "model.pt"
    else:
        candidate = BASE_DIR / "models" / f"round{round_n - 1}" / "model.pt"
    return candidate if candidate.exists() else BASE_DIR / "models" / "round1" / "model.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Active learning round pipeline for Plant Disease Detection"
    )
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Which phase of the active learning loop to run")
    parser.add_argument("--subcommand", type=str,
                        choices=["generate", "verify", "compose", "train"],
                        help="Subcommand for --phase 2 ({generate, verify}), "
                             "--phase 4 (compose), or --phase 5 (train)")
    parser.add_argument("--round", type=int, default=AL_DEFAULT_ROUND,
                        help="Round number (default: 2)")
    parser.add_argument("--model", type=Path, default=None,
                        help="Model checkpoint (default: round N-1 model, "
                             "or round 1 model for round 1/2)")
    parser.add_argument("--patches-dir", type=Path, default=DEFAULT_PATCHES_DIR)
    parser.add_argument("--consensus-csv", type=Path, default=DEFAULT_CONSENSUS_CSV)
    parser.add_argument("--predictions-csv", type=Path, default=None,
                        help="WARNING: this flag is ONLY for Phase 2/3/verify "
                             "(input CSV). Phase 1 IGNORES it silently — use "
                             "--output to set the Phase 1 output path instead. "
                             "Default: predictions/master_predictions_round{N}.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Phase 1 output CSV. Default: "
                             "predictions/master_predictions_round{N}.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--confidence-threshold", type=float, default=0.9,
                        help="Default confidence threshold (global)")
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD,
                        help="Margin below which a patch is 'uncertain' (for plots)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--class-thresholds-json", type=Path, default=None,
                        help="Optional JSON: {class_name: per-class threshold} "
                             "overrides --confidence-threshold for that class")
    parser.add_argument("--k", type=int, default=None,
                        help="K-Means k override. Default: auto from compute_k()")
    parser.add_argument("--budget", type=int, default=AL_DEFAULT_BUDGET,
                        help="HITL hard cap per round (Phase 3, default: 100)")
    parser.add_argument("--n-classes", type=int, default=AL_DEFAULT_N_CLASSES,
                        help="Number of output classes for K floor (default: 2)")
    parser.add_argument("--min-cluster-size", type=int, default=AL_MIN_CLUSTER_SIZE,
                        help="Min patches per cluster (default: 5)")
    parser.add_argument("--annotators", nargs="+", default=AL_DEFAULT_ANNOTATORS,
                        help="Annotator names (default: Cinta Diaz Muna Oki Sarah)")
    parser.add_argument("--no-init-checkpoint", action="store_true",
                        help="Phase 5: start training from ImageNet instead of "
                             "the previous round's checkpoint")
    parser.add_argument("--train-epochs", type=int, default=None,
                        help="Phase 5: override training epochs (default: 15)")
    parser.add_argument("--train-lr", type=float, default=None,
                        help="Phase 5: override learning rate (default: 5e-5)")
    parser.add_argument("--train-batch-size", type=int, default=None,
                        help="Phase 5: override batch size (default: 64)")

    args = parser.parse_args()
    if args.phase == 2 and args.subcommand is None:
        parser.error("--phase 2 requires --subcommand {generate, verify}")
    if args.phase == 4 and args.subcommand is None:
        parser.error("--phase 4 requires --subcommand compose")
    if args.phase == 5 and args.subcommand is None:
        parser.error("--phase 5 requires --subcommand train")
    return args


def round_paths(round_n: int) -> dict:
    """Centralize all round-N file paths."""
    return {
        "master":      PREDICTIONS_DIR / f"master_predictions_round{round_n}.csv",
        "assignments": PREDICTIONS_DIR / f"al_assignments_round{round_n}.json",
        "embeddings":  PREDICTIONS_DIR / f"embeddings_round{round_n}.npy",
        "pseudo_set":  PREDICTIONS_DIR / f"pseudo_labeled_set_round{round_n}.csv",
        "hitl_set":    PREDICTIONS_DIR / f"hitl_annotated_round{round_n}.csv",
        "per_class":   PREDICTIONS_DIR / f"per_class_accuracy_round{round_n}.json",
        "cluster_map": PREDICTIONS_DIR / f"cluster_representatives_round{round_n}.json",
    }


def _default_master_csv(round_n: int) -> Path:
    """Default master predictions CSV for the given round."""
    return PREDICTIONS_DIR / f"master_predictions_round{round_n}.csv"


def _resolve_master_csv(args) -> Path:
    """Resolve the master predictions CSV from --predictions-csv or --round default."""
    if args.predictions_csv is not None:
        return args.predictions_csv
    return _default_master_csv(args.round)


# ============================================================================
# Inference (Phase 1-specific: returns 4 arrays, not 2)
# ============================================================================

@torch.no_grad()
def run_inference(model, loader, device):
    """Run EfficientNet-B0 on the unlabeled disease pool.

    Returns:
        preds        (np.array of int)   — argmax class (0=healthy, 1=unhealthy)
        confs        (np.array of float) — top-1 softmax probability
        margins      (np.array of float) — P(top-1) - P(top-2)
        top2_confs   (np.array of float) — top-2 softmax probability
    """
    all_preds, all_confs, all_margins, all_top2_confs = [], [], [], []

    for images, indices in tqdm(loader, desc="Inference", unit="batch"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        top2_confs, top2_labels = probs.topk(2, dim=1)
        all_preds.extend(top2_labels[:, 0].cpu().numpy())
        all_confs.extend(top2_confs[:, 0].cpu().numpy())
        all_margins.extend((top2_confs[:, 0] - top2_confs[:, 1]).cpu().numpy())
        all_top2_confs.extend(top2_confs[:, 1].cpu().numpy())

    return (np.array(all_preds), np.array(all_confs),
            np.array(all_margins), np.array(all_top2_confs))


# ============================================================================
# Phase 1 — Inference & Split
# ============================================================================


def compute_k(budget: int,
              n_uncertain: int,
              n_classes: int = AL_DEFAULT_N_CLASSES,
              min_cluster_size: int = AL_MIN_CLUSTER_SIZE) -> int:
    """Derive K-Means k from budget, pool size, and class count.

    Rules (applied in strict priority order):
      1. Hard ceiling: K <= n_uncertain // min_cluster_size
         Guarantees every cluster has >= min_cluster_size patches.
         This rule is NEVER overridden.
      2. Budget: K starts from budget (1 cluster -> 1 HITL pick).
         Capped by Rule 1.
      3. Floor at min(n_classes, ceiling):
         Encourages class coverage, but never violates Rule 1.
    """
    if n_uncertain == 0:
        return 0

    k_ceiling = max(1, n_uncertain // min_cluster_size)  # Rule 1 -- hard, never violated
    k = min(budget, k_ceiling)                            # Rule 2 -- budget, capped
    k = max(k, min(n_classes, k_ceiling))                 # Rule 3 -- floor, respects ceiling
    return k


def load_class_thresholds(path):
    """Load per-class threshold overrides from a JSON file.

    Returns {} if the path is None or the file does not exist.
    """
    if path is None:
        return {}
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def phase1_main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Default output: round-aware path
    if args.output is None:
        args.output = _default_master_csv(args.round)

    if args.output.suffix != ".csv":
        args.output = args.output / f"master_predictions_round{args.round}.csv"

    print(f"  Round:                  {args.round}")
    print(f"  Model:                  {args.model}")
    print(f"  Output:                 {args.output}")
    print(f"  Consensus CSV:          {args.consensus_csv}")
    print(f"  Patches dir:            {args.patches_dir}")

    if args.output.exists():
        args.output.unlink()

    print("Building consensus set...")
    consensus_set = build_consensus_set(args.consensus_csv)
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
    predict_entries = train_unlabeled + test_unlabeled
    total_patches = len(all_entries)
    n_to_predict = len(predict_entries)
    print(f"  Total patches collected: {total_patches:,}")
    print(f"  Patches to predict (unlabeled disease, GPU): {n_to_predict:,}")
    print(f"  Patches skipped (auto_healthy, no GPU):      "
          f"{len(train_healthy) + len(test_healthy):,}")
    for key, entries in label_map.items():
        print(f"    {key}: {len(entries):,}")

    if n_to_predict == 0:
        print("No unlabeled disease patches to predict. Exiting.")
        return

    print("\nLoading model...")
    model = load_model(args.model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {args.model.name} ({n_params / 1e6:.1f}M params)")

    transform = build_transform()
    ds = InferenceDataset(predict_entries, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    pred_labels, confidences, margins, top2_confs = run_inference(model, loader, device)

    # Reinsert auto-healthy placeholders so the full CSV has one row per patch.
    pred_labels = (
        [0] * len(train_healthy)
        + pred_labels[:len(train_unlabeled)].tolist()
        + [0] * len(test_healthy)
        + pred_labels[len(train_unlabeled):].tolist()
    )
    confidences = (
        [1.0] * len(train_healthy)
        + confidences[:len(train_unlabeled)].tolist()
        + [1.0] * len(test_healthy)
        + confidences[len(train_unlabeled):].tolist()
    )
    margins = (
        [1.0] * len(train_healthy)
        + margins[:len(train_unlabeled)].tolist()
        + [1.0] * len(test_healthy)
        + margins[len(train_unlabeled):].tolist()
    )
    top2_confs = (
        [0.0] * len(train_healthy)
        + top2_confs[:len(train_unlabeled)].tolist()
        + [0.0] * len(test_healthy)
        + top2_confs[len(train_unlabeled):].tolist()
    )

    class_thresholds = load_class_thresholds(args.class_thresholds_json)
    default_threshold = args.confidence_threshold

    csv_headers = [
        "patch_path", "class_name", "split", "agreement_level",
        "is_consensus", "predicted_label", "confidence",
        "top2_label", "top2_confidence", "margin",
        "needs_review", "effective_threshold",
    ]

    written = 0
    buffer = []

    for i, entry in enumerate(tqdm(all_entries, desc="Writing CSV", unit="row")):
        effective_thr = class_thresholds.get(entry["class_name"], default_threshold)
        is_review = confidences[i] < effective_thr
        row = {
            "patch_path": entry["rel_path"],
            "class_name": entry["class_name"],
            "split": entry["split"],
            "agreement_level": entry["agreement_level"],
            "is_consensus": entry["is_consensus"],
            "predicted_label": int(pred_labels[i]),
            "confidence": round(float(confidences[i]), 6),
            "top2_label": int(1 - pred_labels[i]),
            "top2_confidence": round(float(top2_confs[i]), 6),
            "margin": round(float(margins[i]), 6),
            "needs_review": bool(is_review),
            "effective_threshold": float(effective_thr),
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
    plot_dir = args.output.parent / f"plots_round{args.round}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_confidence_distribution(df, plot_dir, default_threshold)
    plot_per_class_label_distribution(df, plot_dir)
    plot_needs_review_summary(df, plot_dir)
    plot_margin_distribution(df, plot_dir)
    plot_confidence_vs_margin(df, plot_dir)
    plot_top1_vs_top2(df, plot_dir)
    plot_uncertain_per_class(df, plot_dir, args.margin_threshold)

    print_phase1_summary(df, default_threshold, args.margin_threshold)

    print(f"  Round:                  {args.round}")
    print(f"\n  Next step:")
    print(f"    uv run active_learning_round.py --phase 2 --subcommand generate "
          f"--round {args.round}")

    print(f"\nDone. All outputs in: {args.output.parent}")


# ============================================================================
# Phase 2 — generate: build al_assignments_round{N}.json
# ============================================================================

def phase2_generate(args):
    """Phase 2, subcommand: generate.

    Reads predictions/master_predictions.csv (from Phase 1), samples a
    verify_pseudo pool (3-10 per class, stratified, high_conf only),
    merges any existing label_hitl patches from a prior --phase 3 select
    run, and writes predictions/al_assignments_round{N}.json.

    All annotators get all patches (overlap, no split).
    """
    round_n = args.round
    paths = round_paths(round_n)
    master_path = _resolve_master_csv(args)
    assignments_path = paths["assignments"]

    if not master_path.exists():
        raise FileNotFoundError(
            f"Master predictions CSV not found: {master_path}\n"
            f"Run --phase 1 first to generate it, or pass "
            f"--predictions-csv to point at a different file."
        )

    print(f"Loading master predictions: {master_path}")
    df = pd.read_csv(master_path)
    print(f"  Round:                  {round_n}")
    print(f"  Total rows:             {len(df):,}")
    print(f"  Annotators:             {len(args.annotators)} ({args.annotators})")

    # ----- 1. Verification pool: high_conf (confidence > 0.9) by class -----
    high_conf = df[
        (df["agreement_level"] == "unlabeled") &
        (df["confidence"] > args.confidence_threshold)
    ].copy()
    print(f"\n  High-confidence pool "
          f"(unlabeled, conf > {args.confidence_threshold}): {len(high_conf):,}")

    n_classes = high_conf["class_name"].nunique()
    verify_patches = []
    per_class_sampled = []
    skipped_classes = []

    for class_name, group in high_conf.groupby("class_name"):
        if len(group) < AL_MIN_VERIFY_PER_CLASS:
            skipped_classes.append((class_name, len(group)))
            continue
        target = min(
            AL_MAX_VERIFY_PER_CLASS,
            max(AL_MIN_VERIFY_PER_CLASS, int(len(group) * AL_VERIFY_FRACTION)),
        )
        target = min(target, len(group))
        sampled = group.sample(n=target, random_state=AL_DEFAULT_VERIFY_SEED)
        for _, row in sampled.iterrows():
            verify_patches.append({
                "patch_path": row["patch_path"],
                "class_name": row["class_name"],
                "split": row["split"],
                "task_type": "verify_pseudo",
                "model_prediction": "unhealthy" if int(row["predicted_label"]) == 1
                                    else "healthy",
                "model_confidence": round(float(row["confidence"]), 6),
                "model_margin": round(float(row["margin"]), 6),
            })
        per_class_sampled.append((class_name, target))

    n_verify = len(verify_patches)
    print(f"  Sampled verify_pseudo:  {n_verify:,} "
          f"(across {len(per_class_sampled)} classes)")
    avg_per_class = n_verify / max(1, len(per_class_sampled))
    print(f"  Avg per class:          {avg_per_class:.1f} "
          f"(min={AL_MIN_VERIFY_PER_CLASS}, max={AL_MAX_VERIFY_PER_CLASS})")
    if skipped_classes:
        print(f"  Skipped classes:        {len(skipped_classes)} "
          f"(had < {AL_MIN_VERIFY_PER_CLASS} high_conf patches)")
        for cls, n in skipped_classes[:5]:
            print(f"    - {cls} (had {n})")
        if len(skipped_classes) > 5:
            print(f"    ... and {len(skipped_classes) - 5} more")

    # ----- 2. Note the uncertain pool (final selection is --phase 3) -----
    uncertain = df[
        (df["agreement_level"] == "unlabeled") &
        (df["margin"] < args.margin_threshold)
    ].copy()
    print(f"  Uncertain pool: {len(uncertain):,} patches "
          f"(margin < {args.margin_threshold})")
    print(f"  -> Final selection happens in --phase 3 "
          f"(KMeans, budget={args.budget})")

    # ----- 3. Merge existing label_hitl patches from a prior --phase 3 -----
    existing_hitl = []
    if assignments_path.exists():
        with open(assignments_path) as f:
            existing = json.load(f)
        first_ann = next(iter(existing["patches"]))
        existing_hitl = [
            p for p in existing["patches"][first_ann]
            if p.get("task_type") == "label_hitl"
        ]
        if existing_hitl:
            print(f"\n  Existing assignments JSON found with "
                  f"{len(existing_hitl)} label_hitl patches "
                  f"(from a prior --phase 3) — will be merged in")

    # ----- 4. Build per-annotator patch list. All 5 see all patches. -----
    rng = np.random.default_rng(seed=AL_DEFAULT_VERIFY_SEED)
    patches_per_ann = {
        a: list(verify_patches) + list(existing_hitl) for a in args.annotators
    }
    for a in args.annotators:
        rng.shuffle(patches_per_ann[a])

    n_per_ann = len(patches_per_ann[args.annotators[0]])

    # ----- 5. Write JSON -----
    output = {
        "round": round_n,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotators": list(args.annotators),
        "task_types": ["verify_pseudo", "label_hitl"],
        "patch_count": n_per_ann,
        "per_annotator_count": n_per_ann,
        "verify_pseudo_count": n_verify,
        "label_hitl_count": len(existing_hitl),
        "patches": patches_per_ann,
    }

    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    with open(assignments_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 65}")
    print(f"  PHASE 2 GENERATE — SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Output:                 {assignments_path}")
    print(f"  Round:                  {round_n}")
    print(f"  Annotators:             {len(args.annotators)}")
    print(f"  Per annotator:          {n_per_ann} patches")
    print(f"    verify_pseudo:        {n_verify}")
    print(f"    label_hitl:           {len(existing_hitl)}")
    est_minutes = n_per_ann * 10 / 60  # ~10 sec per patch (binary check or label)
    print(f"  Est. time per annotator: ~{est_minutes:.0f} min "
          f"(@ 10 sec/patch)")
    print(f"\n  Next steps:")
    print(f"    1. Phase 3 (add label_hitl via KMeans — MUST run BEFORE annotators):")
    print(f"       uv run active_learning_round.py --phase 3 --round {round_n}")
    print(f"    2. Annotators open the web app (mode='al'), complete session")
    print(f"       -> writes annotations/annotations_{{name}}_al_round{round_n}.csv")
    print(f"    3. Phase 2 verify (tally votes, output pseudo + HITL sets):")
    print(f"       uv run active_learning_round.py --phase 2 --subcommand verify "
          f"--round {round_n}")
    print(f"{'=' * 65}")


# ============================================================================
# Phase 2 — verify: tally votes, compute per-class accuracy, output sets
# ============================================================================

# Per-class decision thresholds (overridable via constants)
AL_CLASS_ACC_COMMIT = 0.95       # accuracy >= this -> commit pseudo-labels
AL_CLASS_ACC_RECHECK = 0.50      # accuracy >= this -> bump threshold to 0.95
                                 # accuracy <  RECHECK -> disable pseudo-labels
                                 # (0.50 = random-chance baseline for binary task;
                                 #  classes below this are catastrophically bad)
AL_RECHECK_THRESHOLD = 0.95      # bumped threshold for recheck classes
AL_HITL_MIN_AGREEMENT = 4        # 4/5 or 5/5 accepted (3/5 or worse -> drop)


def _class_decision(accuracy: float) -> str:
    if accuracy >= AL_CLASS_ACC_COMMIT:
        return "commit"
    if accuracy >= AL_CLASS_ACC_RECHECK:
        return "recheck"
    return "disabled"


def _fleiss_kappa_binary(per_patch: pd.DataFrame, n_raters: int) -> float | None:
    """Fleiss' Kappa for binary annotation (correct/wrong), fixed n raters.

    Only patches with exactly n_raters votes are used (others skipped,
    Fleiss' Kappa requires fixed n). Returns None if no usable patches.
    """
    complete = per_patch[per_patch["n_votes"] == n_raters]
    if len(complete) == 0:
        return None
    N = len(complete)
    n = n_raters
    # Per-subject agreement
    p_i = ((complete["n_correct"] ** 2 + complete["n_wrong"] ** 2 - n)
           / (n * (n - 1)))
    P_bar = p_i.mean()
    # Category proportions
    P_correct = complete["n_correct"].sum() / (N * n)
    P_wrong = complete["n_wrong"].sum() / (N * n)
    P_e = P_correct ** 2 + P_wrong ** 2
    if P_e >= 1.0:
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def phase2_verify(args):
    """Phase 2, subcommand: verify.

    Reads annotator CSVs (annotations/annotations_{name}_al_round{N}.csv),
    tallies votes for verify_pseudo and label_hitl patches, computes
    per-class accuracy and Fleiss' Kappa, applies per-class decisions
    to commit/recheck/disable pseudo-labels, and outputs:
      - predictions/pseudo_labeled_set_round{N}.csv
      - predictions/hitl_annotated_round{N}.csv
      - predictions/per_class_accuracy_round{N}.json
    """
    round_n = args.round
    paths = round_paths(round_n)
    assignments_path = paths["assignments"]
    pseudo_out = paths["pseudo_set"]
    hitl_out = paths["hitl_set"]
    per_class_out = paths["per_class"]
    master_path = _resolve_master_csv(args)

    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Assignment file not found: {assignments_path}\n"
            f"Run --phase 2 generate first."
        )

    with open(assignments_path) as f:
        assignments = json.load(f)
    annotators = assignments["annotators"]
    n_annotators = len(annotators)

    if not master_path.exists():
        raise FileNotFoundError(
            f"Master predictions CSV not found: {master_path}\n"
            f"Run --phase 1 first, or pass --predictions-csv."
        )
    master_df = pd.read_csv(master_path)

    print(f"Loading annotator CSVs (round {round_n}):")
    all_votes = []
    for ann in annotators:
        csv_path = ANNOTATIONS_DIR / f"annotations_{ann}_al_round{round_n}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path.name} not found — assuming 0 votes from {ann}")
            continue
        df = pd.read_csv(csv_path)
        print(f"  {ann}: {len(df):,} entries from {csv_path.name}")
        all_votes.append(df)

    if not all_votes:
        raise RuntimeError(
            "No annotator CSVs found. Annotators must complete their session first."
        )
    votes = pd.concat(all_votes, ignore_index=True)
    n_unique_ann = votes["annotator"].nunique()
    print(f"\n  Total entries: {len(votes):,}")
    print(f"  Unique annotators in CSVs: {n_unique_ann} / {n_annotators}")
    if n_unique_ann < n_annotators:
        print(f"  WARNING: {n_annotators - n_unique_ann} annotator(s) did not submit")

    verify_votes = votes[votes["task_type"] == "verify_pseudo"].copy()
    hitl_votes = votes[votes["task_type"] == "label_hitl"].copy()
    print(f"\n  verify_pseudo entries: {len(verify_votes):,}")
    print(f"  label_hitl entries:    {len(hitl_votes):,}")

    # ------------------------------------------------------------------
    # 1. verify_pseudo: tally per patch
    # ------------------------------------------------------------------
    per_patch = pd.DataFrame()
    class_stats = pd.DataFrame()
    kappa = None

    if len(verify_votes) > 0:
        verify_votes["is_correct_bool"] = (
            verify_votes["is_correct"].astype(str).str.strip().str.lower() == "true"
        )
        per_patch = (
            verify_votes.groupby(["patch_path", "class_name"], as_index=False)
            .agg(
                n_correct=("is_correct_bool", "sum"),
                n_votes=("is_correct_bool", "count"),
            )
        )
        per_patch["n_wrong"] = per_patch["n_votes"] - per_patch["n_correct"]
        per_patch["accepted"] = per_patch["n_correct"] >= AL_HITL_MIN_AGREEMENT

        n_total = len(per_patch)
        n_accepted = int(per_patch["accepted"].sum())
        n_rejected = n_total - n_accepted
        n_complete = int((per_patch["n_votes"] == n_annotators).sum())
        print(f"\n  verify_pseudo unique patches: {n_total:,}")
        print(f"    with all {n_annotators} votes:    {n_complete:,}")
        print(f"    accepted (>= {AL_HITL_MIN_AGREEMENT}/{n_annotators} correct): "
              f"{n_accepted:,} ({n_accepted / n_total * 100:.1f}%)")
        print(f"    rejected (< {AL_HITL_MIN_AGREEMENT}/{n_annotators} correct): "
              f"{n_rejected:,} ({n_rejected / n_total * 100:.1f}%)")

        # Per-class accuracy
        class_stats = (
            per_patch.groupby("class_name")
            .apply(
                lambda g: pd.Series({
                    "n_patches": len(g),
                    "n_accepted": int(g["accepted"].sum()),
                    "accuracy": float(g["accepted"].mean()),
                }),
                include_groups=False,
            )
            .reset_index()
        )
        class_stats["decision"] = class_stats["accuracy"].apply(_class_decision)
        class_stats["effective_threshold"] = class_stats["decision"].map(
            {"commit": args.confidence_threshold,
             "recheck": AL_RECHECK_THRESHOLD,
             "disabled": np.nan}
        )

        print(f"\n  Per-class accuracy breakdown:")
        for _, r in class_stats.sort_values("accuracy", ascending=False).iterrows():
            print(f"    {r['class_name']:<55} acc={r['accuracy']:.2f} "
                  f"({int(r['n_accepted'])}/{int(r['n_patches'])}) "
                  f"-> {r['decision']}")

        # Fleiss' Kappa
        kappa = _fleiss_kappa_binary(per_patch, n_annotators)

    else:
        print("\n  No verify_pseudo votes found.")

    # ------------------------------------------------------------------
    # 2. Build pseudo_labeled_set_round{N}.csv
    #    Commit pseudo-labels for classes with decision="commit"
    #    Use recheck threshold for decision="recheck"
    #    Skip decision="disabled" entirely
    # ------------------------------------------------------------------
    if not class_stats.empty:
        commit_classes = set(
            class_stats[class_stats["decision"] == "commit"]["class_name"]
        )
        recheck_classes = set(
            class_stats[class_stats["decision"] == "recheck"]["class_name"]
        )
        disabled_classes = set(
            class_stats[class_stats["decision"] == "disabled"]["class_name"]
        )

        pseudo_rows = []
        for _, row in master_df.iterrows():
            if row.get("agreement_level") == "auto_healthy":
                continue
            cls = row["class_name"]
            if cls in commit_classes:
                thr = args.confidence_threshold
            elif cls in recheck_classes:
                thr = AL_RECHECK_THRESHOLD
            else:
                continue  # disabled or not in verification set
            if float(row["confidence"]) > thr:
                pseudo_rows.append({
                    "patch_path": row["patch_path"],
                    "class_name": cls,
                    "split": row["split"],
                    "predicted_label": int(row["predicted_label"]),
                    "confidence": round(float(row["confidence"]), 6),
                    "margin": round(float(row["margin"]), 6),
                    "source": "pseudo_label_verified",
                    "class_decision": (
                        "commit" if cls in commit_classes else "recheck"
                    ),
                })
        pseudo_df = pd.DataFrame(pseudo_rows)
        pseudo_out.parent.mkdir(parents=True, exist_ok=True)
        pseudo_df.to_csv(pseudo_out, index=False)
        print(f"\n  pseudo_labeled_set: {len(pseudo_df):,} rows -> {pseudo_out.name}")
        print(f"    commit classes:    {len(commit_classes)}")
        print(f"    recheck classes:   {len(recheck_classes)}")
        print(f"    disabled classes:  {len(disabled_classes)}")
    else:
        pseudo_df = pd.DataFrame()
        pseudo_df.to_csv(pseudo_out, index=False)
        print(f"\n  pseudo_labeled_set: 0 rows (no verify_pseudo data)")

    # ------------------------------------------------------------------
    # 3. label_hitl: tally per patch, keep 4/5+5/5 only
    # ------------------------------------------------------------------
    if len(hitl_votes) > 0:
        per_hitl = (
            hitl_votes.groupby(["patch_path", "class_name", "split"], as_index=False)
            .agg(
                n_healthy=("label", lambda s: (s == "healthy").sum()),
                n_unhealthy=("label", lambda s: (s == "unhealthy").sum()),
                n_votes=("label", "count"),
            )
        )
        per_hitl["n_majority"] = per_hitl[["n_healthy", "n_unhealthy"]].max(axis=1)
        per_hitl["majority_label"] = per_hitl.apply(
            lambda r: "healthy" if r["n_healthy"] >= r["n_unhealthy"]
            else "unhealthy",
            axis=1,
        )
        per_hitl["agreement"] = (
            per_hitl["n_majority"].astype(int).astype(str)
            + "/"
            + per_hitl["n_votes"].astype(int).astype(str)
        )

        accepted_hitl = per_hitl[
            per_hitl["n_majority"] >= AL_HITL_MIN_AGREEMENT
        ].copy()

        n_total_hitl = len(per_hitl)
        n_accepted_hitl = len(accepted_hitl)
        print(f"\n  label_hitl unique patches: {n_total_hitl:,}")
        print(f"    accepted (>= {AL_HITL_MIN_AGREEMENT}/{n_annotators} agreement): "
              f"{n_accepted_hitl:,} ({n_accepted_hitl / n_total_hitl * 100:.1f}%)")

        hitl_df = pd.DataFrame([{
            "patch_path": r["patch_path"],
            "class_name": r["class_name"],
            "split": r["split"],
            "label": r["majority_label"],
            "agreement": r["agreement"],
            "n_unanimous": int(r["n_majority"]),
        } for _, r in accepted_hitl.iterrows()])
        hitl_out.parent.mkdir(parents=True, exist_ok=True)
        hitl_df.to_csv(hitl_out, index=False)
        print(f"  hitl_annotated: {len(hitl_df):,} rows -> {hitl_out.name}")
    else:
        print("\n  No label_hitl votes found (Phase 3 may not have run yet).")
        hitl_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "patch_path", "class_name", "split", "label",
            "agreement", "n_unanimous"
        ]).to_csv(hitl_out, index=False)

    # ------------------------------------------------------------------
    # 4. per_class_accuracy_round{N}.json
    # ------------------------------------------------------------------
    per_class_data = {
        "round": round_n,
        "n_annotators": n_annotators,
        "fleiss_kappa": round(kappa, 4) if kappa is not None else None,
        "commit_threshold": AL_CLASS_ACC_COMMIT,
        "recheck_threshold": AL_CLASS_ACC_RECHECK,
        "recheck_class_threshold": AL_RECHECK_THRESHOLD,
        "min_agreement": AL_HITL_MIN_AGREEMENT,
        "class_decisions": {},
    }
    if not class_stats.empty:
        for _, r in class_stats.iterrows():
            per_class_data["class_decisions"][r["class_name"]] = {
                "n_patches": int(r["n_patches"]),
                "n_accepted": int(r["n_accepted"]),
                "accuracy": round(float(r["accuracy"]), 4),
                "decision": r["decision"],
                "effective_threshold": (
                    float(r["effective_threshold"])
                    if pd.notna(r["effective_threshold"]) else None
                ),
            }
    with open(per_class_out, "w") as f:
        json.dump(per_class_data, f, indent=2, ensure_ascii=False)
    print(f"  per_class_accuracy: -> {per_class_out.name}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  PHASE 2 VERIFY — SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Round: {round_n}")
    if kappa is not None:
        interp = (
            "excellent" if kappa >= 0.81
            else "substantial" if kappa >= 0.61
            else "moderate" if kappa >= 0.41
            else "fair" if kappa >= 0.21
            else "slight" if kappa >= 0.0
            else "poor"
        )
        print(f"  Fleiss' Kappa (verify_pseudo): {kappa:.4f} ({interp})")
        if kappa < 0.6:
            print(f"    WARNING: Kappa < 0.6 — flag low-agreement annotators "
                  f"for re-training on guides")
    if not class_stats.empty:
        n_commit = int((class_stats["decision"] == "commit").sum())
        n_recheck = int((class_stats["decision"] == "recheck").sum())
        n_disabled = int((class_stats["decision"] == "disabled").sum())
        print(f"  Per-class decisions: {n_commit} commit, "
              f"{n_recheck} recheck, {n_disabled} disabled")
    print(f"\n  Outputs:")
    print(f"    {pseudo_out}")
    print(f"    {hitl_out}")
    print(f"    {per_class_out}")
    print(f"\n  Next step:")
    print(f"    uv run active_learning_round.py --phase 4 --subcommand compose "
          f"--round {round_n}")
    print(f"    -> merges initial + pseudo + HITL into round{round_n}_dataset.csv")
    print(f"{'=' * 65}")


# ============================================================================
# Phase 3 — select: KMeans on 1280-d embeddings, pick cluster representatives
# ============================================================================

def phase3_select(args):
    """Phase 3, subcommand: select.

    Extracts 1280-d embeddings from EfficientNet-B0's avgpool for the
    uncertain pool (patches with margin < threshold), clusters via
    KMeans(k), and selects the cluster representative with the lowest
    margin per cluster. Caps at --budget and appends to
    al_assignments_round{N}.json as label_hitl patches (model_prediction=null).

    Outputs:
      - predictions/embeddings_round{N}.npy            (N_uncertain, 1280)
      - predictions/cluster_representatives_round{N}.json
      - updates predictions/al_assignments_round{N}.json (adds label_hitl)
    """
    round_n = args.round
    paths = round_paths(round_n)
    master_path = _resolve_master_csv(args)
    assignments_path = paths["assignments"]
    embeddings_path = paths["embeddings"]
    cluster_map_path = paths["cluster_map"]

    if not master_path.exists():
        raise FileNotFoundError(
            f"Master predictions CSV not found: {master_path}\n"
            f"Run --phase 1 first."
        )
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Assignment file not found: {assignments_path}\n"
            f"Run --phase 2 generate first."
        )

    master_df = pd.read_csv(master_path)
    uncertain = master_df[
        (master_df["agreement_level"] == "unlabeled") &
        (master_df["margin"] < args.margin_threshold)
    ].copy().reset_index(drop=True)

    print(f"Phase 3 select (round {round_n})")
    print(f"  Uncertain pool:  {len(uncertain):,} patches "
          f"(margin < {args.margin_threshold})")
    print(f"  Budget cap:      {args.budget}")

    if len(uncertain) == 0:
        print("  No uncertain patches. Writing empty outputs and exiting.")
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, np.empty((0, 1280), dtype=np.float32))
        with open(cluster_map_path, "w") as f:
            json.dump({"round": round_n, "k": 0, "budget": args.budget,
                       "n_uncertain_pool": 0, "representatives": []}, f, indent=2)
        return

    # Compute K: either user override or compute_k() formula
    if args.k is not None:
        k = min(args.k, len(uncertain))
        if k < args.k:
            print(f"  WARNING: pool ({len(uncertain)}) < user-specified k ({args.k}); "
                  f"using k={k}")
        k_source = f"user override ({args.k})"
    else:
        k = compute_k(args.budget, len(uncertain),
                      n_classes=args.n_classes,
                      min_cluster_size=args.min_cluster_size)
        k_source = (f"compute_k(budget={args.budget}, pool={len(uncertain)}, "
                    f"n_classes={args.n_classes}, "
                    f"min_cluster_size={args.min_cluster_size})")
        # Annotate non-default outcomes for the user
        if k == args.n_classes and k < args.budget:
            print(f"  NOTE: K was floored at n_classes={args.n_classes} "
                  f"(budget={args.budget} too small to dominate)")
        k_ceiling = max(1, len(uncertain) // args.min_cluster_size)
        if k == k_ceiling and k < args.budget:
            print(f"  NOTE: K was capped at pool//min_cluster_size={k_ceiling} "
                  f"(budget={args.budget} exceeds pool capacity)")
    print(f"  KMeans k:        {k}  (source: {k_source})")

    # ------------------------------------------------------------------
    # 1. Load model, register forward hook on avgpool for 1280-d features
    # ------------------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"  Device:          {device}")
    print(f"  Loading model:   {args.model}")
    model = load_model(args.model, device)
    model.eval()

    features: dict = {}
    def hook(module, input, output):
        # EfficientNet avgpool output: (B, 1280, 1, 1) -> flatten to (B, 1280)
        features["x"] = output.flatten(1)
    handle = model.avgpool.register_forward_hook(hook)

    # ------------------------------------------------------------------
    # 2. Build dataset, run inference, collect embeddings
    # ------------------------------------------------------------------
    transform = build_transform()
    entries = [
        {
            "abs_path": str(args.patches_dir / row["patch_path"]),
            "rel_path": row["patch_path"],
            "class_name": row["class_name"],
        }
        for _, row in uncertain.iterrows()
    ]
    ds = InferenceDataset(entries, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_embs = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Embeddings", unit="batch"):
            images = images.to(device)
            _ = model(images)  # hook captures features["x"]
            all_embs.append(features["x"].cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0)
    print(f"  Embeddings:      {embeddings.shape}")
    handle.remove()

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)

    # ------------------------------------------------------------------
    # 3. KMeans clustering
    # ------------------------------------------------------------------
    from sklearn.cluster import KMeans
    print(f"  Running KMeans(k={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    cluster_id_to_size: dict[int, int] = {}
    for cid in range(k):
        cluster_id_to_size[cid] = int((cluster_labels == cid).sum())
    sizes_arr = np.array(list(cluster_id_to_size.values()))
    max_cluster_pct = float(sizes_arr.max() / len(uncertain))
    print(f"  Cluster sizes:   min={sizes_arr.min()}, "
          f"max={sizes_arr.max()}, mean={sizes_arr.mean():.1f}")
    if max_cluster_pct > AL_CLUSTER_SANITY_THRESHOLD:
        print(f"  WARNING: largest cluster has {max_cluster_pct * 100:.1f}% of pool "
              f"(> {AL_CLUSTER_SANITY_THRESHOLD * 100:.0f}%). "
              f"Consider HDBSCAN or larger --k.")

    # ------------------------------------------------------------------
    # 4. Pick cluster representative (argmin margin)
    #    Skip empty clusters (KMeans can produce them on small pools).
    # ------------------------------------------------------------------
    representatives = []
    for cid in range(k):
        cluster_mask = cluster_labels == cid
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_margins = uncertain.iloc[cluster_indices]["margin"].values
        best_local = int(np.argmin(cluster_margins))
        best_idx = int(cluster_indices[best_local])
        best_row = uncertain.iloc[best_idx]
        representatives.append({
            "cluster_id": cid,
            "patch_path": best_row["patch_path"],
            "class_name": best_row["class_name"],
            "split": best_row["split"],
            "margin": float(best_row["margin"]),
            "cluster_size": cluster_id_to_size[cid],
        })

    # Sort by margin ascending (lowest margin = most informative), cap at budget
    representatives.sort(key=lambda r: r["margin"])
    if len(representatives) > args.budget:
        print(f"  Capping at budget={args.budget} "
              f"(had {len(representatives)} reps)")
        representatives = representatives[: args.budget]

    # ------------------------------------------------------------------
    # 5. Update al_assignments_round{N}.json: add label_hitl patches
    # ------------------------------------------------------------------
    with open(assignments_path) as f:
        assignments = json.load(f)

    hitl_patches = [{
        "patch_path": r["patch_path"],
        "class_name": r["class_name"],
        "split": r["split"],
        "task_type": "label_hitl",
        "model_prediction": None,
        "cluster_id": r["cluster_id"],
        "margin": round(r["margin"], 6),
    } for r in representatives]

    n_verify = int(assignments.get("verify_pseudo_count", 0))
    rng = np.random.default_rng(seed=AL_DEFAULT_VERIFY_SEED)
    for ann in assignments["annotators"]:
        verify_only = [p for p in assignments["patches"][ann]
                       if p.get("task_type") == "verify_pseudo"]
        combined = list(verify_only) + hitl_patches
        rng.shuffle(combined)
        assignments["patches"][ann] = combined

    assignments["verify_pseudo_count"] = n_verify
    assignments["label_hitl_count"] = len(hitl_patches)
    assignments["patch_count"] = n_verify + len(hitl_patches)
    assignments["per_annotator_count"] = n_verify + len(hitl_patches)
    assignments["created_at"] = datetime.now(timezone.utc).isoformat()

    with open(assignments_path, "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6. cluster_representatives_round{N}.json
    # ------------------------------------------------------------------
    cluster_map_data = {
        "round": round_n,
        "k": int(k),
        "k_source": k_source,
        "budget": int(args.budget),
        "n_uncertain_pool": int(len(uncertain)),
        "max_cluster_pct": round(max_cluster_pct, 4),
        "cluster_sizes": {str(cid): size for cid, size
                          in cluster_id_to_size.items()},
        "representatives": [{
            "cluster_id": r["cluster_id"],
            "patch_path": r["patch_path"],
            "class_name": r["class_name"],
            "margin": round(r["margin"], 6),
            "cluster_size": r["cluster_size"],
        } for r in representatives],
    }
    with open(cluster_map_path, "w") as f:
        json.dump(cluster_map_data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  PHASE 3 SELECT — SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Round:                  {round_n}")
    print(f"  Uncertain pool:         {len(uncertain):,}")
    print(f"  KMeans k:               {k}  (source: {k_source})")
    print(f"  Cluster reps:           {len(representatives)}")
    print(f"  Budget cap:             {args.budget}")
    print(f"  Annotators:             {len(assignments['annotators'])}")
    print(f"  Per annotator total:    "
          f"{n_verify} verify_pseudo + {len(hitl_patches)} label_hitl = "
          f"{n_verify + len(hitl_patches)}")
    print(f"\n  Outputs:")
    print(f"    {embeddings_path}")
    print(f"    {cluster_map_path}")
    print(f"    {assignments_path} (updated with {len(hitl_patches)} label_hitl)")
    print(f"\n  Next steps:")
    print(f"    1. Annotators open the web app (mode='al'), complete session")
    print(f"    2. Run: uv run active_learning_round.py --phase 2 --subcommand verify "
          f"--round {round_n}")
    print(f"{'=' * 65}")


# ============================================================================
# Phase 4 — compose: merge initial + pseudo + HITL into round{N}_dataset.csv
# ============================================================================

# Per-class cap defaults for pseudo-labels
AL_PSEUDO_CAP_RATIO = 2.0    # pseudo count <= ratio * (initial + HITL) per class
AL_PSEUDO_CAP_MIN = 100      # minimum cap when initial+HITL = 0
AL_PSEUDO_CAP_MAX = 1000     # hard cap to bound per-class pseudo count


# NOTE: cross-round design is "clean-slate" — round{N}_dataset.csv contains
#       only THIS round's pseudo + HITL, NOT previous rounds'. See module
#       header docstring "Cross-round design" section for rationale.
def phase4_compose(args):
    """Phase 4, subcommand: compose.

    Merges three sources into round{N}_dataset.csv:
      - initial: 4/5 + 5/5 from consensus_review_master.csv
      - pseudo: pseudo_labeled_set_round{N}.csv (from phase 2 verify)
      - hitl:    hitl_annotated_round{N}.csv (from phase 2 verify)

    Applies a per-class cap to pseudo-labels to prevent imbalance
    (pseudo count <= AL_PSEUDO_CAP_RATIO * (initial + HITL) per class,
     bounded by [AL_PSEUDO_CAP_MIN, AL_PSEUDO_CAP_MAX]).

    Output: round{N}_dataset.csv with columns:
      patch_path, class_name, split, suggested_numeric_label,
      source, agreement_level
    """
    round_n = args.round
    paths = round_paths(round_n)
    consensus_path = args.consensus_csv
    pseudo_path = paths["pseudo_set"]
    hitl_path = paths["hitl_set"]
    output_path = BASE_DIR / f"round{round_n}_dataset.csv"

    if not consensus_path.exists():
        raise FileNotFoundError(
            f"consensus CSV not found: {consensus_path}"
        )
    if not pseudo_path.exists():
        raise FileNotFoundError(
            f"pseudo_labeled_set not found: {pseudo_path}\n"
            f"Run --phase 2 verify first."
        )
    if not hitl_path.exists():
        raise FileNotFoundError(
            f"hitl_annotated not found: {hitl_path}\n"
            f"Run --phase 2 verify first."
        )

    print(f"Phase 4 compose (round {round_n})")

    # ------------------------------------------------------------------
    # 1. Load initial annotated (4/5 + 5/5)
    # ------------------------------------------------------------------
    print(f"\n  Loading initial annotated: {consensus_path.name}")
    consensus_df = pd.read_csv(consensus_path)
    initial_df = consensus_df[
        consensus_df["agreement_level"].isin(["4/5", "5/5"])
    ].copy()
    initial_df["split"] = initial_df["patch_path"].apply(
        lambda p: str(p).split("/")[0] if pd.notna(p) else "train"
    )
    initial_df["source"] = "initial"
    initial_df = initial_df[[
        "patch_path", "class_name", "split",
        "suggested_numeric_label", "source", "agreement_level",
    ]]
    print(f"    Initial (4/5+5/5): {len(initial_df):,} patches")

    # ------------------------------------------------------------------
    # 2. Load pseudo-labeled
    # ------------------------------------------------------------------
    print(f"\n  Loading pseudo-labeled: {pseudo_path.name}")
    pseudo_df = pd.read_csv(pseudo_path)
    if len(pseudo_df) > 0:
        pseudo_df = pseudo_df.rename(
            columns={"predicted_label": "suggested_numeric_label"}
        )
        pseudo_df["source"] = "pseudo"
        pseudo_df["agreement_level"] = pseudo_df.get(
            "class_decision", "auto"
        ).fillna("auto")
        pseudo_df = pseudo_df[[
            "patch_path", "class_name", "split",
            "suggested_numeric_label", "source", "agreement_level",
        ]]
    else:
        pseudo_df = pd.DataFrame(columns=[
            "patch_path", "class_name", "split",
            "suggested_numeric_label", "source", "agreement_level",
        ])
    print(f"    Pseudo-labeled: {len(pseudo_df):,} patches (pre-cap)")

    # ------------------------------------------------------------------
    # 3. Load HITL annotated
    # ------------------------------------------------------------------
    print(f"\n  Loading HITL: {hitl_path.name}")
    hitl_df = pd.read_csv(hitl_path)
    if len(hitl_df) > 0:
        hitl_df["suggested_numeric_label"] = hitl_df["label"].map(
            {"healthy": 0, "unhealthy": 1}
        )
        hitl_df = hitl_df.rename(columns={"agreement": "agreement_level"})
        hitl_df["source"] = "hitl"
        hitl_df = hitl_df[[
            "patch_path", "class_name", "split",
            "suggested_numeric_label", "source", "agreement_level",
        ]]
    else:
        hitl_df = pd.DataFrame(columns=[
            "patch_path", "class_name", "split",
            "suggested_numeric_label", "source", "agreement_level",
        ])
    print(f"    HITL annotated: {len(hitl_df):,} patches")

    # ------------------------------------------------------------------
    # 4. Apply per-class cap to pseudo-labels
    # ------------------------------------------------------------------
    print(f"\n  Applying per-class cap to pseudo-labels "
          f"(ratio={AL_PSEUDO_CAP_RATIO}, min={AL_PSEUDO_CAP_MIN}, "
          f"max={AL_PSEUDO_CAP_MAX})...")
    real_per_class = (
        pd.concat([initial_df[["class_name"]], hitl_df[["class_name"]]])
        .groupby("class_name").size().to_dict()
    )

    capped_rows = []
    cap_log = []
    for cls, group in pseudo_df.groupby("class_name"):
        real_count = real_per_class.get(cls, 0)
        cap = max(AL_PSEUDO_CAP_MIN, int(AL_PSEUDO_CAP_RATIO * real_count))
        cap = min(cap, AL_PSEUDO_CAP_MAX)
        n_orig = len(group)
        n_kept = min(n_orig, cap)
        if n_orig > cap:
            sampled = group.sample(n=cap, random_state=AL_DEFAULT_VERIFY_SEED)
            cap_log.append((cls, n_orig, n_kept))
        else:
            sampled = group
        capped_rows.append(sampled)

    if capped_rows:
        pseudo_capped_df = pd.concat(capped_rows, ignore_index=True)
    else:
        pseudo_capped_df = pseudo_df.iloc[:0]

    n_pseudo_dropped = len(pseudo_df) - len(pseudo_capped_df)
    if cap_log:
        print(f"    Capped {len(cap_log)} classes (top 10):")
        for cls, n_orig, n_kept in sorted(cap_log, key=lambda x: -x[1])[:10]:
            real = real_per_class.get(cls, 0)
            print(f"      {cls:<55} {n_orig:>5} -> {n_kept:>5} "
                  f"(real={real}, cap={max(AL_PSEUDO_CAP_MIN, int(AL_PSEUDO_CAP_RATIO * real))})")
        if len(cap_log) > 10:
            print(f"      ... and {len(cap_log) - 10} more")
    print(f"    Pseudo after cap: {len(pseudo_capped_df):,} "
          f"(dropped {n_pseudo_dropped:,})")

    # ------------------------------------------------------------------
    # 5. Combine all sources
    # ------------------------------------------------------------------
    final_df = pd.concat([
        initial_df,
        pseudo_capped_df,
        hitl_df,
    ], ignore_index=True)

    # Deduplicate: prefer initial > hitl > pseudo (in that priority)
    n_before = len(final_df)
    priority = {"initial": 0, "hitl": 1, "pseudo": 2}
    final_df["_priority"] = final_df["source"].map(priority).fillna(3)
    final_df = (
        final_df.sort_values("_priority")
        .drop_duplicates(subset=["patch_path"], keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
    n_dupes_removed = n_before - len(final_df)
    if n_dupes_removed > 0:
        print(f"\n  Removed {n_dupes_removed} duplicate patches "
              f"(kept highest-priority source per patch)")

    # Sanity: validate labels
    bad_labels = final_df[
        ~final_df["suggested_numeric_label"].isin([0, 1])
    ]
    if len(bad_labels) > 0:
        print(f"\n  WARNING: {len(bad_labels)} patches have invalid labels")

    # ------------------------------------------------------------------
    # 6. Write output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  PHASE 4 COMPOSE — SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Output:           {output_path}")
    print(f"  Total patches:    {len(final_df):,}")
    src_counts = final_df["source"].value_counts().to_dict()
    print(f"    initial:        {src_counts.get('initial', 0):,}")
    print(f"    pseudo:         {src_counts.get('pseudo', 0):,}")
    print(f"    hitl:           {src_counts.get('hitl', 0):,}")
    total = len(final_df)
    label_dist = final_df["suggested_numeric_label"].value_counts().to_dict()
    print(f"  Label distribution:")
    print(f"    healthy (0):    {label_dist.get(0, 0):,} "
          f"({label_dist.get(0, 0) / total * 100:.1f}%)")
    print(f"    unhealthy (1):  {label_dist.get(1, 0):,} "
          f"({label_dist.get(1, 0) / total * 100:.1f}%)")

    # Per-source per-class breakdown (top 10)
    if total > 0:
        print(f"\n  Per-source per-class breakdown (top 10 by total):")
        breakdown = (
            final_df.groupby(["class_name", "source"]).size()
            .unstack(fill_value=0)
        )
        for col in ("initial", "pseudo", "hitl"):
            if col not in breakdown.columns:
                breakdown[col] = 0
        breakdown["total"] = breakdown[["initial", "pseudo", "hitl"]].sum(axis=1)
        for cls, row in breakdown.sort_values("total", ascending=False).head(10).iterrows():
            print(f"    {str(cls)[:55]:<55} init={int(row.get('initial', 0)):>4} "
                  f"pseudo={int(row.get('pseudo', 0)):>4} "
                  f"hitl={int(row.get('hitl', 0)):>3} "
                  f"total={int(row['total'])}")

    print(f"\n  Next step:")
    print(f"    uv run active_learning_round.py --phase 5 --subcommand train "
          f"--round {round_n}")
    print(f"    -> fine-tunes train_consensus_model.py on {output_path.name}")
    print(f"{'=' * 65}")


# ============================================================================
# Phase 5 — train: invoke train_consensus_model.py on round{N}_dataset.csv
# ============================================================================

import subprocess
import sys as _sys

TRAIN_SCRIPT = BASE_DIR / "train_consensus_model.py"

# Hyperparameter overrides for active learning rounds (user can still pass
# their own --epochs, --lr, etc. via the wrapper's --train-extra-args passthrough)
AL_DEFAULT_TRAIN_EPOCHS = 15
AL_DEFAULT_TRAIN_LR = 5e-5    # lower than Round 1 (1e-4) for fine-tuning
AL_DEFAULT_TRAIN_BATCH_SIZE = 64


def _resolve_init_checkpoint(round_n: int) -> Path | None:
    """Default init checkpoint for fine-tuning:
      - Round 2: Round 1 (consensus) model
      - Round N > 2: Round N-1 model
    Returns None if neither exists (caller will use ImageNet).

    For Round 2, candidates are tried in order:
      1. models/round1/model.pt                                       (current convention)
      2. models/efficientnet_b0_consensus/efficientnet_b0_consensus.pt  (legacy)
    """
    if round_n == 2:
        candidates = [
            BASE_DIR / "models" / "round1" / "model.pt",
            BASE_DIR / "models" / "efficientnet_b0_consensus" / "efficientnet_b0_consensus.pt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None
    candidate = BASE_DIR / "models" / f"round{round_n - 1}" / "model.pt"
    return candidate if candidate.exists() else None


def phase5_train(args):
    """Phase 5, subcommand: train.

    Invokes train_consensus_model.py as a subprocess to fine-tune
    EfficientNet-B0 on round{N}_dataset.csv. Saves the resulting
    checkpoint to models/round{N}/model.pt.

    By default, fine-tunes from the previous round's checkpoint
    (or Round 1's consensus model for Round 2). Pass
    --no-init-checkpoint to start from ImageNet instead.
    """
    round_n = args.round
    round_dataset = BASE_DIR / f"round{round_n}_dataset.csv"
    output_dir = BASE_DIR / "models" / f"round{round_n}"
    output_pt = output_dir / "model.pt"

    if not round_dataset.exists():
        raise FileNotFoundError(
            f"Round {round_n} dataset not found: {round_dataset}\n"
            f"Run --phase 4 compose first."
        )
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"train_consensus_model.py not found: {TRAIN_SCRIPT}"
        )

    # Resolve init checkpoint (default: previous round)
    if args.no_init_checkpoint:
        init_ckpt = None
    else:
        init_ckpt = _resolve_init_checkpoint(round_n)

    # Build subprocess command
    cmd = [
        _sys.executable, str(TRAIN_SCRIPT),
        "--consensus-csv", str(round_dataset),
        "--patches-dir", str(args.patches_dir),
        "--output", str(output_pt),
        "--use-all-rows",       # include pseudo + HITL (not just 4/5+5/5)
    ]
    if init_ckpt is not None:
        cmd += ["--init-checkpoint", str(init_ckpt)]
    if args.train_epochs is not None:
        cmd += ["--epochs", str(args.train_epochs)]
    if args.train_lr is not None:
        cmd += ["--lr", str(args.train_lr)]
    if args.train_batch_size is not None:
        cmd += ["--batch-size", str(args.train_batch_size)]

    print(f"Phase 5 train (round {round_n})")
    print(f"  Dataset:        {round_dataset}")
    print(f"  Patches dir:    {args.patches_dir}")
    print(f"  Output:         {output_pt}")
    print(f"  Init ckpt:      {init_ckpt if init_ckpt else '(ImageNet)'}")
    print(f"  Epochs:         {args.train_epochs or AL_DEFAULT_TRAIN_EPOCHS}")
    print(f"  LR:             {args.train_lr or AL_DEFAULT_TRAIN_LR}")
    print(f"  Batch size:     {args.train_batch_size or AL_DEFAULT_TRAIN_BATCH_SIZE}")
    print(f"\n  Command:")
    print(f"    {' '.join(cmd)}")
    print(f"\n  Running training (this will take a while)...")

    # Stream output to console
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Training failed with exit code {e.returncode}. "
            f"See output above for details."
        ) from e

    print(f"\n{'=' * 65}")
    print(f"  PHASE 5 TRAIN — SUMMARY")
    print(f"{'=' * 65}")
    if output_pt.exists():
        size_mb = output_pt.stat().st_size / (1024 * 1024)
        print(f"  Checkpoint:     {output_pt} ({size_mb:.1f} MB)")
    else:
        print(f"  WARNING: expected checkpoint not found at {output_pt}")
    print(f"  Round: {round_n}  ->  {output_pt.name} ({size_mb:.1f} MB)")
    print(f"\n  Next step (iterate to Round {round_n + 1}):")
    print(f"    1. Re-run --phase 1 with the new model + new --round:")
    print(f"       uv run active_learning_round.py --phase 1 "
          f"--round {round_n + 1} --model {output_pt}")
    print(f"    2. uv run active_learning_round.py --phase 2 --subcommand generate "
          f"--round {round_n + 1}")
    print(f"    3. uv run active_learning_round.py --phase 3 --round {round_n + 1}")
    print(f"    4. [Annotators complete session in web app, mode='al']")
    print(f"    5. uv run active_learning_round.py --phase 2 --subcommand verify "
          f"--round {round_n + 1}")
    print(f"    6. uv run active_learning_round.py --phase 4 --subcommand compose "
          f"--round {round_n + 1}")
    print(f"    7. uv run active_learning_round.py --phase 5 --subcommand train "
          f"--round {round_n + 1}")
    print(f"\n  Or evaluate before iterating: uv run stop_check.py --round {round_n}")
    print(f"{'=' * 65}")


# ============================================================================
# Phase 1 — Plots (3 carried over + 4 new AL-specific)
# ============================================================================

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


def plot_margin_distribution(df, output_dir):
    """Primary active-learning signal. Patches with low margin are the most
    informative — they sit near the decision boundary. Auto-healthy rows
    are excluded (margin=1.0 by construction, would dominate the histogram).
    """
    uncertain = df[df["agreement_level"] == "unlabeled"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for split_name, color in [("train", "#3498db"), ("test", "#e74c3c")]:
        subset = uncertain[uncertain["split"] == split_name]["margin"]
        if len(subset) > 0:
            ax.hist(subset, bins=80, alpha=0.55, density=True,
                    label=f"{split_name} (n={len(subset):,})", color=color)
            med = float(np.median(subset))
            ax.axvline(med, color=color, linestyle=":", linewidth=1.0,
                       alpha=0.7, label=f"{split_name} median ({med:.3f})")

    ax.set_xlabel("Margin = P(top-1) - P(top-2)")
    ax.set_ylabel("Density")
    ax.set_title("Phase 1 — Margin Distribution (Unlabeled Disease Patches)\n"
                 "Left side = near decision boundary = most informative for HITL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "margin_distribution.png")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'margin_distribution.png'}")


def plot_confidence_vs_margin(df, output_dir):
    """2D scatter showing the four AL quadrants. Bottom-left = HITL candidates.

    Uses a hexbin density background (all unlabeled rows) overlaid with
    a downsampled scatter (50k max) for per-class color detail. This
    keeps both pieces of information visible at 800k+ rows.
    """
    uncertain = df[df["agreement_level"] == "unlabeled"]
    if len(uncertain) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 9))

    # Density background — all points binned into hexes (grayscale)
    hb = ax.hexbin(uncertain["confidence"], uncertain["margin"],
                   gridsize=40, cmap="Greys", alpha=0.3, mincnt=5)

    # Downsampled scatter overlay for class-color detail
    if len(uncertain) > AL_SCATTER_MAX_POINTS:
        plot_data = uncertain.sample(n=AL_SCATTER_MAX_POINTS, random_state=42)
    else:
        plot_data = uncertain
    colors = np.where(plot_data["predicted_label"].values == 0,
                      "#3498db", "#e74c3c")
    ax.scatter(plot_data["confidence"], plot_data["margin"],
               c=colors, alpha=0.4, s=10, edgecolors="none")

    ax.axvline(0.9, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0.2, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

    ax.text(0.04, 0.97, "Confident + high margin\n(auto-accept candidates)",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            color="#27ae60", alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#27ae60", alpha=0.7))
    ax.text(0.97, 0.04, "Low conf + low margin\n(HITL candidates)",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            color="#c0392b", alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#c0392b", alpha=0.7))
    ax.text(0.04, 0.04, "Confident but\nlow margin",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            color="gray", alpha=0.7)
    ax.text(0.97, 0.97, "High margin\nbut low conf",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            color="gray", alpha=0.7)

    cb = fig.colorbar(hb, ax=ax, label="patch count", shrink=0.7)

    ax.set_xlabel("Confidence (P(top-1))")
    ax.set_ylabel("Margin (P(top-1) - P(top-2))")
    ax.set_title("Phase 1 — Confidence vs Margin\n"
                 "blue=predicted healthy, red=predicted unhealthy")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(output_dir / "confidence_vs_margin_scatter.png")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'confidence_vs_margin_scatter.png'}")


def plot_top1_vs_top2(df, output_dir):
    """Top-1 vs top-2 confidence. Above y=x the model is correctly ranked;
    near the y=x diagonal the top-1 and top-2 are nearly tied (confused).
    """
    uncertain = df[df["agreement_level"] == "unlabeled"]
    if len(uncertain) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 9))

    hb = ax.hexbin(uncertain["confidence"], uncertain["top2_confidence"],
                   gridsize=40, cmap="Greys", alpha=0.3, mincnt=5)

    if len(uncertain) > AL_SCATTER_MAX_POINTS:
        plot_data = uncertain.sample(n=AL_SCATTER_MAX_POINTS, random_state=42)
    else:
        plot_data = uncertain
    colors = np.where(plot_data["predicted_label"].values == 0,
                      "#3498db", "#e74c3c")
    ax.scatter(plot_data["confidence"], plot_data["top2_confidence"],
               c=colors, alpha=0.4, s=10, edgecolors="none")

    lims = [0, 1]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5,
            label="y = x (top-1 == top-2, model is tied)")
    ax.plot(lims, [1 - x for x in lims], "k:", linewidth=0.8, alpha=0.5,
            label="y = 1 - x (sum=1, valid 2-class region)")

    cb = fig.colorbar(hb, ax=ax, label="patch count", shrink=0.7)

    ax.set_xlabel("Top-1 Confidence")
    ax.set_ylabel("Top-2 Confidence")
    ax.set_title("Phase 1 — Top-1 vs Top-2 Confidence\n"
                 "Near diagonal = confused; far from diagonal = confidently correct")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(output_dir / "top1_vs_top2_scatter.png")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'top1_vs_top2_scatter.png'}")


def plot_uncertain_per_class(df, output_dir, margin_thr):
    """Horizontal bar of uncertain patches per disease class.

    These are the classes the model is struggling with — directly informs
    Phase 2 per-class threshold bumps.
    """
    uncertain = df[(df["agreement_level"] == "unlabeled") & (df["margin"] < margin_thr)]
    if len(uncertain) == 0:
        return

    counts = (
        uncertain.groupby("class_name")
        .size()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.32)))
    ax.barh(range(len(counts)), counts.values, color="#e67e22", alpha=0.85)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Patches with margin < {margin_thr}")
    ax.set_title(f"Phase 1 — Uncertain Patches per Class (margin < {margin_thr})\n"
                 "Top of list = classes the model is struggling with")
    ax.grid(True, axis="x", alpha=0.3)

    if len(counts) > 0:
        max_v = max(counts.values)
        for i, val in enumerate(counts.values):
            ax.text(val + max_v * 0.005, i, f"{val:,}",
                    va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(output_dir / "uncertain_per_class.png")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'uncertain_per_class.png'}")


# ============================================================================
# Phase 1 — Summary printer
# ============================================================================

def print_phase1_summary(df, default_threshold, margin_thr):
    total = len(df)
    n_review = int(df["needs_review"].sum())
    n_auto = total - n_review
    label_dist = df["predicted_label"].value_counts()

    uncertain = df[df["agreement_level"] == "unlabeled"]
    high_conf = int((uncertain["confidence"] > default_threshold).sum()) if len(uncertain) else 0
    low_margin = int((uncertain["margin"] < margin_thr).sum()) if len(uncertain) else 0

    print(f"\n{'=' * 65}")
    print(f"  PHASE 1 — INFERENCE & SPLIT SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Total patches:                  {total:,}")
    print(f"  Predicted healthy:              {int(label_dist.get(0, 0)):,} "
          f"({label_dist.get(0, 0) / total * 100:.1f}%)")
    print(f"  Predicted unhealthy:            {int(label_dist.get(1, 0)):,} "
          f"({label_dist.get(1, 0) / total * 100:.1f}%)")
    print(f"  Auto-accepted (conf > thr):     {n_auto:,} "
          f"({n_auto / total * 100:.1f}%)")
    print(f"  Needs review (conf <= thr):     {n_review:,} "
          f"({n_review / total * 100:.1f}%)")

    if len(uncertain) > 0:
        print(f"\n  Unlabeled disease pool:         {len(uncertain):,}")
        print(f"    High confidence (> {default_threshold}):       "
              f"{high_conf:,} ({high_conf / len(uncertain) * 100:.1f}%)  "
              f"-> pseudo-label candidates")
        print(f"    Low margin (< {margin_thr}):              "
              f"{low_margin:,} ({low_margin / len(uncertain) * 100:.1f}%)  "
              f"-> HITL candidates")
        print(f"    Median margin:                 {float(np.median(uncertain['margin'])):.4f}")
        print(f"    Median confidence:             {float(np.median(uncertain['confidence'])):.4f}")

    print(f"\n  By split:")
    for split in ["train", "test"]:
        for subdir_name in ["healthy", "needs_annotation"]:
            key_label = "auto_healthy" if subdir_name == "healthy" else "unlabeled"
            subset = df[(df["split"] == split) & (df["agreement_level"] == key_label)]
            if len(subset) > 0:
                n_r = int(subset["needs_review"].sum())
                med_m = float(np.median(subset["margin"])) if "margin" in subset.columns else 0.0
                print(f"    {split}/{subdir_name}/: {len(subset):,} predicted, "
                      f"{n_r:,} needs review ({n_r / len(subset) * 100:.1f}%), "
                      f"median margin={med_m:.3f}")

    uncertain_class = (
        uncertain[uncertain["margin"] < margin_thr]
        .groupby("class_name")
        .size()
        .sort_values(ascending=False)
    )
    if len(uncertain_class) > 0:
        print(f"\n  Top 5 classes with most uncertain patches (margin < {margin_thr}):")
        for i, (cls, cnt) in enumerate(uncertain_class.head(5).items(), 1):
            print(f"    {i}. {cls:<50} {cnt:,} patches")


# ============================================================================
# Main dispatcher
# ============================================================================

def main():
    args = parse_args()

    if args.model is None:
        args.model = _resolve_default_model(args.round)

    if args.phase == 1:
        phase1_main(args)
    elif args.phase == 2:
        if args.subcommand == "generate":
            phase2_generate(args)
        elif args.subcommand == "verify":
            phase2_verify(args)
        else:
            raise ValueError(f"Unknown --subcommand: {args.subcommand}")
    elif args.phase == 3:
        phase3_select(args)
    elif args.phase == 4:
        if args.subcommand == "compose":
            phase4_compose(args)
        else:
            raise ValueError(f"Unknown --subcommand for --phase 4: {args.subcommand}")
    elif args.phase == 5:
        if args.subcommand == "train":
            phase5_train(args)
        else:
            raise ValueError(f"Unknown --subcommand for --phase 5: {args.subcommand}")


if __name__ == "__main__":
    main()
