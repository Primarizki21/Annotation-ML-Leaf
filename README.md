# 🌿 Annotation-ML-Leaf

> End-to-end pipeline for **plant disease detection** on PlantVillage: 5-annotator consensus labeling → Fleiss-κ review → active learning rounds → 5-architecture model training with ONNX export.

![Annotation interface — full leaf, current patch, and grid overview](screenshot/First_annotation_stage.png)

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA 13](https://img.shields.io/badge/CUDA-13.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda)
[![uv](https://img.shields.io/badge/uv-FF6B35?style=for-the-badge&logo=astral&logoColor=white)](https://docs.astral.sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![Patches](https://img.shields.io/badge/patches-234,729-4CAF50?style=flat-square)](dataset_patches/)
[![Classes](https://img.shields.io/badge/classes-22-4CAF50?style=flat-square)](dataset_patches/)
[![Annotators](https://img.shields.io/badge/annotators-5-2196F3?style=flat-square)](assignments_consensus.json)
[![AL rounds](https://img.shields.io/badge/active_learning-rounds_2--4-FF6B6B?style=flat-square)](predictions/)
[![Last commit](https://img.shields.io/github/last-commit/Primarizki21/Annotation-ML-Leaf?style=flat-square)](https://github.com/Primarizki21/Annotation-ML-Leaf/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/Primarizki21/Annotation-ML-Leaf?style=flat-square)](https://github.com/Primarizki21/Annotation-ML-Leaf)

---

## What is this?

A solo-built research codebase that closes the loop on **plant leaf disease classification** end-to-end:

1. **Annotation** — A FastAPI web app where 5 human annotators label 32×32 patches as *healthy* or *unhealthy* across 22 PlantVillage classes. Consensus voting + Fleiss' κ flags disputed patches for review.
2. **Active learning** — A 5-phase loop (`inference → verify pseudo-label → HITL on uncertain clusters → compose → train`) that runs round after round, with an automatic `stop_check` based on agreement and accuracy plateaus.
3. **Training** — 5 mobile-friendly architectures (MobileNetV3-S, EfficientNet-B0, ShuffleNetV2, SqueezeNet, SmallInception) trained and exported to ONNX for inference.

It's the smallest full-stack ML pipeline I could ship that still does everything for real: human-in-the-loop labeling, multi-annotator agreement, multi-round active learning, and a model zoo. No external services, no managed databases — just a single Python process, a folder of patches, and a CSV.

---

## Screenshots

**Annotation mode** — single patch in focus, full-leaf context on the left, grid overview on the right. All three views stay in sync as you label.

![Annotation_HITL mode](screenshot/Annotation_HITL.png)

**Active Learning — Round 2 (HITL)** — model predictions and uncertain-cluster patches, both annotatable. The blue banner is *verify pseudo-label*; the orange banner is *label manually* (model was unsure).

---

## Features

| | |
|---|---|
| 🎯 **Multi-annotator patch annotation** | 5 annotators, per-class assignment, autosave every annotation, CSV backup every 100. |
| 🔄 **Review mode for disputes** | Patches where 3+ annotators disagree are routed back for re-labeling. Fleiss' κ computed per class. |
| 🧠 **Active learning loop** | Verify pseudo-labels (high confidence) + label HITL (low margin, KMeans-clustered). 4 rounds completed. |
| 🏗️ **5-architecture training** | MobileNetV3-S, EfficientNet-B0, ShuffleNetV2, SqueezeNet, SmallInception. 2-phase (frozen backbone → fine-tune), early stop, resume-safe. |
| 📊 **Live dashboard** | Per-annotator, per-class, per-round progress at `/dashboard`, auto-refreshes every 30s. |
| 💾 **Resume-safe** | Local CSVs survive crashes. Web app state is gitignored; operator artifacts (`*.json`) are committed. |

---

## Pipeline

```mermaid
flowchart LR
    A["🌿 Source images<br/>PlantVillage"] --> B["patch_splitting.py<br/>32×32 patches"]
    B --> C["5 annotators<br/>consensus voting"]
    C --> D{"Fleiss κ"}
    D -->|"5/5 or 4/5"| F["Round 1 train<br/>EfficientNet-B0"]
    D -->|"≤3/2 dispute"| E["Review mode"]
    E --> C
    F --> G["Phase 1<br/>inference + margin"]
    G --> H["Phase 2<br/>verify_pseudo"]
    G --> I["Phase 3<br/>HITL KMeans"]
    H --> J["Web app<br/>annotate"]
    I --> J
    J --> K["Phase 4<br/>compose"]
    K --> L["Phase 5<br/>train round N"]
    L --> M{"stop_check"}
    M -->|"continue"| G
    M -->|"stop κ≥0.81"| N["Final model<br/>+ ONNX"]
```

Detailed per-phase mermaid diagrams: see [`docs/active_learning_flow.md`](docs/active_learning_flow.md).

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Backend** | [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com) [![Uvicorn](https://img.shields.io/badge/Uvicorn-2E2E2E?style=flat-square&logo=python&logoColor=white)](https://www.uvicorn.org) [![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=flat-square&logo=jinja&logoColor=white)](https://palletsprojects.com/p/jinja/) |
| **Frontend** | [![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)]() [![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)]() [![Vanilla JS](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)]() — no SPA framework, just templates + a small JS module layer |
| **Annotation** | [![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org) [![Pillow](https://img.shields.io/badge/Pillow-8A8A8A?style=flat-square&logo=python&logoColor=white)](https://python-pillow.org) |
| **ML** | [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org) [![torchvision](https://img.shields.io/badge/torchvision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/vision) [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org) [![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnx.ai) [![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai) |
| **Viz** | [![matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=flat-square)](https://matplotlib.org) |
| **Tooling** | [![uv](https://img.shields.io/badge/uv-FF6B35?style=flat-square&logo=astral&logoColor=white)](https://docs.astral.sh/uv) — single lockfile, CUDA 13 PyTorch index |

---

## Project Structure

```
Annotation-ML-Leaf/
├── main.py                          # FastAPI backend server
├── pyproject.toml                   # Project config & dependencies (uv)
├── requirements.txt                 # Core dependencies for annotation
├── requirements-ml.txt              # ML dependencies (training/prediction)
│
├── extract_consensus.py             # Extract 5% consensus subset
├── assignments_generator.py         # Generate assignments_consensus.json
├── assignments_consensus.json       # Annotator-to-folder mapping
│
├── merge_annotations.py             # Merge per-annotator CSVs → master CSV
├── merge_annotation_consensus.py    # Vote tally, Fleiss Kappa, review CSV
│
├── train_consensus_model.py         # Train EfficientNet-B0 (active learning)
├── predict_unlabeled_patches.py     # Predict labels for remaining patches
├── active_learning_round.py         # AL pipeline (phases 1-5)
├── filtering_gambar.py              # Filter source images by selected crops
├── patch_splitting.py               # Split source images into 32×32 patches
├── eda.py                           # Exploratory data analysis on patches
│
├── stop_check.py                    # Convergence gate (κ, plateau, disabled classes)
│
├── templates/
│   ├── index.html                   # Annotation interface
│   └── dashboard.html               # Progress dashboard
├── static/                          # CSS + JS modules
├── screenshot/                      # README screenshots
├── docs/                            # active_learning_flow.md (mermaid)
│
├── predictions/                     # Active learning artifacts
│   ├── al_assignments_round{N}.json       # AL mode assignments (committed)
│   ├── cluster_representatives_round{N}.json  # Cluster metadata (committed)
│   ├── hitl_annotated_round{N}.csv        # HITL labels per round (committed)
│   ├── per_class_accuracy_round{N}.json   # Round accuracy report (committed)
│   ├── master_predictions_round{N}.csv    # Predictions (gitignored — too large)
│   └── embeddings_round{N}.npy            # Patch embeddings (gitignored)
│
├── dataset_patches/                 # Full patch dataset (gitignored)
├── dataset_consensus_only/          # 5% consensus subset (gitignored)
├── dataset_filtered/                # Filtered source images (gitignored)
├── annotations/                     # Per-annotator CSVs (gitignored)
├── models/                          # Trained checkpoints (gitignored)
└── outputs/                         # Training outputs (gitignored)
```

---

## Quick Start

### 🧑‍🏫 Annotator (friend)

You only need the web app to label. The operator handles the rest.

```bash
git clone https://github.com/Primarizki21/Annotation-ML-Leaf.git
cd Annotation-ML-Leaf

# install
uv sync                                    # or: pip install -r requirements.txt

# get the shared dataset
ln -s /path/to/shared-drive/dataset_patches dataset_patches

# run
uv run uvicorn main:app --reload --port 8000
# open http://localhost:8000
```

For AL rounds: `git pull` first to fetch the latest `predictions/al_assignments_round{N}.json`, then click **Active Learning** in the top bar.

### 🛠️ Operator (you)

Build the consensus dataset and kick off round 1:

```bash
uv sync --extra ml

# 1. Sample 5% consensus subset from full dataset
uv run python extract_consensus.py

# 2. Generate per-annotator assignments
uv run python assignments_generator.py

# 3. (annotators label via the web app)

# 4. Merge annotations + Fleiss κ
uv run python merge_annotations.py
uv run python merge_annotation_consensus.py

# 5. Train round 1 baseline
uv run python train_consensus_model.py
```

### 🤖 ML Engineer

Train any of the 5 architectures:

```bash
uv sync --extra ml
uv run python train_model_script/train_mobilenetv3_small.py
uv run python train_model_script/train_efficientnet_b0.py
uv run python train_model_script/train_shufflenetv2.py
uv run python train_model_script/train_squeezenet.py
uv run python train_model_script/train_small_inception.py
```

See [`train_model_script/INFERENCE.md`](train_model_script/INFERENCE.md) for ONNX export.

---

## Usage

### Annotation mode
1. On first launch, enter your name (must match `assignments_consensus.json`).
2. View the enlarged patch (32px upscaled with pixelated rendering).
3. Use keyboard shortcuts or buttons:
   - `H` / **Healthy** — label as healthy
   - `U` / **Unhealthy** — label as unhealthy
   - `S` / **Skip** — skip
   - `←` / **Undo** — undo last annotation
4. Progress autosaves after each annotation. CSV backup every 100.

### Leaf context view
Click **Show Leaf** to open a grid of all patches from the same source leaf. Annotated patches are color-coded (green = healthy, red = unhealthy). Click any tile to jump to it.

### Review mode
After consensus voting, split-decision patches are flagged `needs_discussion`. Re-annotate them in review mode → re-run the merge script to finalize.

### Progress dashboard
Visit [`/dashboard`](http://localhost:8000/dashboard) for per-annotator, per-class, and overall progress. Auto-refreshes every 30s.

---

## Active Learning

5 phases, run as separate commands. Output paths auto-update with `--round N`.

| Phase | Command | Output |
|---|---|---|
| 1. Inference | `uv run active_learning_round.py --phase 1 --round N` | `master_predictions_roundN.csv` |
| 2a. Generate | `uv run active_learning_round.py --phase 2 --subcommand generate --round N` | `al_assignments_roundN.json` |
| 2b. Verify | `uv run active_learning_round.py --phase 2 --subcommand verify --round N` | `pseudo_labeled_set_roundN.csv`, `hitl_annotated_roundN.csv` |
| 3. HITL select | `uv run active_learning_round.py --phase 3 --round N --budget 100 --margin-threshold 0.7` | adds `label_hitl` patches to `al_assignments_roundN.json` |
| 4. Compose | `uv run active_learning_round.py --phase 4 --subcommand compose --round N` | `roundN_dataset.csv` |
| 5. Train | `uv run active_learning_round.py --phase 5 --subcommand train --round N` | `models/roundN/model.pt` |

**Annotators** complete steps 2a → 3 via the web app (Active Learning mode).

**Stop gate** (`stop_check.py`): continues while Fleiss κ < 0.81, val accuracy still improving, uncertain pool ≥ 50, no disabled classes.

Full mermaid flow: [`docs/active_learning_flow.md`](docs/active_learning_flow.md).

---

## Training

5 architectures, all with the same 2-phase schedule (frozen backbone warmup → full fine-tune) and early stop.

| Architecture | File | Use case |
|---|---|---|
| MobileNetV3-Small | `train_mobilenetv3_small.py` | Mobile / edge deployment, smallest |
| EfficientNet-B0 | `train_efficientnet_b0.py` | Best accuracy/efficiency tradeoff |
| ShuffleNetV2 | `train_shufflenetv2.py` | Mobile, low FLOPs |
| SqueezeNet 1.1 | `train_squeezenet.py` | Tiny model, 0.5 MB |
| SmallInception | `train_small_inception.py` | Custom small inception block |

All scripts share common logic in [`_train_common.py`](train_model_script/_train_common.py). Default hyperparams tuned for 8-12 GB VRAM:

```bash
--batch_size 128 --save_every 1 --num_workers 8 --patience 5
```

ONNX export and inference recipe: [`train_model_script/INFERENCE.md`](train_model_script/INFERENCE.md).

---

## Dataset Statistics

| Dataset | Patches | Classes |
|---|---|---|
| Full (`dataset_patches/`, gitignored) | 234,729 | 22 |
| Consensus subset (`dataset_consensus_only/`, gitignored) | 34,611 | 22 |
| AL round 2 (verify_pseudo + label_hitl, 5 annotators) | 320 | 22 |
| AL round 3 | 320 | 22 |
| AL round 4 | 320 | 22 |

- **Consensus subset:** 5% random sample (per class) of diseased source leaves from the full dataset
- **Full dataset:** all patches before consensus voting — used for final model training and prediction
- **AL rounds:** 220 verify_pseudo + 100 label_hitl patches per round, distributed across 5 annotators. 87 are test split (used as gold-standard for evaluating future complex models)

---

## Documentation

- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — common issues by pipeline stage
- [`docs/active_learning_flow.md`](docs/active_learning_flow.md) — full mermaid diagrams of the AL pipeline
- [`train_model_script/INFERENCE.md`](train_model_script/INFERENCE.md) — ONNX export and inference
- [`active_learning_int.md`](active_learning_int.md) — AL internals deep-dive
- [`training_script.md`](training_script.md) — training pipeline notes
- [`fixing_loader.md`](fixing_loader.md) — dataloader fix history

---

## License

[MIT](LICENSE) © 2026 Primarizki Hariyono.

---

## Author

**Primarizki Hariyono** — [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/primarizkihariyono/) [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Primarizki21)

Built solo. Acknowledgments to the PlantVillage dataset and the 5 friends who did the annotation rounds.
