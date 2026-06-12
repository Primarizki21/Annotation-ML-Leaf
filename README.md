# Patch Annotation App

A local web application for image patch annotation using FastAPI (backend)
and plain HTML/CSS/JavaScript (frontend). Used by 5 annotators to label plant
leaf patches as **healthy** or **unhealthy** for a plant disease detection ML
project.

Supports a **normal annotation mode** for initial labeling and a **review mode**
for resolving disputed patches after consensus voting.

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
├── consensus_review_master.csv      # Patches needing discussion
│
├── train_consensus_model.py         # Train EfficientNet-B0 (active learning)
├── predict_unlabeled_patches.py     # Predict labels for remaining patches
├── filtering_gambar.py              # Filter source images by selected crops
├── patch_splitting.py               # Split source images into 32×32 patches
│
├── fleiss_kappa_per_class.png       # Inter-annotator agreement plot
│
├── templates/
│   ├── index.html                   # Annotation interface
│   └── dashboard.html               # Progress dashboard
│
├── dataset_patches/                 # Full patch dataset (source of truth)
│   ├── train/needs_annotation/      # Training patches (22 classes)
│   └── test/needs_annotation/       # Test patches (22 classes)
│
├── dataset_consensus_only/          # 5% consensus subset (what gets annotated)
│   └── train/needs_annotation/      # Sampled patches (22 classes)
│
├── dataset_filtered/                # Filtered source images for leaf context
│   └── raw/segmented/{class_name}/
│
├── annotations/                     # Per-annotator CSV files (output)
├── models/                          # Trained model checkpoints
└── docs/
```

---

## Setup

### Prerequisites

- Python 3.12+
- **Recommended:** [uv](https://docs.astral.sh/uv/) — fast Python package manager
- **Alternative:** pip + venv (if uv is not available)

### Quick Start (Annotation)

For annotators who only need to label patches:

**With uv:**
```bash
uv sync
uv run python extract_consensus.py
uv run python assignments_generator.py
```

**With pip:**
```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
python extract_consensus.py
python assignments_generator.py
```

### ML Training Setup (Optional)

Only needed if you're training the model or running predictions:

**With uv:**
```bash
uv sync --extra ml
```

**With pip:**
```bash
pip install -r requirements-ml.txt
```

---

## Running the App

```bash
uv run uvicorn main:app --reload --host localhost --port 8000
# or: uvicorn main:app --reload --host localhost --port 8000
```

Open your browser at [http://localhost:8000](http://localhost:8000).

---

## Usage

### Normal Annotation Mode

1. On first launch, enter your name (must match a name in `assignments_consensus.json`)
2. View the enlarged patch image (32px upscaled with pixelated rendering)
3. Use keyboard shortcuts or click buttons:
   - **H** / **Healthy** — Label as healthy
   - **U** / **Unhealthy** — Label as unhealthy
   - **S** / **Skip** — Skip this patch
   - **ArrowLeft** / **Undo** — Undo last annotation
4. Progress is saved automatically after each annotation
5. CSV backup is created every 100 annotations

### Leaf Context View

Click **"Show Leaf"** to open a grid overlay showing all patches from the
same source leaf. Annotated patches are color-coded (green = healthy,
red = unhealthy, gray = unlabeled). Click any patch in the grid to jump
directly to it.

### Review Mode

After consensus voting, some patches may be flagged as `needs_discussion`.
These can be re-annotated in review mode:

1. Click **"Review Mode"** on the setup page
2. Annotate only the disputed patches
3. Same shortcuts and workflow as normal mode
4. Results are saved to `annotations/annotations_{name}_review.csv`

### Progress Dashboard

- Access at [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- Shows per-annotator, per-class, and overall progress
- Auto-refreshes every 30 seconds

---

## Annotation Pipeline (Post-Annotation)

After all annotators finish labeling, run these steps to build the
consensus dataset:

### 1. Merge Individual CSVs

Collect all `annotations/annotations_*.csv` files, then:

```bash
python merge_annotations.py
```

Output: `annotations_master.csv` — all annotations in one file, with
duplicate and conflict detection.

### 2. Vote Tally & Fleiss Kappa

```bash
python merge_annotation_consensus.py
```

This script:
- Tallies votes per patch (healthy / unhealthy / skip) across annotators
- Computes **Fleiss' Kappa** inter-annotator agreement (overall + per class)
- Generates `consensus_review_master.csv` with:
  - **5/5 agreement** → automatically labeled as consensus
  - **4/5 agreement** → automatically labeled as majority
  - **split votes (3-2 or lower)** → flagged `needs_discussion`
  - Plots agreement (`fleiss_kappa_per_class.png`)

### 3. Resolve Disputed Patches (Review)

Annotators use **Review Mode** (see above) to re-annotate patches flagged
`needs_discussion`. After resolving, re-run the merge script to finalize
the consensus dataset.

---

## Active Learning (Optional ML Loop)

After consensus labels are finalized, you can train a model and predict
labels for the remaining unlabeled patches.

### Fleiss Kappa Agreement

The inter-annotator agreement plot is saved to `fleiss_kappa_per_class.png`.
Open it to see agreement levels per disease class.

### Train Model

```bash
python train_consensus_model.py
```

Trains an **EfficientNet-B0** on consensus patches. Saves the best
checkpoint, training metrics, and evaluation plots to `models/`.

### Predict Remaining Patches

```bash
python predict_unlabeled_patches.py
```

Loads the trained checkpoint and predicts labels for all patches in
`dataset_patches/` that are not part of the consensus set. Output goes
to `predictions/master_predictions.csv`.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| H | Label as Healthy |
| U | Label as Unhealthy |
| S | Skip patch |
| ArrowLeft | Undo last annotation |

---

## CSV Output Format

```csv
patch_path,class_name,split,label,annotator,timestamp,is_skipped
train/needs_annotation/Tomato___Bacterial_spot/img.jpg,Tomato___Bacterial_spot,train,healthy,Budi,2026-06-01T14:30:00,False
```

---

## Dataset Statistics

| Dataset | Patches | Classes |
|---------|---------|---------|
| Full (dataset_patches) | 234,729 | 22 |
| Consensus subset (dataset_consensus_only) | 34,611 | 22 |

- **Consensus subset:** 5% random sample (per class) of diseased source
  leaves from the full dataset
- **Full dataset:** all patches before consensus voting — used for final
  model training and prediction

---

## Troubleshooting

**"assignments_consensus.json not found"**
- Run `python assignments_generator.py` or `uv run python assignments_generator.py`

**"Annotator not found"**
- Check your name matches exactly in `assignments_consensus.json`
- Run `assignments_generator.py` with your name included

**"consensus_review_master.csv not found"**
- Make sure `merge_annotation_consensus.py` has been run first

**"No patches for review"**
- Review mode requires `consensus_review_master.csv` with `needs_discussion`
  patches — run the merge script first

**Server won't start**
- Make sure you're in the project directory
- Check if port 8000 is available
- Try a different port: `uvicorn main:app --port 8001`

**Images not loading**
- Verify `dataset_consensus_only` folder exists
- Run `python extract_consensus.py` if missing
- Check folder permissions

**Error installing torch with CUDA**
- You don't need ML dependencies for annotation — skip `requirements-ml.txt`
- Only run `pip install -r requirements.txt` or `uv sync`
