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
├── active_learning_round.py         # AL pipeline (phases 1-5: select, compose, train)
├── filtering_gambar.py              # Filter source images by selected crops
├── patch_splitting.py               # Split source images into 32×32 patches
│
├── fleiss_kappa_per_class.png       # Inter-annotator agreement plot
│
├── templates/
│   ├── index.html                   # Annotation interface
│   └── dashboard.html               # Progress dashboard
│
├── predictions/                     # Active learning artifacts
│   ├── al_assignments_round{N}.json       # AL mode assignments per round (committed)
│   ├── cluster_representatives_round{N}.json  # Cluster metadata for label_hitl (committed)
│   ├── master_predictions_round{N}.csv    # Model predictions (gitignored — too large)
│   └── embeddings_round{N}.npy            # Patch embeddings (gitignored — too large)
│
├── dataset_patches/                 # Full patch dataset (gitignored, source of truth)
│   ├── train/needs_annotation/      # Training patches (22 classes)
│   └── test/needs_annotation/       # Test patches (22 classes)
│
├── dataset_consensus_only/          # 5% consensus subset (gitignored, what gets annotated)
│   └── train/needs_annotation/      # Sampled patches (22 classes)
│
├── dataset_filtered/                # Filtered source images for leaf context (gitignored)
│   └── raw/segmented/{class_name}/
│
├── annotations/                     # Per-annotator CSV files (gitignored, output)
├── models/                          # Trained model checkpoints (gitignored)
└── docs/
```

---

## Setup

**For friends/annotators who only need to label patches:** skip to the
[AL Mode for Annotators](#al-mode-for-annotators-friends) section below —
you don't need most of the operator setup (extract, assign, train, etc.).

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

### Generate Round Assignments (Operator Only)

```bash
python active_learning_round.py --phase 2 generate --round 2
```

Produces `predictions/al_assignments_round2.json` (320 patches × 5
annotators) and `cluster_representatives_round2.json`. Commit both
files so annotators can `git pull` to start the next AL round.

**Note:** annotators do NOT run this. The operator runs phases 1-5 and
commits the assignments JSON.

### AL Mode for Annotators (Friends)

If you're a friend/annotator and only need to do AL annotation (not
run the operator pipeline):

1. **Pull latest code:**
   ```bash
   git pull
   ```

   This brings down `predictions/al_assignments_round2.json` and the
   latest source code. Your local `annotations/` and `annotator_config.json`
   are gitignored and remain untouched.

2. **Ensure the shared drive is mounted.** AL mode needs
   `dataset_patches/` (gitignored, ~4.6 GB) for patch images. The
   shared drive typically has this. If it's not at the project root:
   ```bash
   ln -s /path/to/shared-drive/dataset_patches dataset_patches
   ```
   Without `dataset_patches/`, AL mode will show broken-image icons.
   The server prints a `[WARN]` at startup if it's missing.

3. **Run the app** (same as normal annotation):
   ```bash
   uv run uvicorn main:app --reload --host localhost --port 8000
   ```

   First startup takes ~20s while the leaf index is built. Subsequent
   starts are faster (the lock is cached).

4. **Enter your name** in the setup modal (must match
   `assignments_consensus.json`).

5. **Click "Active Learning"** in the top bar. The default round is 2
   (`AL_DEFAULT_ROUND = 2` in `main.py`); the modal prompts you to
   confirm or change it.

6. **Annotate** — two task types alternate:
   - **VERIFIKASI PSEUDO-LABEL** (blue banner): **Benar** / **Salah** —
     validate the model's prediction
   - **LABEL HITL** (orange banner): **Sehat** / **Tidak Sehat** —
     label the patch manually (model was uncertain about this cluster)

7. **Resumable.** Your local
   `annotations/annotations_{name}_al_round{N}.csv` is gitignored.
   Close the browser or restart the app anytime; your progress is
   preserved. Re-opening AL mode picks up where you left off.

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
| AL round 2 (verify_pseudo + label_hitl, 5 annotators) | 320 | 22 |

- **Consensus subset:** 5% random sample (per class) of diseased source
  leaves from the full dataset
- **Full dataset:** all patches before consensus voting — used for final
  model training and prediction
- **AL round 2:** 220 verify_pseudo + 100 label_hitl patches distributed
  across 5 annotators; 87 are test split (used as gold-standard for
  evaluating future complex models)

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

**AL mode shows broken-image icons for all patches**
- AL mode needs `dataset_patches/` for the patch images (gitignored)
- Normal/review modes use `dataset_consensus_only/` (different, smaller set)
- Check the server log on startup for `[WARN] dataset_patches/ not found`
- Fix: mount the shared drive, or create a symlink:
  ```bash
  ln -s /path/to/shared-drive/dataset_patches dataset_patches
  ```

**Server startup takes 20s on first run**
- Eager leaf index build (34,795 leaves) on startup is intentional
- Subsequent starts are fast (still 12-20s depending on disk speed)
- To skip the eager build and use lazy build instead, comment out the
  `build_leaf_index()` call in `startup_event` in `main.py`

**Error installing torch with CUDA**
- You don't need ML dependencies for annotation — skip `requirements-ml.txt`
- Only run `pip install -r requirements.txt` or `uv sync`
