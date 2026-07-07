# Troubleshooting

Common issues and fixes, organized by stage of the pipeline.

---

## Setup

### `assignments_consensus.json not found`
- Run `python assignments_generator.py` (or `uv run python assignments_generator.py`) to create it.

### `Annotator not found`
- Your entered name must match a key in `assignments_consensus.json` exactly (case-sensitive).
- Re-run `assignments_generator.py` with your name included in the annotator list.

### `Error installing torch with CUDA`
- You don't need ML dependencies for pure annotation work. Skip `requirements-ml.txt` and only run `pip install -r requirements.txt` or `uv sync`.
- For CUDA 13 wheels, the lockfile already pins the correct index; if pip complains, use `uv` instead.

---

## Running the Web App

### Server won't start
- Confirm you're in the project root directory.
- Check if port 8000 is available: `lsof -i :8000` (Linux/macOS) or `netstat -ano | findstr :8000` (Windows).
- Try a different port: `uvicorn main:app --port 8001`.

### Images not loading
- Verify `dataset_consensus_only/` exists in the project root.
- If missing, run `python extract_consensus.py` to recreate it.
- Check folder permissions: `chmod -R u+r dataset_consensus_only`.

### Server startup takes 15-20s on first run
- Eager leaf index build (34,795 leaves) on startup is intentional — it powers the "Leaf Overview" grid.
- Subsequent starts are faster (still 12-20s depending on disk speed).
- To skip eager build and use lazy build instead, comment out the `build_leaf_index()` call in the `startup_event` in `main.py`.

---

## Annotation

### `consensus_review_master.csv not found`
- Run `python merge_annotation_consensus.py` first to generate it from the per-annotator CSVs.

### `No patches for review`
- Review mode requires `consensus_review_master.csv` to contain rows flagged `needs_discussion`.
- That happens when 3 or more annotators disagree (Fleiss κ vote is split). If none exist, every patch has clear consensus — nothing to review.

---

## Active Learning (Round Mode)

### AL mode shows broken-image icons for all patches
- AL mode needs the full `dataset_patches/` directory (gitignored, ~4.6 GB) for patch images.
- Normal/review modes use the smaller `dataset_consensus_only/` — different data, different folders.
- Check the server log at startup for `[WARN] dataset_patches/ not found`.
- Fix: mount the shared drive, or create a symlink:
  ```bash
  ln -s /path/to/shared-drive/dataset_patches dataset_patches
  ```

### `master_predictions_roundN.csv not found`
- Phase 1 (inference) must run first. The command is in `active_learning_round.py --phase 1 --round N`.
- Renaming `predictions/master_predictions.csv` to add the `_roundN` suffix is also a valid first step.

### Phase 3 KMeans produces no HITL patches
- The margin threshold may be too strict for a confident model. Try lowering `--margin-threshold` (e.g. `0.5` or `0.3`).
- If the unlabeled pool is already exhausted, AL has nothing left to sample — consider stopping.

### `al_assignments_roundN.json` not visible in the app
- Restart the web server after the operator commits a new round's JSON. The assignments are read on startup, not hot-reloaded.
- Confirm the file is at the path the server expects: `predictions/al_assignments_roundN.json`.

---

## Training (ML)

### CUDA out of memory during training
- Lower `--batch_size` (default is 128 after the latest tune; try 64 or 32 for tight VRAM).
- For ShuffleNetV2 / SqueezeNet / SmallInception, batch size 128 fits on most 8GB+ cards.
- For EfficientNet-B0 at full resolution, 64 is safer.
- Close other GPU processes: `nvidia-smi` then `kill -9 <pid>`.

### `train_consensus_model.py` slow first epoch
- The DataLoader warms up on epoch 0 (prefetch_factor=4 + path cache build). Epochs 1+ are typically 3-5× faster.
- Check `train_model_script/benchmark_dataloader.py` to measure your dataloader throughput before assuming a real slowdown.

### ONNX export fails (`torch.onnx.export` errors)
- Confirm `torch` and `onnx` versions are compatible — pinned in `uv.lock`.
- Try `train_model_script/preprocess_inference.py` first to get a sanitized state dict.
- See `train_model_script/INFERENCE.md` for the full ONNX export + inference recipe.

### `stop_check.py` says STOP too early
- Default thresholds: Fleiss κ ≥ 0.81, no disabled classes, val-acc plateau, uncertain pool < 50, needs-review ratio < 1%.
- Loosen `--kappa-min` and `--max-uncertain` flags to defer the stop.
- Inspect the printed reasons — the script lists which gate tripped.

---

## Still stuck?

Open an issue on the [GitHub repo](https://github.com/Primarizki21/Annotation-ML-Leaf/issues) with:
1. The full command you ran
2. The full error (traceback + last 20 lines of the server log)
3. Output of `python --version` and `uv --version` (or `pip --version`)
4. Your OS and Python version
