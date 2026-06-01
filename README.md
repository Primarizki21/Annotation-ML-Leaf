# Patch Annotation App

A local web application for image patch annotation using FastAPI (backend) and plain HTML/CSS/JavaScript (frontend). Used by 5 annotators to label plant leaf patches as "healthy" or "unhealthy" for a plant disease detection ML project.

## Project Structure

```
AnotasiProgramML/
├── main.py                      # FastAPI backend server
├── merge_annotations.py         # Merge all annotator CSV files
├── assignments_generator.py     # Generate assignments.json
├── templates/
│   ├── index.html               # Annotation interface
│   └── dashboard.html           # Progress dashboard
├── dataset_patches/             # INPUT: image patches
│   ├── train/needs_annotation/  # Training patches (22 classes)
│   └── test/needs_annotation/   # Test patches (22 classes)
├── annotations/                 # OUTPUT: per-annotator CSV files
├── annotator_config.json        # Created on first launch
└── assignments.json             # Created by project lead
```

## Setup Instructions

### For Project Lead (uv)

1. Clone or download the project
2. Install dependencies:
   ```bash
   cd AnotasiProgramML
   uv init --python 3.12
   uv add fastapi uvicorn python-multipart pandas pillow jinja2
   ```
3. Generate assignments for annotators:
   ```bash
   uv run python assignments_generator.py
   ```
   This creates `assignments.json` mapping each annotator to their assigned class folders.
4. Share the project folder with team members (via Google Drive, USB, etc.)

### For Team Members (pip)

1. Download the shared project folder
2. Install dependencies:
   ```bash
   cd AnotasiProgramML
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   uvicorn main:app --reload --host localhost --port 8000
   ```
4. Open browser: `http://localhost:8000`

## Usage

### Annotating Patches

1. On first launch, enter your name (must match assignments.json)
2. View the enlarged patch image (64px upscaled to 320px with pixelated rendering)
3. Use keyboard shortcuts or click buttons:
   - **H** or click **Healthy** - Label as healthy
   - **U** or click **Unhealthy** - Label as unhealthy
   - **S** or click **Skip** - Skip this patch
   - **ArrowLeft** or click **Undo** - Undo last annotation
4. Progress is saved automatically after each annotation
5. CSV backup is created every 100 annotations

### Progress Dashboard

- Access at: `http://localhost:8000/dashboard`
- Shows per-annotator, per-class, and overall progress
- Auto-refreshes every 30 seconds

### Merging Annotations

After all annotators complete their work:

1. Collect all `annotations/annotations_*.csv` files
2. Place them in the `annotations/` directory
3. Run the merge script:
   ```bash
   python merge_annotations.py
   ```
4. Output: `annotations_master.csv` with all annotations merged

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| H | Label as Healthy |
| U | Label as Unhealthy |
| S | Skip patch |
| ArrowLeft | Undo last annotation |

## CSV Output Format

```csv
patch_path,class_name,split,label,annotator,timestamp,is_skipped
train/needs_annotation/Tomato___Bacterial_spot/img.jpg,Tomato___Bacterial_spot,train,healthy,Budi,2026-06-01T14:30:00,False
```

## Dataset Statistics

- **Train:** 186,490 patches across 22 disease classes
- **Test:** 48,239 patches across 22 disease classes
- **Total:** 234,729 patches to annotate

## Troubleshooting

**"assignments.json not found"**
- Run `python assignments_generator.py` first

**"Annotator not found"**
- Check your name matches exactly in assignments.json
- Run `python assignments_generator.py` with your name included

**Server won't start**
- Make sure you're in the project directory
- Check if port 8000 is available
- Try a different port: `uvicorn main:app --port 8001`

**Images not loading**
- Verify dataset_patches folder exists
- Check folder permissions
