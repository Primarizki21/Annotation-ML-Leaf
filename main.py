"""
main.py - FastAPI Image Patch Annotation App

A local web application for image patch annotation using FastAPI.
Used by annotators to label plant leaf patches as "healthy" or "unhealthy".

============================================================================
USAGE — Three modes
============================================================================
The annotator_config.json has a "mode" field that selects one of three modes.

Mode 1: Normal annotation (initial ~34k patches)
  annotator_config.json: {"name": "Oki", "mode": "normal"}
  Reads:  assignments_consensus.json
  Writes: annotations/annotations_{name}.csv
  Start: python main.py -> open http://localhost:8000 -> "Start Annotating"

Mode 2: Review disputed (verify 3/5 patches)
  annotator_config.json: {"name": "Oki", "mode": "review"}
  Reads:  consensus_review_master.csv (filters needs_discussion=True)
  Writes: annotations/annotations_{name}_review.csv
  Or: click "Review Disputed" button in the UI

Mode 3: Active Learning (NEW — verify_pseudo + label_hitl, Round N)
  annotator_config.json: {"name": "Oki", "mode": "al"}
  Reads:  predictions/al_assignments_round{N}.json (must exist; run
            `python active_learning_round.py --phase 2 generate --round N` first)
  Writes: annotations/annotations_{name}_al_round{N}.csv
  After all 5 annotators finish, run:
            `python active_learning_round.py --phase 2 verify --round N`

============================================================================
SWITCHING MODES
============================================================================
Edit annotator_config.json manually, OR use the API:
  POST /api/setup           -> mode = "normal"
  POST /api/setup-review    -> mode = "review"
  POST /api/setup-al        -> mode = "al" (with round in body)
"""

import csv
import json
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI(title="Patch Annotation App")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class ExceptionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # Print full traceback to stderr (shows in uvicorn log)
            print("\n" + "=" * 60, file=sys.stderr)
            print(f"ERROR: {request.method} {request.url.path}", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("=" * 60 + "\n", file=sys.stderr)
            # Return error page with traceback
            tb = traceback.format_exc()
            return PlainTextResponse(
                f"Internal Server Error\n\n{tb}",
                status_code=500,
            )


app.add_middleware(ExceptionLoggingMiddleware)

# Paths
BASE_DIR = Path(__file__).parent
# DATASET_DIR = BASE_DIR / "dataset_patches" # original patches
DATASET_DIR = BASE_DIR / "dataset_consensus_only"
DATASET_FILTERED_DIR = BASE_DIR / "dataset_filtered"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
CONFIG_PATH = BASE_DIR / "annotator_config.json"
# ASSIGNMENTS_PATH = BASE_DIR / "assignments.json" # original json
ASSIGNMENTS_PATH = BASE_DIR / "assignments_consensus.json"
CONSENSUS_REVIEW_CSV = BASE_DIR / "consensus_review_master.csv"
PREDICTIONS_DIR = BASE_DIR / "predictions"  # active learning artifacts (AL mode)

# Ensure annotations directory exists
ANNOTATIONS_DIR.mkdir(exist_ok=True)

# In-memory session store
sessions: dict[str, dict] = {}

# Placeholder SVG for missing images
PLACEHOLDER_SVG = b'''<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32">
  <rect width="32" height="32" fill="#333"/>
  <text x="32" y="30" text-anchor="middle" fill="#888" font-size="9">Missing</text>
  <text x="32" y="42" text-anchor="middle" fill="#888" font-size="9">Image</text>
</svg>'''

BACKUP_INTERVAL = 100
MAX_HISTORY = 20
PATCH_SIZE = 32

# Regex to extract row/col from patch filename
PATCH_RC_RE = re.compile(r"__r(\d+)_c(\d+)\.\w+$")

# In-memory leaf index: (split, class_name, leaf_stem) -> leaf data
leaf_index: dict[tuple[str, str, str], dict] = {}


def build_leaf_index():
    """Build leaf index from metadata. Called lazily on first leaf-context request.

    Looks in two locations, in priority order:
      1. dataset_patches/metadata_{split}.json  (newer, may have test data)
      2. dataset_consensus_only/metadata_{split}.json  (older, train only)
    First source wins for duplicate entries. Falls back to image-only mode
    (api_leaf_context) for leaves that aren't in any metadata.
    """
    global leaf_index
    if leaf_index:
        return  # Already built

    import time
    start = time.time()
    leaf_index = {}

    # Collect metadata sources in priority order
    metadata_sources = []
    # 1. dataset_patches/ (newer — may have test metadata from patch_splitting.py)
    if (BASE_DIR / "dataset_patches").exists():
        for split in ("train", "test"):
            p = BASE_DIR / "dataset_patches" / f"metadata_{split}.json"
            if p.exists():
                metadata_sources.append((split, p, "dataset_patches"))
    # 2. dataset_consensus_only/ (older — train metadata)
    for split in ("train", "test"):
        p = DATASET_DIR / f"metadata_{split}.json"
        if p.exists():
            # Skip if already added from dataset_patches (first source wins)
            already = any(s == split and "dataset_patches" in src
                          for s, _, src in metadata_sources)
            if not already:
                metadata_sources.append((split, p, "dataset_consensus_only"))

    for split, meta_path, source in metadata_sources:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to read {meta_path}: {e}", file=sys.stderr)
            continue
        for entry in entries:
            patch_path_raw = entry["patch_path"]
            filename = patch_path_raw.replace("\\", "/").split("/")[-1]
            m = PATCH_RC_RE.search(filename)
            if not m:
                continue
            leaf_stem = filename[: m.start()]
            row = entry["row"]
            col = entry["col"]
            class_name = entry["class_name"]
            norm_patch = f"{split}/needs_annotation/{class_name}/{filename}"

            key = (split, class_name, leaf_stem)
            if key not in leaf_index:
                source_url = f"raw/segmented/{class_name}/{leaf_stem}.jpg"
                leaf_index[key] = {
                    "source_image_url": source_url,
                    "patches": [],
                    "grid_rows": 8,
                    "grid_cols": 8,
                }
            leaf_data = leaf_index[key]
            leaf_data["patches"].append({
                "patch_path": norm_patch,
                "row": row,
                "col": col,
            })
            leaf_data["grid_rows"] = max(leaf_data["grid_rows"], row + 1)
            leaf_data["grid_cols"] = max(leaf_data["grid_cols"], col + 1)
        del entries

    # Sort patches by row, col
    for leaf_data in leaf_index.values():
        leaf_data["patches"].sort(key=lambda p: (p["row"], p["col"]))

    elapsed = time.time() - start
    print(f"Leaf index built: {len(leaf_index)} leaves in {elapsed:.1f}s "
          f"(sources: {[src for _, _, src in metadata_sources]})", file=sys.stderr)


# --- Pydantic models ---

class SetupRequest(BaseModel):
    name: str


class AnnotateRequest(BaseModel):
    patch_path: str
    label: str  # "healthy" or "unhealthy"


class SkipRequest(BaseModel):
    patch_path: str


# Active learning mode (Phase 2/3 of active_learning_round.py)
AL_ASSIGNMENTS_DIR = PREDICTIONS_DIR
AL_DEFAULT_ROUND = 2
AL_MAX_HISTORY = 20


class ALSetupRequest(BaseModel):
    name: str
    round: int = AL_DEFAULT_ROUND


class ALAnnotateRequest(BaseModel):
    patch_path: str
    task_type: str             # "verify_pseudo" or "label_hitl"
    label: str = ""            # for label_hitl: "healthy" or "unhealthy"
    is_correct: bool | None = None  # for verify_pseudo


# --- Helper functions ---

def load_assignments() -> dict:
    if not ASSIGNMENTS_PATH.exists():
        raise HTTPException(400, "assignments.json not found. Run assignments_generator.py first.")
    with open(ASSIGNMENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config() -> dict | None:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_config(name: str, mode: str = "normal"):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"name": name, "mode": mode}, f, indent=2)


def update_config(updates: dict):
    """Merge updates into annotator_config.json (preserves existing fields)."""
    config = load_config()
    if not config:
        return
    config.update(updates)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def init_session(name: str) -> dict:
    """Initialize or restore a session for the given annotator."""
    assignments = load_assignments()
    if name not in assignments:
        raise HTTPException(400, f"Annotator '{name}' not found in assignments.json")

    patch_list = []
    for folder_rel in assignments[name]:
        folder_path = DATASET_DIR / folder_rel
        parts = folder_rel.split("/")
        split = parts[0]  # "train" or "test"
        class_name = parts[-1]  # last segment

        if folder_path.exists():
            for img_file in sorted(folder_path.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    patch_list.append({
                        "patch_path": f"{folder_rel}/{img_file.name}",
                        "class_name": class_name,
                        "split": split,
                    })

    # Load already-annotated set for resume
    csv_path = ANNOTATIONS_DIR / f"annotations_{name}.csv"
    annotated_set = set()
    skipped_set = set()
    label_map: dict[str, str] = {}  # patch_path -> label
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pp = row["patch_path"]
                    if row.get("is_skipped") == "True":
                        skipped_set.add(pp)
                        label_map[pp] = "skipped"
                    else:
                        annotated_set.add(pp)
                        if row.get("label"):
                            label_map[pp] = row["label"]
        except Exception:
            pass  # Corrupted CSV, start fresh

    # Filter out annotated (but keep skipped for re-annotation)
    remaining = [p for p in patch_list if p["patch_path"] not in annotated_set]

    session = {
        "patch_list": remaining,
        "path_to_index": {p["patch_path"]: i for i, p in enumerate(remaining)},
        "current_index": 0,
        "history": [],
        "annotated_set": annotated_set,
        "skipped_set": skipped_set,
        "label_map": label_map,
        "csv_path": csv_path,
        "total_original": len(patch_list),
        "annotated_count": len(annotated_set),
        "annotation_counter": 0,
    }
    sessions[name] = session
    return session


def init_review_session(name: str) -> dict:
    """Initialize a session for reviewing needs_discussion patches."""
    if not CONSENSUS_REVIEW_CSV.exists():
        raise HTTPException(400, "consensus_review_master.csv not found.")

    assignments = load_assignments()
    if name not in assignments:
        raise HTTPException(400, f"Annotator '{name}' not found in assignments.json")

    patch_list = []
    with open(CONSENSUS_REVIEW_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("needs_discussion") != "True":
                continue
            pp = row["patch_path"]
            parts = pp.split("/")
            patch_list.append({
                "patch_path": pp,
                "class_name": parts[2],
                "split": parts[0],
            })

    review_csv_path = ANNOTATIONS_DIR / f"annotations_{name}_review.csv"
    annotated_set = set()
    skipped_set = set()
    label_map: dict[str, str] = {}
    if review_csv_path.exists():
        try:
            with open(review_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pp = row["patch_path"]
                    if row.get("is_skipped") == "True":
                        skipped_set.add(pp)
                        label_map[pp] = "skipped"
                    else:
                        annotated_set.add(pp)
                        if row.get("label"):
                            label_map[pp] = row["label"]
        except Exception:
            pass

    remaining = [p for p in patch_list if p["patch_path"] not in annotated_set]

    session = {
        "patch_list": remaining,
        "path_to_index": {p["patch_path"]: i for i, p in enumerate(remaining)},
        "current_index": 0,
        "history": [],
        "annotated_set": annotated_set,
        "skipped_set": skipped_set,
        "label_map": label_map,
        "csv_path": review_csv_path,
        "total_original": len(patch_list),
        "annotated_count": len(annotated_set),
        "annotation_counter": 0,
    }
    sessions[name] = session
    return session


def init_al_session(name: str, round_n: int) -> dict:
    """Initialize a session for active learning mode (verify_pseudo + label_hitl).

    Reads predictions/al_assignments_round{round_n}.json and builds the
    patch list for this annotator. Resumes from annotations already
    written to annotations/annotations_{name}_al_round{round_n}.csv.
    """
    al_path = AL_ASSIGNMENTS_DIR / f"al_assignments_round{round_n}.json"
    if not al_path.exists():
        raise HTTPException(
            400,
            f"Assignment file not found: {al_path}\n"
            f"Run: python active_learning_round.py --phase 2 generate --round {round_n}"
        )
    with open(al_path, "r", encoding="utf-8") as f:
        al = json.load(f)
    if name not in al.get("patches", {}):
        raise HTTPException(
            400,
            f"Annotator '{name}' not in assignment file. "
            f"Available: {list(al.get('patches', {}).keys())}"
        )
    patches = al["patches"][name]

    csv_path = ANNOTATIONS_DIR / f"annotations_{name}_al_round{round_n}.csv"
    annotated = set()
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("is_skipped") != "True":
                        annotated.add(row["patch_path"])
        except Exception:
            pass

    remaining = [p for p in patches if p["patch_path"] not in annotated]

    # Per-task-type totals (from original full list, not remaining)
    task_type_total = {"verify_pseudo": 0, "label_hitl": 0}
    for p in patches:
        tt = p.get("task_type")
        if tt in task_type_total:
            task_type_total[tt] += 1

    # Per-class totals (from original full list)
    class_totals: dict[str, int] = {}
    for p in patches:
        cn = p.get("class_name", "")
        class_totals[cn] = class_totals.get(cn, 0) + 1

    # Per-task-type already-annotated counts (rebuilt from CSV on resume)
    task_type_annotated = {"verify_pseudo": 0, "label_hitl": 0}
    class_annotated: dict[str, int] = {}
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("is_skipped") == "True":
                        continue
                    tt = row.get("task_type")
                    if tt in task_type_annotated:
                        task_type_annotated[tt] += 1
                    cn = row.get("class_name", "")
                    class_annotated[cn] = class_annotated.get(cn, 0) + 1
        except Exception:
            pass

    sessions[name] = {
        "mode": "al",
        "round": round_n,
        "patch_list": remaining,
        "path_to_index": {p["patch_path"]: i for i, p in enumerate(remaining)},
        "current_index": 0,
        "history": [],
        "annotated_set": annotated,
        "skipped_set": set(),
        "label_map": {},
        "csv_path": csv_path,
        "total_original": len(patches),
        "annotated_count": len(annotated),
        "task_type_total": task_type_total,
        "task_type_annotated": task_type_annotated,
        "class_totals": class_totals,
        "class_annotated": class_annotated,
        "annotation_counter": 0,
    }
    return sessions[name]


def append_al_csv(csv_path: Path, row: dict) -> None:
    """Append a single row to an AL annotator CSV file."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patch_path", "class_name", "split", "task_type",
            "label", "is_correct", "annotator", "timestamp", "is_skipped",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def rewrite_al_csv_without_last(csv_path: Path) -> None:
    """Remove the last row from an AL annotator CSV (for undo)."""
    if not csv_path.exists():
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    rows.pop()  # Remove last

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patch_path", "class_name", "split", "task_type",
            "label", "is_correct", "annotator", "timestamp", "is_skipped",
        ])
        writer.writeheader()
        writer.writerows(rows)


def has_disputed_patches() -> bool:
    """Check if consensus review CSV exists with any needs_discussion patches."""
    if not CONSENSUS_REVIEW_CSV.exists():
        return False
    try:
        with open(CONSENSUS_REVIEW_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("needs_discussion") == "True":
                    return True
    except Exception:
        pass
    return False


def append_csv(csv_path: Path, row: dict):
    """Append a single row to the CSV file."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patch_path", "class_name", "split", "label",
            "annotator", "timestamp", "is_skipped"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def rewrite_csv_without_last(csv_path: Path):
    """Remove the last row from CSV (for undo)."""
    if not csv_path.exists():
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    rows.pop()  # Remove last

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patch_path", "class_name", "split", "label",
            "annotator", "timestamp", "is_skipped"
        ])
        writer.writeheader()
        writer.writerows(rows)


def backup_csv(csv_path: Path):
    """Create a backup of the CSV file."""
    if csv_path.exists():
        backup_path = csv_path.with_suffix(".csv.bak")
        shutil.copy2(csv_path, backup_path)


# --- Startup ---

@app.on_event("startup")
async def startup_event():
    print("Leaf index will be built on first request", file=sys.stderr)


# --- Page routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    config = load_config()
    annotator_name = config["name"] if config else ""
    return templates.TemplateResponse(request=request, name="index.html", context={
        "annotator_name": annotator_name,
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


# --- API routes ---

@app.get("/api/status")
async def api_status():
    config = load_config()
    if not config:
        return {"setup": False, "name": None, "mode": None, "has_disputed": has_disputed_patches()}
    name = config["name"]
    mode = config.get("mode", "normal")
    disputed = has_disputed_patches()

    if name not in sessions:
        try:
            if mode == "al":
                init_al_session(name, config.get("round", AL_DEFAULT_ROUND))
            elif mode == "review":
                init_review_session(name)
            else:
                init_session(name)
        except Exception:
            if mode == "al":
                # al mode failure -> return to normal
                save_config(name, mode="normal")
                return {"setup": False, "name": None, "mode": "al",
                        "has_disputed": disputed,
                        "error": "AL assignment file not found. "
                                 "Run active_learning_round.py --phase 2 generate first."}
            if mode == "review":
                save_config(name, mode="normal")
                mode = "normal"
                try:
                    init_session(name)
                except Exception:
                    return {"setup": False, "name": None, "mode": "normal", "has_disputed": disputed}
            else:
                return {"setup": False, "name": None, "mode": "normal", "has_disputed": disputed}

    session = sessions[name]
    resp = {
        "setup": True,
        "name": name,
        "mode": mode,
        "has_disputed": disputed,
        "total": session["total_original"],
        "annotated": session["annotated_count"],
    }
    if mode == "al":
        resp["round"] = session.get("round", AL_DEFAULT_ROUND)
        # Phase 4.1: first-time intro flag (per annotator, per round)
        seen_rounds = config.get("al_intro_seen", [])
        resp["seen_intro"] = resp["round"] in seen_rounds
    return resp


@app.post("/api/setup")
async def api_setup(req: SetupRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")

    # Validate against assignments
    assignments = load_assignments()
    if name not in assignments:
        valid_names = list(assignments.keys())
        raise HTTPException(400, f"Annotator '{name}' not found. Valid names: {valid_names}")

    session = init_session(name)
    save_config(name, mode="normal")
    return {
        "setup": True,
        "name": name,
        "total": session["total_original"],
        "annotated": session["annotated_count"],
    }


@app.post("/api/setup-review")
async def api_setup_review(req: SetupRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")

    assignments = load_assignments()
    if name not in assignments:
        valid_names = list(assignments.keys())
        raise HTTPException(400, f"Annotator '{name}' not found. Valid names: {valid_names}")

    session = init_review_session(name)
    save_config(name, mode="review")
    return {
        "setup": True,
        "mode": "review",
        "name": name,
        "total": session["total_original"],
        "annotated": session["annotated_count"],
    }


@app.post("/api/setup-normal")
async def api_setup_normal(req: SetupRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")

    assignments = load_assignments()
    if name not in assignments:
        valid_names = list(assignments.keys())
        raise HTTPException(400, f"Annotator '{name}' not found. Valid names: {valid_names}")

    session = init_session(name)
    save_config(name, mode="normal")
    return {
        "setup": True,
        "mode": "normal",
        "name": name,
        "total": session["total_original"],
        "annotated": session["annotated_count"],
    }


@app.post("/api/setup-al")
async def api_setup_al(req: ALSetupRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")

    session = init_al_session(name, req.round)
    save_config(name, mode="al")
    # Also save the round in config for resume
    config_path = CONFIG_PATH
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["round"] = req.round
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    return {
        "setup": True,
        "mode": "al",
        "round": req.round,
        "name": name,
        "total": session["total_original"],
        "annotated": session["annotated_count"],
    }


@app.post("/api/mark-intro-seen")
async def api_mark_intro_seen():
    """Mark that the annotator has seen the AL intro modal for the current round."""
    config = load_config()
    if not config or config.get("mode") != "al":
        raise HTTPException(400, "AL mode not active")
    round_n = config.get("round", AL_DEFAULT_ROUND)
    seen = config.get("al_intro_seen", [])
    if round_n not in seen:
        seen.append(round_n)
        update_config({"al_intro_seen": seen})
    return {"ok": True, "round": round_n}


@app.get("/api/history-al")
async def api_history_al():
    """Return the last 20 AL annotations for the current annotator (from CSV)."""
    config = load_config()
    if not config or config.get("mode") != "al":
        raise HTTPException(400, "AL mode not active")
    name = config["name"]
    round_n = config.get("round", AL_DEFAULT_ROUND)
    csv_path = ANNOTATIONS_DIR / f"annotations_{name}_al_round{round_n}.csv"
    if not csv_path.exists():
        return {"history": []}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return {"history": []}
    # Return last 20 rows, newest first
    recent = rows[-20:][::-1]
    out = []
    for r in recent:
        task_type = r.get("task_type", "")
        if task_type == "verify_pseudo":
            label = "Correct" if r.get("is_correct", "").lower() == "true" else "Wrong"
        elif task_type == "label_hitl":
            label = r.get("label", "") or "—"
        else:
            label = r.get("label", "") or "—"
        out.append({
            "patch_path": r.get("patch_path", ""),
            "class_name": r.get("class_name", ""),
            "split": r.get("split", ""),
            "task_type": task_type,
            "label": label,
            "is_skipped": r.get("is_skipped", "False"),
            "timestamp": r.get("timestamp", ""),
        })
    return {"history": out}


@app.get("/api/patch/current-al")
async def api_patch_current_al():
    config = load_config()
    if not config or config.get("mode") != "al":
        raise HTTPException(400, "AL mode not active")
    name = config["name"]
    if name not in sessions:
        init_al_session(name, config.get("round", AL_DEFAULT_ROUND))
    session = sessions[name]

    if session["current_index"] >= len(session["patch_list"]):
        return {"done": True, "message": "All AL patches processed!"}

    patch = session["patch_list"][session["current_index"]]
    ttc = session.get("task_type_total", {"verify_pseudo": 0, "label_hitl": 0})
    tta = session.get("task_type_annotated", {"verify_pseudo": 0, "label_hitl": 0})
    cn = patch.get("class_name", "")
    class_totals = session.get("class_totals", {})
    class_annotated = session.get("class_annotated", {})
    return {
        "done": False,
        "patch_path": patch["patch_path"],
        "class_name": patch["class_name"],
        "split": patch["split"],
        "task_type": patch["task_type"],
        "model_prediction": patch.get("model_prediction"),
        "model_confidence": patch.get("model_confidence"),
        "model_margin": patch.get("model_margin"),
        "cluster_id": patch.get("cluster_id"),
        "cluster_margin": patch.get("margin"),
        "index": session["current_index"],
        "total": len(session["patch_list"]),
        "annotated_count": session["annotated_count"],
        "verify_total": ttc.get("verify_pseudo", 0),
        "verify_done": tta.get("verify_pseudo", 0),
        "hitl_total": ttc.get("label_hitl", 0),
        "hitl_done": tta.get("label_hitl", 0),
        "class_total": class_totals.get(cn, 0),
        "class_done": class_annotated.get(cn, 0),
    }


@app.post("/api/annotate-al")
async def api_annotate_al(req: ALAnnotateRequest):
    config = load_config()
    if not config or config.get("mode") != "al":
        raise HTTPException(400, "AL mode not active")
    name = config["name"]
    session = sessions.get(name)
    if not session:
        raise HTTPException(400, "Session not found")
    if session["current_index"] >= len(session["patch_list"]):
        raise HTTPException(400, "No more AL patches")

    patch = session["patch_list"][session["current_index"]]
    if patch["patch_path"] != req.patch_path:
        raise HTTPException(400, "Patch path mismatch")

    if req.task_type == "verify_pseudo":
        if req.is_correct is None:
            raise HTTPException(400, "verify_pseudo requires is_correct")
    elif req.task_type == "label_hitl":
        if req.label not in ("healthy", "unhealthy"):
            raise HTTPException(400, "label_hitl requires label in {healthy, unhealthy}")
    else:
        raise HTTPException(400, f"Unknown task_type: {req.task_type}")

    now = datetime.now(timezone.utc).isoformat()
    row = {
        "patch_path": patch["patch_path"],
        "class_name": patch["class_name"],
        "split": patch["split"],
        "task_type": req.task_type,
        "label": req.label,
        "is_correct": str(req.is_correct) if req.is_correct is not None else "",
        "annotator": name,
        "timestamp": now,
        "is_skipped": "False",
    }
    append_al_csv(session["csv_path"], row)

    session["history"].append({"patch": patch, "index": session["current_index"], "row": row})
    if len(session["history"]) > AL_MAX_HISTORY:
        session["history"].pop(0)

    session["annotated_set"].add(patch["patch_path"])
    session["annotated_count"] += 1
    session["annotation_counter"] += 1
    session["current_index"] += 1

    tt = patch.get("task_type")
    tta = session.get("task_type_annotated", {})
    if tt in tta:
        tta[tt] += 1
    cn = patch.get("class_name", "")
    ca = session.get("class_annotated", {})
    ca[cn] = ca.get(cn, 0) + 1

    if session["annotation_counter"] >= BACKUP_INTERVAL:
        backup_csv(session["csv_path"])
        session["annotation_counter"] = 0

    if session["current_index"] >= len(session["patch_list"]):
        return {"done": True, "message": "All AL patches processed!"}

    next_patch = session["patch_list"][session["current_index"]]
    ttc = session.get("task_type_total", {"verify_pseudo": 0, "label_hitl": 0})
    tta = session.get("task_type_annotated", {"verify_pseudo": 0, "label_hitl": 0})
    ct = session.get("class_totals", {})
    ca = session.get("class_annotated", {})
    next_cn = next_patch.get("class_name", "")
    return {
        "done": False,
        "patch_path": next_patch["patch_path"],
        "class_name": next_patch["class_name"],
        "split": next_patch["split"],
        "task_type": next_patch["task_type"],
        "model_prediction": next_patch.get("model_prediction"),
        "model_confidence": next_patch.get("model_confidence"),
        "model_margin": next_patch.get("model_margin"),
        "cluster_id": next_patch.get("cluster_id"),
        "cluster_margin": next_patch.get("margin"),
        "index": session["current_index"],
        "total": len(session["patch_list"]),
        "annotated_count": session["annotated_count"],
        "verify_total": ttc.get("verify_pseudo", 0),
        "verify_done": tta.get("verify_pseudo", 0),
        "hitl_total": ttc.get("label_hitl", 0),
        "hitl_done": tta.get("label_hitl", 0),
        "class_total": ct.get(next_cn, 0),
        "class_done": ca.get(next_cn, 0),
    }


@app.post("/api/skip-al")
async def api_skip_al(req: SkipRequest):
    config = load_config()
    if not config or config.get("mode") != "al":
        raise HTTPException(400, "AL mode not active")
    name = config["name"]
    session = sessions.get(name)
    if not session:
        raise HTTPException(400, "Session not found")
    if session["current_index"] >= len(session["patch_list"]):
        raise HTTPException(400, "No more AL patches")

    patch = session["patch_list"][session["current_index"]]
    if patch["patch_path"] != req.patch_path:
        raise HTTPException(400, "Patch path mismatch")

    now = datetime.now(timezone.utc).isoformat()
    row = {
        "patch_path": patch["patch_path"],
        "class_name": patch["class_name"],
        "split": patch["split"],
        "task_type": patch["task_type"],
        "label": "",
        "is_correct": "",
        "annotator": name,
        "timestamp": now,
        "is_skipped": "True",
    }
    append_al_csv(session["csv_path"], row)

    session["skipped_set"].add(patch["patch_path"])
    session["current_index"] += 1

    if session["current_index"] >= len(session["patch_list"]):
        return {"done": True, "message": "All AL patches processed!"}

    next_patch = session["patch_list"][session["current_index"]]
    ttc = session.get("task_type_total", {"verify_pseudo": 0, "label_hitl": 0})
    tta = session.get("task_type_annotated", {"verify_pseudo": 0, "label_hitl": 0})
    ct = session.get("class_totals", {})
    ca = session.get("class_annotated", {})
    next_cn = next_patch.get("class_name", "")
    return {
        "done": False,
        "patch_path": next_patch["patch_path"],
        "class_name": next_patch["class_name"],
        "split": next_patch["split"],
        "task_type": next_patch["task_type"],
        "model_prediction": next_patch.get("model_prediction"),
        "model_confidence": next_patch.get("model_confidence"),
        "model_margin": next_patch.get("model_margin"),
        "cluster_id": next_patch.get("cluster_id"),
        "cluster_margin": next_patch.get("margin"),
        "index": session["current_index"],
        "total": len(session["patch_list"]),
        "annotated_count": session["annotated_count"],
        "verify_total": ttc.get("verify_pseudo", 0),
        "verify_done": tta.get("verify_pseudo", 0),
        "hitl_total": ttc.get("label_hitl", 0),
        "hitl_done": tta.get("label_hitl", 0),
        "class_total": ct.get(next_cn, 0),
        "class_done": ca.get(next_cn, 0),
    }


@app.get("/api/patch/current")
async def api_patch_current():
    config = load_config()
    if not config:
        raise HTTPException(400, "Setup not complete")
    name = config["name"]
    if name not in sessions:
        init_session(name)
    session = sessions[name]

    if session["current_index"] >= len(session["patch_list"]):
        return {"done": True, "message": "All patches annotated!"}

    patch = session["patch_list"][session["current_index"]]
    return {
        "done": False,
        "patch_path": patch["patch_path"],
        "class_name": patch["class_name"],
        "split": patch["split"],
        "index": session["current_index"],
        "total": len(session["patch_list"]),
        "annotated_count": session["annotated_count"],
    }


@app.post("/api/annotate")
async def api_annotate(req: AnnotateRequest):
    config = load_config()
    if not config:
        raise HTTPException(400, "Setup not complete")
    name = config["name"]
    session = sessions.get(name)
    if not session:
        raise HTTPException(400, "Session not found")

    if session["current_index"] >= len(session["patch_list"]):
        raise HTTPException(400, "No more patches to annotate")

    patch = session["patch_list"][session["current_index"]]
    if patch["patch_path"] != req.patch_path:
        raise HTTPException(400, "Patch path mismatch")

    # Save annotation
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "patch_path": patch["patch_path"],
        "class_name": patch["class_name"],
        "split": patch["split"],
        "label": req.label,
        "annotator": name,
        "timestamp": now,
        "is_skipped": "False",
    }
    append_csv(session["csv_path"], row)

    # Update history for undo
    session["history"].append({
        "patch": patch,
        "index": session["current_index"],
        "row": row,
    })
    if len(session["history"]) > MAX_HISTORY:
        session["history"].pop(0)

    # Update state
    session["annotated_set"].add(patch["patch_path"])
    session["label_map"][patch["patch_path"]] = req.label
    session["annotated_count"] += 1
    session["annotation_counter"] += 1
    session["current_index"] += 1

    # Backup every N annotations
    if session["annotation_counter"] >= BACKUP_INTERVAL:
        backup_csv(session["csv_path"])
        session["annotation_counter"] = 0

    # Return next patch info
    if session["current_index"] >= len(session["patch_list"]):
        return {"done": True, "message": "All patches annotated!"}

    next_patch = session["patch_list"][session["current_index"]]
    return {
        "done": False,
        "patch_path": next_patch["patch_path"],
        "class_name": next_patch["class_name"],
        "split": next_patch["split"],
        "index": session["current_index"],
        "total": len(session["patch_list"]),
        "annotated_count": session["annotated_count"],
    }


@app.post("/api/skip")
async def api_skip(req: SkipRequest):
    config = load_config()
    if not config:
        raise HTTPException(400, "Setup not complete")
    name = config["name"]
    session = sessions.get(name)
    if not session:
        raise HTTPException(400, "Session not found")

    if session["current_index"] >= len(session["patch_list"]):
        raise HTTPException(400, "No more patches")

    patch = session["patch_list"][session["current_index"]]
    if patch["patch_path"] != req.patch_path:
        raise HTTPException(400, "Patch path mismatch")

    # Save skip entry
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "patch_path": patch["patch_path"],
        "class_name": patch["class_name"],
        "split": patch["split"],
        "label": "",
        "annotator": name,
        "timestamp": now,
        "is_skipped": "True",
    }
    append_csv(session["csv_path"], row)

    session["skipped_set"].add(patch["patch_path"])
    session["label_map"][patch["patch_path"]] = "skipped"
    session["current_index"] += 1

    # Return next patch
    if session["current_index"] >= len(session["patch_list"]):
        return {"done": True, "message": "All patches processed!"}

    next_patch = session["patch_list"][session["current_index"]]
    return {
        "done": False,
        "patch_path": next_patch["patch_path"],
        "class_name": next_patch["class_name"],
        "split": next_patch["split"],
        "index": session["current_index"],
        "total": len(session["patch_list"]),
        "annotated_count": session["annotated_count"],
    }


@app.post("/api/undo")
async def api_undo():
    config = load_config()
    if not config:
        raise HTTPException(400, "Setup not complete")
    name = config["name"]
    session = sessions.get(name)
    if not session:
        raise HTTPException(400, "Session not found")

    if not session["history"]:
        raise HTTPException(400, "Nothing to undo")

    # Pop last annotation from history
    last = session["history"].pop()

    # Rewrite CSV without last row (mode-aware to preserve fieldnames)
    if session.get("mode") == "al":
        rewrite_al_csv_without_last(session["csv_path"])
    else:
        rewrite_csv_without_last(session["csv_path"])

    # Restore state
    session["current_index"] = last["index"]
    session["annotated_set"].discard(last["patch"]["patch_path"])
    session["label_map"].pop(last["patch"]["patch_path"], None)
    session["annotated_count"] = max(0, session["annotated_count"] - 1)

    return {
        "patch_path": last["patch"]["patch_path"],
        "class_name": last["patch"]["class_name"],
        "split": last["patch"]["split"],
        "index": session["current_index"],
        "total": len(session["patch_list"]),
        "annotated_count": session["annotated_count"],
    }


@app.get("/api/history")
async def api_history():
    config = load_config()
    if not config:
        return {"history": []}
    name = config["name"]
    session = sessions.get(name)
    if not session:
        return {"history": []}

    # Return last 5 annotations
    recent = session["history"][-5:]
    return {
        "history": [
            {
                "patch_path": h["patch"]["patch_path"],
                "class_name": h["patch"]["class_name"],
                "label": h["row"]["label"],
            }
            for h in recent
        ]
    }


@app.get("/api/progress")
async def api_progress():
    config = load_config()
    if not config:
        raise HTTPException(400, "Setup not complete")
    name = config["name"]
    session = sessions.get(name)
    if not session:
        raise HTTPException(400, "Session not found")

    total = len(session["patch_list"])
    current = session["current_index"]
    remaining = total - current
    percent = (current / total * 100) if total > 0 else 0

    return {
        "total": total,
        "current": current,
        "annotated": session["annotated_count"],
        "remaining": remaining,
        "percent": round(percent, 1),
    }


@app.get("/api/dashboard-data")
async def api_dashboard_data():
    """Read all annotations CSV files and return dashboard statistics."""
    annotator_stats = []
    class_stats: dict[str, dict] = {}
    total_annotated = 0
    total_skipped = 0
    label_counts = {"healthy": 0, "unhealthy": 0}

    # Load assignments to get total per annotator
    assignments = {}
    if ASSIGNMENTS_PATH.exists():
        with open(ASSIGNMENTS_PATH, "r", encoding="utf-8") as f:
            assignments = json.load(f)

    # Count total patches per annotator from assignments
    annotator_totals = {}
    if assignments:
        for ann_name, folders in assignments.items():
            count = 0
            for folder_rel in folders:
                folder_path = DATASET_DIR / folder_rel
                if folder_path.exists():
                    count += sum(1 for f in folder_path.iterdir()
                                 if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
            annotator_totals[ann_name] = count

    # Read all CSV files
    for csv_file in ANNOTATIONS_DIR.glob("annotations_*.csv"):
        if csv_file.suffix == ".bak":
            continue
        annotator_name = csv_file.stem.replace("annotations_", "")
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            continue

        done = sum(1 for r in rows if r.get("is_skipped") != "True")
        skipped = sum(1 for r in rows if r.get("is_skipped") == "True")
        total = annotator_totals.get(annotator_name, 0)

        annotator_stats.append({
            "name": annotator_name,
            "assigned": total,
            "done": done,
            "skipped": skipped,
            "percent": round(done / total * 100, 1) if total > 0 else 0,
        })

        total_annotated += done
        total_skipped += skipped

        for row in rows:
            if row.get("is_skipped") != "True":
                label = row.get("label", "")
                if label in label_counts:
                    label_counts[label] += 1
                class_name = row.get("class_name", "unknown")
                if class_name not in class_stats:
                    class_stats[class_name] = {"total": 0, "done": 0}
                class_stats[class_name]["done"] += 1

    # Count total patches per class
    for split in ["train", "test"]:
        needs_annot = DATASET_DIR / split / "needs_annotation"
        if needs_annot.exists():
            for class_dir in needs_annot.iterdir():
                if class_dir.is_dir():
                    cn = class_dir.name
                    if cn not in class_stats:
                        class_stats[cn] = {"total": 0, "done": 0}
                    count = sum(1 for f in class_dir.iterdir()
                                if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
                    class_stats[cn]["total"] += count

    # Build class list with percentages
    class_list = []
    for cn, stats in sorted(class_stats.items()):
        pct = round(stats["done"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
        class_list.append({
            "name": cn,
            "total": stats["total"],
            "done": stats["done"],
            "percent": pct,
        })

    # Overall total
    grand_total = sum(c["total"] for c in class_list)
    grand_done = sum(c["done"] for c in class_list)
    grand_pct = round(grand_done / grand_total * 100, 1) if grand_total > 0 else 0

    # ── Active Learning rounds (Phase 3.2) ──
    al_rounds = []
    if PREDICTIONS_DIR.exists():
        for al_path in sorted(PREDICTIONS_DIR.glob("al_assignments_round*.json")):
            try:
                with open(al_path, "r", encoding="utf-8") as f:
                    al = json.load(f)
            except Exception:
                continue
            round_n = al.get("round")
            if round_n is None:
                continue
            al_ann_stats = []
            for ann_name in al.get("annotators", []):
                ann_csv = ANNOTATIONS_DIR / f"annotations_{ann_name}_al_round{round_n}.csv"
                assigned = len(al["patches"].get(ann_name, []))
                done_count = 0
                skipped_count = 0
                if ann_csv.exists():
                    try:
                        with open(ann_csv, "r", encoding="utf-8") as f:
                            for row in csv.DictReader(f):
                                if row.get("is_skipped") == "True":
                                    skipped_count += 1
                                else:
                                    done_count += 1
                    except Exception:
                        pass
                pct = round(done_count / assigned * 100, 1) if assigned > 0 else 0
                al_ann_stats.append({
                    "name": ann_name,
                    "assigned": assigned,
                    "done": done_count,
                    "skipped": skipped_count,
                    "percent": pct,
                })
            al_rounds.append({
                "round": round_n,
                "annotators": al_ann_stats,
            })

    return {
        "annotators": annotator_stats,
        "classes": class_list,
        "overall": {
            "total": grand_total,
            "done": grand_done,
            "skipped": total_skipped,
            "percent": grand_pct,
        },
        "labels": label_counts,
        "al_rounds": al_rounds,
    }


@app.get("/api/leaf-context/{split}/{class_name}/{leaf_stem}")
async def api_leaf_context(split: str, class_name: str, leaf_stem: str):
    build_leaf_index()  # Lazy init

    key = (split, class_name, leaf_stem)
    if key not in leaf_index:
        # Graceful fallback: leaf not in any metadata, but the raw image
        # may still exist. Return a partial response with image only
        # (no grid data) so the user can still see the full leaf.
        source_rel = f"raw/segmented/{class_name}/{leaf_stem}.jpg"
        raw_image = DATASET_FILTERED_DIR / source_rel
        if raw_image.exists():
            return {
                "source_image_url": f"/raw-image/{source_rel}",
                "patches": [],
                "grid_rows": 8,
                "grid_cols": 8,
                "img_width": 8 * PATCH_SIZE,
                "img_height": 8 * PATCH_SIZE,
                "annotated_count": 0,
                "total_patches": 0,
                "partial": True,
            }
        raise HTTPException(404, f"Leaf not found: {leaf_stem}")

    leaf_data = leaf_index[key]

    # Get current annotator session for label lookup
    config = load_config()
    current_patch_path = None
    label_map: dict[str, str] = {}
    if config:
        name = config["name"]
        session = sessions.get(name)
        if session:
            if session["current_index"] < len(session["patch_list"]):
                current_patch_path = session["patch_list"][session["current_index"]]["patch_path"]
            label_map = session["label_map"]

    patches_out = []
    annotated_count = 0
    for p in leaf_data["patches"]:
        label = label_map.get(p["patch_path"])
        if label:
            annotated_count += 1
        patches_out.append({
            "patch_path": p["patch_path"],
            "row": p["row"],
            "col": p["col"],
            "label": label,
            "is_current": p["patch_path"] == current_patch_path,
        })

    source_rel = leaf_data["source_image_url"]

    # Compute image dimensions from grid size (cached after first read)
    if "img_width" not in leaf_data:
        leaf_data["img_width"] = leaf_data["grid_cols"] * PATCH_SIZE
        leaf_data["img_height"] = leaf_data["grid_rows"] * PATCH_SIZE

    return {
        "source_image_url": f"/raw-image/{source_rel}",
        "patches": patches_out,
        "grid_rows": leaf_data["grid_rows"],
        "grid_cols": leaf_data["grid_cols"],
        "img_width": leaf_data["img_width"],
        "img_height": leaf_data["img_height"],
        "annotated_count": annotated_count,
        "total_patches": len(patches_out),
    }


@app.get("/api/jump-to-patch")
async def api_jump_to_patch(patch_path: str):
    config = load_config()
    if not config:
        raise HTTPException(400, "Setup not complete")
    session = sessions.get(config["name"])
    if not session:
        raise HTTPException(400, "Session not found")

    idx = session["path_to_index"].get(patch_path)
    if idx is None:
        raise HTTPException(404, f"Patch not found in session: {patch_path}")

    session["current_index"] = idx
    p = session["patch_list"][idx]
    return {
        "done": False,
        "patch_path": p["patch_path"],
        "class_name": p["class_name"],
        "split": p["split"],
        "index": idx,
        "total": len(session["patch_list"]),
        "annotated_count": session["annotated_count"],
    }


def _serve_image_from(base_dir: Path, path: str):
    file_path = base_dir / path
    if not file_path.exists() or not file_path.is_file():
        return Response(content=PLACEHOLDER_SVG, media_type="image/svg+xml")
    return FileResponse(file_path)


@app.get("/image/{path:path}")
async def serve_image(path: str):
    return _serve_image_from(DATASET_DIR, path)


@app.get("/raw-image/{path:path}")
async def serve_raw_image(path: str):
    return _serve_image_from(DATASET_FILTERED_DIR, path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
