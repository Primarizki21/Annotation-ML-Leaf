#!/usr/bin/env python3
"""
extract_consensus.py

Extracts a 5% consensus subset of unique source leaf images per diseased class
from `dataset_patches` to `dataset_consensus_only` to ensure compatibility 
with the existing FastAPI leaf annotation application.
"""

import os
import json
import math
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "dataset_patches"
OUTPUT_DIR = BASE_DIR / "dataset_consensus_only"
RANDOM_SEED = 42
SAMPLE_SIZE = 0.05

# Team members to assign the folders
TEAM_MEMBERS = ["Oki", "Muna", "Diaz", "Sarah", "Cinta"]
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  CONSENSUS DATASET EXTRACTOR")
    print("=" * 60)

    # 1. Verification of inputs
    meta_path = INPUT_DIR / "metadata_train.json"
    if not meta_path.exists():
        print(f"[ERROR] metadata_train.json not found at: {meta_path}")
        print("Please ensure the patch splitting has been executed first.")
        return

    print(f"Loading metadata from: {meta_path.name}...")
    with open(meta_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Loaded {len(entries):,} total patch entries.")

    # 2. Filter for Needs Annotation (Diseased)
    needs_annot_entries = [
        entry for entry in entries if entry.get("label") == "needs_annotation"
    ]
    print(f"Filtered {len(needs_annot_entries):,} entries requiring annotation.")

    # 3. Group by class_name and source_image
    # Structure: class_name -> source_image -> list of patch entries
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in needs_annot_entries:
        class_name = entry["class_name"]
        source_image = entry["source_image"]
        grouped[class_name][source_image].append(entry)

    print(f"Detected {len(grouped)} unique classes requiring annotation.")

    # 4. Perform deterministic random sampling of unique source images
    random.seed(RANDOM_SEED)
    sampled_entries = []
    
    print("\nSampling summary per class:")
    print("-" * 65)
    print(f"{'Class Name':<45} | {'Total Img':<9} | {'Sampled Img':<11}")
    print("-" * 65)

    for class_name, img_dict in sorted(grouped.items()):
        # Sort key to guarantee complete platform independence
        unique_images = sorted(list(img_dict.keys()))
        total_imgs = len(unique_images)
        
        # Determine sample size
        sample_size = math.ceil(total_imgs * SAMPLE_SIZE)
        
        # Deterministic sample
        sampled_images = random.sample(unique_images, sample_size)
        
        # Collect entries
        for img_path in sampled_images:
            sampled_entries.extend(img_dict[img_path])
            
        print(f"{class_name:<45} | {total_imgs:<9} | {sample_size:<11}")
        
    print("-" * 65)
    print(f"Total sampled patches: {len(sampled_entries):,}")

    # 5. Clean / Create isolated output folder structure
    print(f"\nSetting up target directory: {OUTPUT_DIR.name}...")
    if OUTPUT_DIR.exists():
        print(f"Target directory already exists. Creating/updating contents in it.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 6. Copy physical files and compile selected metadata
    print("Copying patch image files...")
    copied_count = 0
    missing_count = 0
    
    for entry in sampled_entries:
        patch_path = entry["patch_path"]
        
        # Source & destination file paths
        # patch_path format: "needs_annotation/<class_name>/<patch_name>"
        src_file = INPUT_DIR / "train" / patch_path
        dst_file = OUTPUT_DIR / "train" / patch_path
        
        if src_file.exists():
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            copied_count += 1
        else:
            missing_count += 1
            
    print(f"Copy complete: {copied_count:,} patches copied successfully.")
    if missing_count > 0:
        print(f"[WARN] {missing_count:,} patch source files were missing and skipped.")

    # 7. Write new metadata_train.json
    out_meta_path = OUTPUT_DIR / "metadata_train.json"
    print(f"Saving new metadata file to: {out_meta_path.name}...")
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(sampled_entries, f, indent=2)

    # 8. Create assignments.json
    # Assign all sampled class folders to all team members
    unique_classes = sorted(list(set(entry["class_name"] for entry in sampled_entries)))
    assigned_folders = [
        f"train/needs_annotation/{class_name}" for class_name in unique_classes
    ]
    
    assignments = {
        member: assigned_folders for member in TEAM_MEMBERS
    }
    
    out_assignments_path = OUTPUT_DIR / "assignments.json"
    print(f"Saving assignments config to: {out_assignments_path.name}...")
    with open(out_assignments_path, "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2)

    print("\n" + "=" * 60)
    print("SUCCESS: CONSENSUS EXTRACTION COMPLETED SUCCESSFULLY!")
    print(f"  Output folder    : {OUTPUT_DIR}")
    print(f"  Patches Copied   : {copied_count:,}")
    print(f"  Metadata Saved   : {out_meta_path.name}")
    print(f"  Assignments Saved: {out_assignments_path.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
