"""
assignments_generator.py

Generates assignments.json that maps annotator names to their assigned class folders.
Uses greedy bin-packing to distribute folders evenly across annotators.

Usage:
    python assignments_generator.py
    python assignments_generator.py --annotators Budi Ani Citra Dedi Eka
"""

import argparse
import json
from pathlib import Path


def count_images(folder: Path) -> int:
    """Count image files in a folder."""
    extensions = {".jpg", ".jpeg", ".png"}
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in extensions)


def main():
    parser = argparse.ArgumentParser(description="Generate assignments.json for annotators")
    parser.add_argument(
        "--annotators",
        nargs="+",
        default=["Budi", "Ani", "Citra", "Dedi", "Eka"],
        help="List of annotator names (default: Budi Ani Citra Dedi Eka)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_patches"),
        help="Path to dataset_patches directory (default: dataset_patches)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assignments.json"),
        help="Output file path (default: assignments.json)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        print(f"Error: Dataset directory '{dataset_dir}' not found")
        return

    # Collect all class folders from train and test needs_annotation
    folders = []
    for split in ["train", "test"]:
        needs_annot = dataset_dir / split / "needs_annotation"
        if not needs_annot.exists():
            print(f"Warning: '{needs_annot}' not found, skipping")
            continue
        for class_dir in sorted(needs_annot.iterdir()):
            if class_dir.is_dir():
                img_count = count_images(class_dir)
                if img_count > 0:
                    # Store path relative to dataset_patches
                    rel_path = f"{split}/needs_annotation/{class_dir.name}"
                    folders.append({"path": rel_path, "count": img_count, "class": class_dir.name})

    if not folders:
        print("Error: No image folders found")
        return

    # Sort by count descending for better bin-packing
    folders.sort(key=lambda f: f["count"], reverse=True)

    # Greedy bin-packing: assign each folder to annotator with fewest patches
    annotators = args.annotators
    loads = {a: 0 for a in annotators}
    assignments = {a: [] for a in annotators}

    for folder in folders:
        # Find annotator with minimum load
        min_annotator = min(annotators, key=lambda a: loads[a])
        assignments[min_annotator].append(folder["path"])
        loads[min_annotator] += folder["count"]

    # Write assignments.json
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"Generated {args.output}")
    print(f"Total folders: {len(folders)}")
    print(f"Annotators: {len(annotators)}")
    print()
    for name in annotators:
        folder_count = len(assignments[name])
        print(f"  {name}: {folder_count} folders, {loads[name]:,} patches")
    print()
    total = sum(loads.values())
    print(f"Total patches: {total:,}")


if __name__ == "__main__":
    main()
