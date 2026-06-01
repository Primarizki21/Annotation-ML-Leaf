"""
merge_annotations.py

Merges individual annotator CSV files into a single master CSV.
Detects duplicate annotations and provides summary statistics.

Usage:
    python merge_annotations.py
    python merge_annotations.py --annotations-dir annotations --output annotations_master.csv
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge annotation CSV files")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("annotations"),
        help="Directory containing annotations_*.csv files (default: annotations)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations_master.csv"),
        help="Output master CSV path (default: annotations_master.csv)",
    )
    args = parser.parse_args()

    annotations_dir = args.annotations_dir
    if not annotations_dir.exists():
        print(f"Error: Directory '{annotations_dir}' not found")
        return

    # Find all CSV files (excluding backups)
    csv_files = sorted(annotations_dir.glob("annotations_*.csv"))
    csv_files = [f for f in csv_files if f.suffix != ".bak"]

    if not csv_files:
        print(f"Error: No annotations_*.csv files found in '{annotations_dir}'")
        return

    print(f"Found {len(csv_files)} annotation file(s):")
    for f in csv_files:
        print(f"  - {f.name}")
    print()

    # Read all CSV files
    all_rows = []
    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                all_rows.extend(rows)
        except Exception as e:
            print(f"Warning: Could not read {csv_file.name}: {e}")

    if not all_rows:
        print("Error: No annotation data found")
        return

    print(f"Total rows read: {len(all_rows)}")

    # Detect duplicates
    patch_annotations = defaultdict(list)
    for row in all_rows:
        patch_path = row.get("patch_path", "")
        patch_annotations[patch_path].append(row)

    # Separate conflicts and errors
    conflicts = []  # Same patch, different annotators, different labels
    errors = []     # Same patch, same annotator, multiple times
    clean_rows = []

    for patch_path, rows in patch_annotations.items():
        # Check for same-annotator duplicates
        annotator_counts = Counter(r.get("annotator", "") for r in rows)
        for annotator, count in annotator_counts.items():
            if count > 1:
                errors.append({
                    "patch_path": patch_path,
                    "annotator": annotator,
                    "count": count,
                })

        # Check for cross-annotator conflicts (only non-skipped)
        labeled_rows = [r for r in rows if r.get("is_skipped") != "True"]
        if len(labeled_rows) > 1:
            labels = set(r.get("label", "") for r in labeled_rows)
            if len(labels) > 1:
                conflicts.append({
                    "patch_path": patch_path,
                    "annotations": [
                        {"annotator": r.get("annotator"), "label": r.get("label")}
                        for r in labeled_rows
                    ],
                })

        # Keep all rows (including duplicates for manual review)
        clean_rows.extend(rows)

    # Report duplicates
    if errors:
        print(f"\nWarning: {len(errors)} same-annotator duplicate(s) found:")
        for err in errors[:10]:
            print(f"  {err['patch_path']} - {err['annotator']}: {err['count']} times")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    if conflicts:
        print(f"\nWarning: {len(conflicts)} cross-annotator conflict(s) found:")
        for conf in conflicts[:10]:
            print(f"  {conf['patch_path']}:")
            for ann in conf["annotations"]:
                print(f"    {ann['annotator']}: {ann['label']}")
        if len(conflicts) > 10:
            print(f"  ... and {len(conflicts) - 10} more")

    # Write master CSV
    fieldnames = ["patch_path", "class_name", "split", "label", "annotator", "timestamp", "is_skipped"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)

    print(f"\nMaster CSV written to: {args.output}")
    print(f"Total rows: {len(clean_rows)}")

    # Summary statistics
    print("\n=== Summary Statistics ===")

    # Per-annotator stats
    annotator_stats = defaultdict(lambda: {"done": 0, "skipped": 0})
    for row in clean_rows:
        ann = row.get("annotator", "unknown")
        if row.get("is_skipped") == "True":
            annotator_stats[ann]["skipped"] += 1
        else:
            annotator_stats[ann]["done"] += 1

    print("\nPer-Annotator:")
    for ann, stats in sorted(annotator_stats.items()):
        total = stats["done"] + stats["skipped"]
        print(f"  {ann}: {stats['done']} annotated, {stats['skipped']} skipped ({total} total)")

    # Per-class stats
    class_stats = defaultdict(lambda: {"healthy": 0, "unhealthy": 0, "skipped": 0})
    for row in clean_rows:
        cn = row.get("class_name", "unknown")
        if row.get("is_skipped") == "True":
            class_stats[cn]["skipped"] += 1
        elif row.get("label") == "healthy":
            class_stats[cn]["healthy"] += 1
        elif row.get("label") == "unhealthy":
            class_stats[cn]["unhealthy"] += 1

    print("\nPer-Class:")
    for cn, stats in sorted(class_stats.items()):
        total = stats["healthy"] + stats["unhealthy"]
        print(f"  {cn}: {stats['healthy']} healthy, {stats['unhealthy']} unhealthy, {stats['skipped']} skipped")

    # Overall label distribution
    label_counts = Counter()
    for row in clean_rows:
        if row.get("is_skipped") != "True":
            label_counts[row.get("label", "")] += 1

    print("\nOverall Label Distribution:")
    total_labeled = sum(label_counts.values())
    for label, count in label_counts.most_common():
        pct = count / total_labeled * 100 if total_labeled > 0 else 0
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    print(f"\nTotal annotated: {total_labeled:,}")
    print(f"Total skipped: {sum(1 for r in clean_rows if r.get('is_skipped') == 'True'):,}")
    print(f"Conflicts: {len(conflicts)}")
    print(f"Duplicates: {len(errors)}")


if __name__ == "__main__":
    main()
