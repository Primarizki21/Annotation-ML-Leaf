"""
merge_with_reviews.py

Merges original annotation CSVs with _review CSVs.
For (patch, annotator) pairs present in a review file, the review label
replaces the original. Outputs round 1 consensus results with Fleiss' Kappa.

Usage:
    python merge_with_reviews.py
"""

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

from merge_annotation_consensus import (
    compute_fleiss_kappa,
    save_class_fleiss_kappa,
    _interpret_kappa,
)

ANNOTATIONS_DIR = "./annotations"
OUTPUT_CSV = "consensus_round1_master.csv"
KAPPA_CHART = "fleiss_kappa_round1_per_class.png"


def read_csvs(pattern, exclude_review=True, exclude_bak=True):
    csv_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, pattern)))
    filtered = []
    for f in csv_files:
        if exclude_review and "_review" in f:
            continue
        if exclude_bak and f.endswith(".bak"):
            continue
        filtered.append(f)
    return [pd.read_csv(f) for f in filtered], filtered


def main():
    orig_dfs, orig_files = read_csvs("annotations_*.csv", exclude_review=True, exclude_bak=True)
    if not orig_dfs:
        print("Error: No original annotation CSV files found.")
        return

    df_orig = pd.concat(orig_dfs, ignore_index=True)
    print(f"Originals: {len(orig_files)} file(s), {len(df_orig):,} rows")

    review_dfs, review_files = read_csvs("annotations_*_review.csv", exclude_review=False, exclude_bak=True)
    df_review_all = pd.concat(review_dfs, ignore_index=True) if review_dfs else pd.DataFrame()
    print(f"Reviews:   {len(review_files)} file(s), {len(df_review_all):,} rows")

    if df_review_all.empty:
        print("No review files — running consensus on originals only.\n")
        df_combined = df_orig
        overrides = 0
    else:
        df_review = df_review_all.drop_duplicates(
            subset=["patch_path", "annotator"], keep="last"
        )
        overrides = len(df_review)

        orig_patches = set(df_orig["patch_path"])
        orphan_reviews = df_review[~df_review["patch_path"].isin(orig_patches)]
        if not orphan_reviews.empty:
            print(f"  Warning: {len(orphan_reviews)} review row(s) reference patches "
                  f"not in originals — they will be added anyway.")

        review_keys = set(zip(df_review["patch_path"], df_review["annotator"]))
        mask = pd.Series(
            [k not in review_keys for k in zip(df_orig["patch_path"], df_orig["annotator"])],
            index=df_orig.index,
        )
        df_orig_clean = df_orig[mask]
        df_combined = pd.concat([df_orig_clean, df_review], ignore_index=True)

        print(f"  Overrides: {overrides} (patch, annotator) pair(s) replaced")
        print(f"  Combined:  {len(df_combined):,} rows\n")

    df_valid = df_combined[df_combined["is_skipped"] == False].copy()

    df_pivot = df_valid.pivot_table(
        index=["patch_path", "class_name"],
        columns="annotator",
        values="label",
        aggfunc="first",
    ).reset_index()

    annotator_cols = df_pivot.columns[2:].tolist()
    kappa_overall, kappa_per_class, agreement_summary = compute_fleiss_kappa(
        df_pivot, annotator_cols
    )

    def calculate_consensus(row):
        answers = row[2:].dropna().tolist()
        if not answers:
            return pd.Series([None, 0, True])
        majority_text = max(set(answers), key=answers.count)
        agreement_ratio = answers.count(majority_text) / len(answers)
        needs_discussion = agreement_ratio <= 0.6
        numeric_label = 0 if majority_text == "healthy" else 1
        return pd.Series(
            [numeric_label, f"{answers.count(majority_text)}/{len(answers)}", needs_discussion]
        )

    df_pivot[["suggested_numeric_label", "agreement_level", "needs_discussion"]] = (
        df_pivot.apply(calculate_consensus, axis=1)
    )
    df_pivot = df_pivot.sort_values(
        by=["needs_discussion", "patch_path"], ascending=[False, True]
    )

    df_pivot.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    total_patches = len(df_pivot)
    needs_discussion = df_pivot["needs_discussion"].sum()
    print(f"Total patches: {total_patches:,}")
    print(f"Needs discussion: {int(needs_discussion):,}")

    if kappa_overall is not None:
        print(f"\n=== Fleiss' Kappa (Round 1) ===")
        print(f"Annotators: {len(annotator_cols)} ({', '.join(annotator_cols)})")
        total_agreed = sum(agreement_summary.values())
        print(f"Patches rated by all annotators: {total_agreed:,}")
        print(f"Fleiss' Kappa: {kappa_overall:.4f} — {_interpret_kappa(kappa_overall)}")

        if total_agreed > 0:
            print("\nAgreement Distribution:")
            for level, count in agreement_summary.items():
                pct = count / total_agreed * 100
                print(f"  {level}: {count:,} ({pct:.1f}%)")

        if kappa_per_class:
            print("\nPer-Class Fleiss' Kappa:")
            for class_name, kappa in sorted(kappa_per_class.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {kappa:.4f} — {_interpret_kappa(kappa)}")
            save_class_fleiss_kappa(kappa_per_class, KAPPA_CHART)
    else:
        print("\nFleiss' Kappa: not computed (no patches with complete annotations).")


if __name__ == "__main__":
    main()
