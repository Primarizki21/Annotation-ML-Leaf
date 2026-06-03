import os
import shutil
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_ROOT = "./dataset_plantvillage"
SPLIT_DIR    = os.path.join(DATASET_ROOT, "splits")
RAW_DIR      = os.path.join(DATASET_ROOT, "raw", "segmented")
OUTPUT_DIR   = "./dataset_filtered"   # folder output hasil filtering

TRAIN_TXT = os.path.join(SPLIT_DIR, "segmented_train.txt")
TEST_TXT  = os.path.join(SPLIT_DIR, "segmented_test.txt")

INDONESIA_RELEVANT_CROPS = {
    "Tomato", "Corn_(maize)", "Potato", 
    "Apple", "Grape", "Strawberry", "Squash"
}

# Kelas final yang sudah fix dari diskusi
SELECTED_CLASSES = {
    # Tomato (10 kelas)
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    # Corn (4 kelas)
    "Corn_(maize)___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    # Potato (3 kelas)
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    # Strawberry (2 kelas)
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    # Squash (1 kelas)
    "Squash___Powdery_mildew",
    # Apple (4 kelas)
    "Apple___healthy",
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    # Grape (4 kelas)
    "Grape___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
}

def filter_split_file(txt_path, output_txt_path):
    kept   = []
    skipped = 0

    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        # format: raw/segmented/<ClassName>/<filename>
        parts = line.split("/")
        if len(parts) < 4:
            skipped += 1
            continue
        class_name = parts[2]
        if class_name in SELECTED_CLASSES:
            kept.append(line)
        else:
            skipped += 1

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as f:
        f.write("\n".join(kept))

    print(f"  {os.path.basename(txt_path)}: {len(kept)} kept, {skipped} skipped")
    return kept


def copy_filtered_images(kept_paths, src_root, dst_root):
    copied  = 0
    missing = 0

    for rel_path in kept_paths:
        src = os.path.join(src_root, rel_path)
        dst = os.path.join(dst_root, rel_path)

        if not os.path.exists(src):
            missing += 1
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

        if copied % 1000 == 0:
            print(f"    Copied {copied} files...")

    print(f"  Done: {copied} copied, {missing} missing/skipped")


def run_filtering():
    print("=" * 55)
    print("  STEP 1: Filtering Split Files")
    print("=" * 55)

    train_out = os.path.join(OUTPUT_DIR, "splits", "train.txt")
    test_out  = os.path.join(OUTPUT_DIR, "splits", "test.txt")

    train_kept = filter_split_file(TRAIN_TXT, train_out)
    test_kept  = filter_split_file(TEST_TXT,  test_out)

    all_kept = list(set(train_kept + test_kept))
    print(f"\n  Total unique images (train+test): {len(all_kept):,}")

    print("\n" + "=" * 55)
    print("  STEP 2: Copying Filtered Images")
    print("=" * 55)
    copy_filtered_images(all_kept, DATASET_ROOT, OUTPUT_DIR)

    print("\n✓ Filtering selesai!")
    print(f"  Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_filtering()