import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────────────────────
FILTERED_DIR  = "./dataset_filtered"
PATCHES_DIR   = "./dataset_patches"      # output patch
PATCH_SIZE    = 64                        # ukuran patch (bisa ganti 32 atau 128)
BLACK_THRESH  = 100.0                     # buang patch dengan % black pixel >= ini
MIN_LEAF_PCT  = 15.0                     # buang patch dengan % pixel daun < threshold ini
# Untuk patch dari gambar DISEASED: perlu anotasi manual
# Untuk patch dari gambar HEALTHY: otomatis label = "healthy"
AUTO_LABEL_HEALTHY = True
# ─────────────────────────────────────────────────────────────────────────────


def get_black_pixel_percentage(patch_array, threshold=10):
    black_mask = (
        (patch_array[:, :, 0] <= threshold) &
        (patch_array[:, :, 1] <= threshold) &
        (patch_array[:, :, 2] <= threshold)
    )
    total_pixels = patch_array.shape[0] * patch_array.shape[1]
    black_count  = black_mask.sum()
    return (black_count / total_pixels) * 100.0


def split_image_to_patches(img_array, patch_size):
    h, w = img_array.shape[:2]

    # Resize ke kelipatan patch_size terdekat
    new_h = round(h / patch_size) * patch_size
    new_w = round(w / patch_size) * patch_size
    if new_h == 0: new_h = patch_size
    if new_w == 0: new_w = patch_size

    if new_h != h or new_w != w:
        img_pil   = Image.fromarray(img_array)
        img_pil   = img_pil.resize((new_w, new_h), Image.LANCZOS)
        img_array = np.array(img_pil)

    patches = []
    n_rows  = new_h // patch_size
    n_cols  = new_w // patch_size

    for r in range(n_rows):
        for c in range(n_cols):
            y1 = r * patch_size
            y2 = y1 + patch_size
            x1 = c * patch_size
            x2 = x1 + patch_size
            patch = img_array[y1:y2, x1:x2]
            patches.append((patch, r, c))

    return patches


def get_original_black_percentage(img_array, threshold=10):
    black_mask = (
        (img_array[:, :, 0] <= threshold) &
        (img_array[:, :, 1] <= threshold) &
        (img_array[:, :, 2] <= threshold)
    )
    total = img_array.shape[0] * img_array.shape[1]
    return (black_mask.sum() / total) * 100.0


def process_one_image(img_path, out_dir, class_name, is_healthy, patch_size):
    try:
        img     = Image.open(img_path).convert("RGB")
        img_arr = np.array(img)
    except Exception as e:
        print(f"  [WARN] Gagal buka {img_path}: {e}")
        return []

    # Threshold filter: % black pixel gambar original (paper pakai ~50%)
    orig_black_pct = get_original_black_percentage(img_arr)
    black_threshold = min(orig_black_pct, 50.0)  # cap di 50% seperti paper

    patches  = split_image_to_patches(img_arr, patch_size)
    metadata = []

    for patch_arr, row, col in patches:
        patch_black_pct = get_black_pixel_percentage(patch_arr)

        # Buang patch yang full hitam atau melebihi threshold
        if patch_black_pct >= BLACK_THRESH:
            continue
        if patch_black_pct > black_threshold:
            continue

        # Buang patch dengan terlalu sedikit pixel daun (absolut)
        patch_leaf_pct = 100.0 - patch_black_pct
        if patch_leaf_pct < MIN_LEAF_PCT:
            continue

        # Tentukan label otomatis untuk gambar healthy
        if is_healthy and AUTO_LABEL_HEALTHY:
            label = "healthy"
        else:
            label = "needs_annotation"  # akan dianotasi manual

        # Nama file patch: <stem>__r{row}_c{col}.jpg
        stem      = Path(img_path).stem
        patch_name = f"{stem}__r{row:02d}_c{col:02d}.jpg"
        save_dir  = os.path.join(out_dir, label, class_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, patch_name)

        Image.fromarray(patch_arr).save(save_path, quality=95)

        metadata.append({
            "patch_path"     : os.path.relpath(save_path, out_dir),
            "source_image"   : str(img_path),
            "class_name"     : class_name,
            "is_healthy_class": is_healthy,
            "label"          : label,
            "row"            : row,
            "col"            : col,
            "black_pct"      : round(patch_black_pct, 2),
        })

    return metadata


def run_patch_splitting(split_txt, split_name):
    print(f"\n{'='*55}")
    print(f"  Processing split: {split_name}")
    print(f"{'='*55}")

    out_dir = os.path.join(PATCHES_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(split_txt) as f:
        lines = [l.strip() for l in f if l.strip()]

    all_metadata  = []
    total_patches = 0
    auto_labeled  = 0
    needs_annot   = 0

    for line in tqdm(lines, desc=f"Splitting {split_name}"):
        parts = line.split("/")
        if len(parts) < 4:
            continue

        class_name = parts[2]
        is_healthy = class_name.endswith("___healthy") or "healthy" in class_name.lower()
        img_path   = os.path.join(FILTERED_DIR, line)

        if not os.path.exists(img_path):
            continue

        patches_meta = process_one_image(
            img_path   = img_path,
            out_dir    = out_dir,
            class_name = class_name,
            is_healthy = is_healthy,
            patch_size = PATCH_SIZE,
        )

        for m in patches_meta:
            if m["label"] == "healthy":
                auto_labeled += 1
            else:
                needs_annot += 1

        all_metadata.extend(patches_meta)
        total_patches += len(patches_meta)

    # Simpan metadata ke JSON
    meta_path = os.path.join(PATCHES_DIR, f"metadata_{split_name}.json")
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n  Hasil split '{split_name}':")
    print(f"    Total patch dihasilkan : {total_patches:,}")
    print(f"    Auto-labeled (healthy) : {auto_labeled:,}")
    print(f"    Perlu anotasi manual   : {needs_annot:,}")
    print(f"    Metadata disimpan di   : {meta_path}")

    return all_metadata


def print_patch_summary(train_meta, test_meta):
    total = len(train_meta) + len(test_meta)
    auto  = sum(1 for m in train_meta + test_meta if m["label"] == "healthy")
    annot = sum(1 for m in train_meta + test_meta if m["label"] == "needs_annotation")

    print(f"\n{'='*55}")
    print(f"  PATCH SPLITTING SUMMARY")
    print(f"{'='*55}")
    print(f"  Total patch keseluruhan  : {total:,}")
    print(f"  Auto-labeled (healthy)   : {auto:,} ({auto/total*100:.1f}%)")
    print(f"  Perlu anotasi manual     : {annot:,} ({annot/total*100:.1f}%)")
    print(f"  Output folder            : {PATCHES_DIR}/")
    print(f"{'='*55}")
    print(f"\n  Struktur output:")
    print(f"  {PATCHES_DIR}/")
    print(f"  ├── train/")
    print(f"  │   ├── healthy/<ClassName>/<patch>.jpg")
    print(f"  │   └── needs_annotation/<ClassName>/<patch>.jpg")
    print(f"  ├── test/")
    print(f"  │   ├── healthy/<ClassName>/<patch>.jpg")
    print(f"  │   └── needs_annotation/<ClassName>/<patch>.jpg")
    print(f"  ├── metadata_train.json")
    print(f"  └── metadata_test.json")


if __name__ == "__main__":
    train_txt = os.path.join(FILTERED_DIR, "splits", "train.txt")
    test_txt  = os.path.join(FILTERED_DIR, "splits", "test.txt")

    train_meta = run_patch_splitting(train_txt, "train")
    test_meta  = run_patch_splitting(test_txt,  "test")

    print_patch_summary(train_meta, test_meta)
    print("\n✓ Patch splitting selesai! Lanjut ke anotasi manual.")