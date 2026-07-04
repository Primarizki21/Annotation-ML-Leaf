"""Inference-time helpers for full-image (256x256) -> per-patch -> aggregate.

Used by the FastAPI backend to serve trained ONNX models. Building blocks:
  - split_into_patches: 256x256 -> list of 32x32 patches
  - filter_patches: drop near-empty patches
  - calculate_black_pixel_percentage: per-patch metric
  - calculate_prevalence_rate: Eq. 10 from the paper
  - predict_full_image: end-to-end single-image inference

Training-side utilities (CSV-driven dataset, dataloaders, transforms,
seed) live in preprocessing.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

PATCH_SIZE = 32


def split_into_patches(
    image: Image.Image, patch_size: int = PATCH_SIZE
) -> list[np.ndarray]:
    """Split a (near-)square image into non-overlapping patches.

    If image dims are not a multiple of patch_size, the image is resized
    to the nearest multiple first (LANCZOS). Returns HxWx3 uint8 arrays.
    """
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    if h % patch_size != 0 or w % patch_size != 0:
        new_h = max(round(h / patch_size) * patch_size, patch_size)
        new_w = max(round(w / patch_size) * patch_size, patch_size)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]

    patches: list[np.ndarray] = []
    for r in range(h // patch_size):
        for c in range(w // patch_size):
            y1 = r * patch_size
            y2 = y1 + patch_size
            x1 = c * patch_size
            x2 = x1 + patch_size
            patches.append(arr[y1:y2, x1:x2])
    return patches


def calculate_black_pixel_percentage(patch: np.ndarray, threshold: int = 10) -> float:
    """% of pixels with all RGB channels <= threshold (default 10)."""
    black = (
        (patch[:, :, 0] <= threshold)
        & (patch[:, :, 1] <= threshold)
        & (patch[:, :, 2] <= threshold)
    )
    return float(black.sum() / (patch.shape[0] * patch.shape[1]) * 100.0)


def filter_patches(
    patches: list[np.ndarray],
    black_threshold: float = 50.0,
) -> list[tuple[int, np.ndarray]]:
    """Drop 100%-black patches and patches with black% > black_threshold.

    Returns list of (original_index, patch) so the caller can map back
    to grid position.
    """
    kept: list[tuple[int, np.ndarray]] = []
    for i, p in enumerate(patches):
        bp = calculate_black_pixel_percentage(p)
        if bp >= 100.0:
            continue
        if bp > black_threshold:
            continue
        kept.append((i, p))
    return kept


def calculate_prevalence_rate(healthy_count: int, unhealthy_count: int) -> float:
    """Eq. 10 from the paper: P = unhealthy / (healthy + unhealthy) * 100."""
    total = healthy_count + unhealthy_count
    if total == 0:
        return 0.0
    return (unhealthy_count * 100.0) / total


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def predict_full_image(
    onnx_session: ort.InferenceSession,
    image: Image.Image,
    preprocess_fn,
    black_threshold: float = 50.0,
    up_sample_size: int = 128,
) -> dict:
    """End-to-end inference on a full leaf image.

    Args:
        onnx_session: a loaded onnxruntime.InferenceSession.
        image: PIL image of the leaf (will be resized to a multiple of 32).
        preprocess_fn: callable PIL.Image -> np.ndarray (CHW float32, normalized),
                       the same pipeline used during training (e.g.
                       ``get_transforms(up_sample_size, train=False)`` applied
                       to a single PIL image and then numpy-converted).
        black_threshold: filter threshold in %; see ``filter_patches``.
        up_sample_size: must match the model's training input size.

    Returns dict with: label, confidence, prevalence_rate, patches_kept,
    patches_total, healthy_count, unhealthy_count.
    """
    patches = split_into_patches(image, PATCH_SIZE)
    kept = filter_patches(patches, black_threshold=black_threshold)
    if not kept:
        return {
            "label": 0,
            "confidence": 0.0,
            "prevalence_rate": 0.0,
            "patches_kept": 0,
            "patches_total": len(patches),
            "healthy_count": 0,
            "unhealthy_count": 0,
        }

    batch = np.stack(
        [preprocess_fn(Image.fromarray(p).resize((up_sample_size, up_sample_size), Image.BILINEAR)) for _, p in kept]
    ).astype(np.float32)

    logits = onnx_session.run(None, {"input": batch})[0]
    probs = _softmax(logits, axis=1)
    pred_labels = probs.argmax(axis=1)
    healthy = int((pred_labels == 0).sum())
    unhealthy = int((pred_labels == 1).sum())
    prevalence = calculate_prevalence_rate(healthy, unhealthy)
    confidence = float(probs[:, 1].mean())
    final_label = 1 if prevalence > 50.0 else 0
    return {
        "label": final_label,
        "confidence": confidence,
        "prevalence_rate": prevalence,
        "patches_kept": len(kept),
        "patches_total": len(patches),
        "healthy_count": healthy,
        "unhealthy_count": unhealthy,
    }


def load_onnx_session(path: str | Path) -> ort.InferenceSession:
    """Load an ONNX model for CPU inference. Use int8 path for cheaper serving."""
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
