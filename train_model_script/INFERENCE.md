# Inference Guide — Trained ONNX Models

This guide shows how to load and serve the 5 trained ONNX models from the
`outputs/{model_name}/exported/` folder. All inference helpers live in
`preprocess_inference.py` (sibling file in this folder).

## 1. Choosing fp32 vs int8

| File | Size (typical) | When to use |
|---|---|---|
| `model.onnx` | 5–20 MB | Best accuracy, GPU or beefy CPU server. |
| `model_int8.onnx` | 1–6 MB | Cheaper CPU serving, edge devices, faster cold-start. Accuracy drop is usually < 1% on a binary task. |

Switch between them at startup based on traffic / latency budget. Both
share the same `input`/`logits` I/O names.

## 2. Wire into a FastAPI endpoint

The existing project already has FastAPI in `pyproject.toml`. Drop this
into a new route file (e.g. `predict_routes.py`) and include it from
`main.py`:

```python
from pathlib import Path
from functools import lru_cache
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from train_model_script.preprocess_inference import (
    load_onnx_session,
    predict_full_image,
)

router = APIRouter(prefix="/predict")

ONNX_DIR = Path(__file__).parent / "train_model_script" / "outputs"

MODELS = {
    "mobilenetv3_small": {
        "fp32": ONNX_DIR / "mobilenetv3_small" / "exported" / "model.onnx",
        "int8": ONNX_DIR / "mobilenetv3_small" / "exported" / "model_int8.onnx",
        "up_sample_size": 128,
    },
    "efficientnet_b0": {
        "fp32": ONNX_DIR / "efficientnet_b0" / "exported" / "model.onnx",
        "int8": ONNX_DIR / "efficientnet_b0" / "exported" / "model_int8.onnx",
        "up_sample_size": 160,
    },
    "shufflenetv2_x1_0": {
        "fp32": ONNX_DIR / "shufflenetv2_x1_0" / "exported" / "model.onnx",
        "int8": ONNX_DIR / "shufflenetv2_x1_0" / "exported" / "model_int8.onnx",
        "up_sample_size": 128,
    },
    "squeezenet1_1": {
        "fp32": ONNX_DIR / "squeezenet1_1" / "exported" / "model.onnx",
        "int8": ONNX_DIR / "squeezenet1_1" / "exported" / "model_int8.onnx",
        "up_sample_size": 128,
    },
    "small_inception": {
        "fp32": ONNX_DIR / "small_inception" / "exported" / "model.onnx",
        "int8": None,  # baseline paper replication: fp32 only
        "up_sample_size": 32,
    },
}

# (mobilenet, efficientnet, shufflenet, squeezenet) all expect the
# ImageNet normalization used during training.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_preprocess(up_sample_size: int):
    def _preprocess(pil_img: Image.Image) -> np.ndarray:
        img = pil_img.convert("RGB").resize(
            (up_sample_size, up_sample_size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(arr, (2, 0, 1))  # CHW
    return _preprocess


@lru_cache(maxsize=8)
def get_session(model: str, quantized: bool):
    cfg = MODELS[model]
    path = cfg["int8"] if (quantized and cfg["int8"]) else cfg["fp32"]
    if path is None or not path.exists():
        raise HTTPException(503, f"Model '{model}' not exported yet.")
    return load_onnx_session(path), cfg["up_sample_size"], make_preprocess(cfg["up_sample_size"])


@router.post("/{model}")
async def predict(
    model: str,
    file: UploadFile = File(...),
    quantized: bool = True,
):
    if model not in MODELS:
        raise HTTPException(404, f"Unknown model '{model}'")
    session, up_size, preprocess = get_session(model, quantized)
    image = Image.open(io.BytesIO(await file.read()))
    return predict_full_image(
        onnx_session=session,
        image=image,
        preprocess_fn=preprocess,
        black_threshold=50.0,
        up_sample_size=up_size,
    )
```

In `main.py`:
```python
from predict_routes import router as predict_router
app.include_router(predict_router)
```

Then:
```bash
curl -F "file=@leaf.jpg" "http://localhost:8000/predict/efficientnet_b0?quantized=true"
# {"label": 1, "confidence": 0.87, "prevalence_rate": 73.2, ...}
```

## 3. About `predict_full_image`

The function:

1. Splits the input image into 32×32 patches (resizes to nearest multiple of 32 first if needed).
2. Drops patches that are 100% black or whose black-pixel % exceeds `black_threshold` (default 50 — matches the paper's pre-processing convention; relax it if you want more conservative predictions).
3. Runs the ONNX model on each kept patch.
4. Aggregates via Eq. 10: `P = unhealthy / (healthy + unhealthy) * 100`.
5. Final label is `1` (unhealthy) if `P > 50`, else `0` (healthy).
6. Returns the label, mean unhealthy-probability confidence, the prevalence rate, and patch counts.

## 4. Troubleshooting

- **`Input size mismatch`**: ensure the `up_sample_size` in `MODELS` matches what was passed to `train_*.py --up_sample_size` (default 128 for most scripts, 160 for EfficientNet, 32 for small_inception).
- **Slow first request**: ONNX Runtime compiles kernels on first call. Use a warmup batch at startup.
- **Different results vs training**: ensure the same `IMAGENET_MEAN`/`STD` and `BILINEAR` resize are used; any deviation shifts the input distribution and degrades accuracy.
