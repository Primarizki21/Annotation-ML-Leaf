"""Compare results across the 5 trained models.

Reads outputs/{model_name}/eval/metrics.json (and the optional benchmark
/ size files) and writes:
  - results_table.csv
  - results_table.md
  - results_comparison.png (3 subplots: accuracy, model size, inference speed)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_NAMES = [
    "small_inception",
    "mobilenetv3_small",
    "efficientnet_b0",
    "shufflenetv2_x1_0",
    "squeezenet1_1",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare the 5 trained models")
    p.add_argument("--output_dir", type=Path, default=Path("./outputs"))
    p.add_argument("--comparison_dir", type=Path, default=Path("./comparison"))
    return p.parse_args()


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for name in MODEL_NAMES:
        model_dir = output_dir / name
        if not model_dir.exists():
            continue
        metrics = read_json(model_dir / "eval" / "metrics.json") or {}
        qmeta = read_json(model_dir / "exported" / "quantization_meta.json") or {}
        bench = read_json(model_dir / "eval" / "inference_benchmark.json") or {}
        size_cmp = read_json(model_dir / "eval" / "model_size_comparison.json") or {}
        rows.append({
            "model": name,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "onnx_fp32_mb": qmeta.get("fp32_size_mb"),
            "onnx_int8_mb": qmeta.get("int8_size_mb"),
            "compression_ratio": qmeta.get("compression_ratio"),
            "inference_mean_ms": bench.get("mean_ms"),
            "inference_std_ms": bench.get("std_ms"),
            "extra": {**size_cmp} if size_cmp else {},
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model", "accuracy", "precision", "recall", "f1_score",
        "onnx_fp32_mb", "onnx_int8_mb", "compression_ratio",
        "inference_mean_ms", "inference_std_ms",
    ]
    with path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            line = []
            for c in cols:
                v = r.get(c)
                line.append("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
            f.write(",".join(line) + "\n")


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Model", "Accuracy", "Precision", "Recall", "F1",
        "fp32 (MB)", "int8 (MB)", "Compr. ratio", "Infer ms (mean±std)",
    ]
    with path.open("w") as f:
        f.write("# 5-Architecture Comparison\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            acc = _fmt(r.get("accuracy"), 4)
            pre = _fmt(r.get("precision"), 4)
            rec = _fmt(r.get("recall"), 4)
            f1 = _fmt(r.get("f1_score"), 4)
            fp32 = _fmt(r.get("onnx_fp32_mb"), 3)
            int8 = _fmt(r.get("onnx_int8_mb"), 3)
            ratio = _fmt(r.get("compression_ratio"), 3)
            infer = ""
            if r.get("inference_mean_ms") is not None:
                infer = f"{r['inference_mean_ms']:.2f} ± {r.get('inference_std_ms', 0):.2f}"
            f.write(f"| {r['model']} | {acc} | {pre} | {rec} | {f1} | {fp32} | {int8} | {ratio} | {infer} |\n")


def _fmt(v, n: int) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def plot_comparison(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if r.get("accuracy") is not None]
    names = [r["model"] for r in valid]
    acc = [r["accuracy"] for r in valid]
    fp32 = [r.get("onnx_fp32_mb") or 0 for r in valid]
    int8 = [r.get("onnx_int8_mb") or 0 for r in valid]
    infer = [r.get("inference_mean_ms") for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].bar(names, acc, color="#3b82f6")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Accuracy")
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    x = range(len(names))
    w = 0.4
    axes[1].bar([i - w / 2 for i in x], fp32, width=w, label="fp32", color="#10b981")
    axes[1].bar([i + w / 2 for i in x], int8, width=w, label="int8", color="#f59e0b")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names, rotation=30)
    axes[1].set_ylabel("Size (MB)")
    axes[1].set_title("ONNX model size")
    axes[1].legend()

    infer_vals = [v if v is not None else 0 for v in infer]
    axes[2].bar(names, infer_vals, color="#8b5cf6")
    axes[2].set_ylabel("Inference time per 64-patch batch (ms)")
    axes[2].set_title("Inference speed (CPU)")
    axes[2].tick_params(axis="x", rotation=30)
    for i, v in enumerate(infer_vals):
        if v > 0:
            axes[2].text(i, v + max(infer_vals) * 0.01, f"{v:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = collect(args.output_dir)
    if not rows:
        print(f"No model outputs found under {args.output_dir}. Train the 5 models first.")
        return
    args.comparison_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.comparison_dir / "results_table.csv", rows)
    write_markdown(args.comparison_dir / "results_table.md", rows)
    plot_comparison(args.comparison_dir / "results_comparison.png", rows)
    print(f"Wrote {args.comparison_dir}/results_table.csv, results_table.md, results_comparison.png")
    print("\nQuick view:")
    for r in rows:
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "—"
        fp32 = f"{r['onnx_fp32_mb']:.2f}MB" if r.get("onnx_fp32_mb") else "—"
        int8 = f"{r['onnx_int8_mb']:.2f}MB" if r.get("onnx_int8_mb") else "—"
        print(f"  {r['model']:22s} acc={acc}  fp32={fp32}  int8={int8}")


if __name__ == "__main__":
    main()
