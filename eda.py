"""
Exploratory Data Analysis for dataset_filtered/raw/segmented/.

Plots are registered via @register_plot; `python eda.py` (no args) runs all.
New plots: add @register_plot("name") + one argparse choice. Nothing else.
"""
import argparse
import math
import random
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DATA_ROOT_DEFAULT = "dataset_filtered/raw/segmented"
OUTPUT_DIR_DEFAULT = "output_eda"

PLOT_KAPPA_TITLE = "Sample leaves with low Fleiss' Kappa score per class"
PLOT_REVIEW_TITLE = "Sample leaves from classes flagged for human review"

KAPPA_CLASSES = [
    "Squash___Powdery_mildew",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
]
REVIEW_CLASSES = [
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Bacterial_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
]

PLOT_REGISTRY: dict[str, callable] = {}


def register_plot(name: str):
    def deco(fn):
        PLOT_REGISTRY[name] = fn
        return fn
    return deco


def _list_images(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() == ".jpg")


def _pick_samples(images: list[Path], n: int, rng: random.Random) -> list[Path]:
    return rng.sample(images, k=min(n, len(images)))


def _grid(n_rows: int, n_cols: int) -> tuple[plt.Figure, list]:
    h = max(2.5, 1.76 * n_rows)  # ponytail: 1.6 * 1.10
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.64 * n_cols, h), squeeze=False)  # 2.4 * 1.10
    fig.subplots_adjust(wspace=0.05, hspace=0.18, top=0.95, bottom=0.02, left=0.02, right=0.98)
    return fig, axes.flat


def _draw_panel(ax, img_path: Path, class_label: str):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  warn: failed to open {img_path}: {e}", file=sys.stderr)
        ax.set_axis_off()
        return
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(class_label, fontsize=8)


@register_plot("samples")
def plot_class_samples(args):
    """One image per class; layout = args.cols classes per row."""
    args.output = args.output or str(Path(OUTPUT_DIR_DEFAULT) / "class_samples.png")
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"  skip: {data_root} not found", file=sys.stderr)
        return
    classes = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not classes:
        print(f"  skip: no class folders under {data_root}", file=sys.stderr)
        return

    rng = random.Random(args.seed)
    panels = []
    for cls in classes:
        imgs = _pick_samples(_list_images(cls), args.per_class, rng)
        if imgs:
            panels.append((cls, imgs))
    if not panels:
        print("  skip: no images found in any class", file=sys.stderr)
        return

    n_cols = args.cols
    n_rows = math.ceil(len(panels) / n_cols)
    fig, axes = _grid(n_rows, n_cols)
    fig.suptitle(f"Class samples ({args.per_class}/class, {n_cols} cols)", fontsize=12, y=0.995)
    for k, (cls, imgs) in enumerate(panels):
        ax = axes[k]
        if imgs:
            _draw_panel(ax, imgs[0], cls.name)
        else:
            ax.set_axis_off()
    for k in range(len(panels), n_rows * n_cols):
        axes[k].set_axis_off()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[samples] saved → {args.output}")


def plot_specific_classes(classes, per_class, output, title, seed):
    data_root = Path(DATA_ROOT_DEFAULT)
    if not data_root.is_dir():
        print(f"  skip: {data_root} not found", file=sys.stderr)
        return
    rng = random.Random(seed)
    rows = []
    for cls_name in classes:
        cls_dir = data_root / cls_name
        imgs = _pick_samples(_list_images(cls_dir), per_class, rng)
        if not imgs:
            print(f"  warn: no images in {cls_dir}", file=sys.stderr)
        rows.append((cls_name, imgs))

    fig, axes = _grid(len(classes), per_class)
    fig.suptitle(title, fontsize=12, y=0.995)
    for idx, (cls_name, imgs) in enumerate(rows):
        for j in range(per_class):
            ax = axes[idx * per_class + j]
            if j < len(imgs):
                _draw_panel(ax, imgs[j], cls_name)
            else:
                ax.set_axis_off()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {output}")


@register_plot("kappa")
def _kappa(args):
    args.output = args.output or str(Path(OUTPUT_DIR_DEFAULT) / "low_fleiss_kappa.png")
    plot_specific_classes(KAPPA_CLASSES, 3, args.output, PLOT_KAPPA_TITLE, args.seed)


@register_plot("review")
def _review(args):
    args.output = args.output or str(Path(OUTPUT_DIR_DEFAULT) / "needs_most_review.png")
    plot_specific_classes(REVIEW_CLASSES, 3, args.output, PLOT_REVIEW_TITLE, args.seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description="EDA plots for dataset_filtered")
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    sub = parser.add_subparsers(dest="command")
    sub.required = False

    p_samples = sub.add_parser("samples", help="one image per class")
    p_samples.add_argument("--per-class", type=int, default=1)
    p_samples.add_argument("--cols", type=int, default=5)
    p_samples.add_argument("--output", default=None)

    p_kappa = sub.add_parser("kappa", help="low Fleiss' Kappa classes (3/class)")
    p_kappa.add_argument("--output", default=None)

    p_review = sub.add_parser("review", help="classes flagged for human review (3/class)")
    p_review.add_argument("--output", default=None)

    args = parser.parse_args(argv)

    if args.command is None:
        args.per_class = 1
        args.cols = 5
        for name, fn in PLOT_REGISTRY.items():
            print(f"--- {name} ---")
            args.output = None
            fn(args)
    else:
        PLOT_REGISTRY[args.command](args)


def _self_test():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for cls in ["alpha", "beta"]:
            (td / cls).mkdir()
            Image.new("RGB", (32, 32), (200, 50, 50)).save(td / cls / f"{cls}_1.jpg")
            Image.new("RGB", (32, 32), (50, 200, 50)).save(td / cls / f"{cls}_2.jpg")
        out = td / "test.png"
        plot_specific_classes(["alpha", "beta"], 2, str(out), "self-test", seed=0)
        assert out.exists() and out.stat().st_size > 0, "self-test failed: PNG not written"
        print("self-test ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
