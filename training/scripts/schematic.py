"""Paper-ready workflow schematic for the final OMI trait model.

Output: artifacts/figures/workflow_schematic.png

Two-branch cross-model ensemble:
  image → [preprocess@518] → ViT-G/14 frozen → 10 MLP heads → mean
       → [preprocess@224] → ViT-L/14 fine-tuned (last block) + head
       → mean of the two branches → 34-d trait vector
       (horizontal-flip TTA applied at inference, averaged with original)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, x, y, w, h, text, fc="#f5f5f5", ec="#333", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.002,rounding_size=0.02",
        linewidth=1.2, facecolor=fc, edgecolor=ec,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#111")


def _arrow(ax, xy1, xy2, color="#222", lw=1.3):
    ax.add_patch(FancyArrowPatch(
        xy1, xy2, arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
    ))


def _stack(ax, x, y, w, h, n=10, fc="#fff1e8", ec="#b3611c"):
    dx, dy = 0.003, 0.004
    for i in range(n):
        ax.add_patch(FancyBboxPatch(
            (x + i * dx, y - i * dy), w, h,
            boxstyle="round,pad=0.002,rounding_size=0.01",
            linewidth=0.8, facecolor=fc, edgecolor=ec, alpha=0.9,
        ))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, default=Path("images/1.jpg"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/figures/workflow_schematic.png"))
    args = ap.parse_args()

    fig = plt.figure(figsize=(17, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    fig.text(0.5, 0.96, "Face → 34-d perceived-trait vector  (OMI)",
             ha="center", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.925,
             "Cross-model ensemble: variance-weighted 518-px ViT-G heads  ⊕  fine-tuned ViT-L (224-px)  ⊕  horizontal-flip TTA",
             ha="center", fontsize=9, color="#555")

    # ---- Columns (x-centers) ----
    X0, X1, X2, X3, X4, X5 = 0.07, 0.21, 0.36, 0.51, 0.66, 0.81
    # ---- Two branch rows (y-centers) ----
    Y_TOP, Y_BOT = 0.67, 0.33
    Y_MERGE = 0.50

    H = 0.09; W = 0.12

    # ---- Input image ----
    ax_img = fig.add_axes([X0 - 0.06, 0.45, 0.11, 0.30])
    img = Image.open(args.image).convert("RGB").resize((220, 220))
    ax_img.imshow(img); ax_img.axis("off")
    ax_img.set_title("face image", fontsize=9)

    # Input arrow from image into the shared preprocessing hub
    _arrow(ax, (X0 + 0.06, 0.50), (X1 - W / 2, 0.50))

    # ---- Shared hub: TTA (horizontal flip + original) ----
    _box(ax, X1 - W / 2, 0.50 - H / 2, W, H,
         "test-time augmentation\norig + horizontal flip",
         fc="#eefaf4", ec="#1e8a5e", fs=8)

    # Split into two branches
    _arrow(ax, (X1 + W / 2, 0.50), (X2 - W / 2, Y_TOP))
    _arrow(ax, (X1 + W / 2, 0.50), (X2 - W / 2, Y_BOT))

    # ---- Top branch: ViT-G at 518 ----
    _box(ax, X2 - W / 2, Y_TOP - H / 2, W, H,
         "preprocess @ 518\n+\nDINOv2 ViT-G/14\n(~1.1 B, frozen)\nCLS → 1536-d",
         fc="#dde8f5", ec="#2a5dab", fs=7)
    _arrow(ax, (X2 + W / 2, Y_TOP), (X3 - W / 2, Y_TOP))

    _stack(ax, X3 - W / 2, Y_TOP - H / 2 + 0.005, W, H - 0.01, n=10)
    ax.text(X3, Y_TOP, "10 MLP heads\n(val-top-10)\nvariance-weighted MSE",
            ha="center", va="center", fontsize=7)
    _arrow(ax, (X3 + W / 2, Y_TOP), (X4 - W / 2, Y_TOP))

    _box(ax, X4 - W / 2, Y_TOP - H / 2, W, H,
         "mean over\n10 heads",
         fc="#fff1e8", ec="#b3611c", fs=8)
    _arrow(ax, (X4 + W / 2, Y_TOP), (X5 - W / 2, Y_MERGE + 0.05))

    # ---- Bottom branch: ViT-L fine-tuned ----
    _box(ax, X2 - W / 2, Y_BOT - H / 2, W, H,
         "preprocess @ 224\n+\nDINOv2 ViT-L/14\nlast-block FT\nCLS → 1024-d",
         fc="#f0e2f7", ec="#7b3fa5", fs=7)
    _arrow(ax, (X2 + W / 2, Y_BOT), (X3 - W / 2, Y_BOT))

    _box(ax, X3 - W / 2, Y_BOT - H / 2, W, H,
         "trained MLP head\n1024 → 1024 → 34",
         fc="#fff1e8", ec="#b3611c", fs=8)
    _arrow(ax, (X3 + W / 2, Y_BOT), (X4 - W / 2, Y_BOT))

    _box(ax, X4 - W / 2, Y_BOT - H / 2, W, H,
         "single\nfine-tune\nprediction",
         fc="#fff1e8", ec="#b3611c", fs=8)
    _arrow(ax, (X4 + W / 2, Y_BOT), (X5 - W / 2, Y_MERGE - 0.05))

    # ---- Merge ----
    _box(ax, X5 - W / 2, Y_MERGE - H / 2, W, H,
         "mean of the two\nbranch predictions",
         fc="#f4e1e1", ec="#a82929", fs=9, bold=True)

    # ---- Output ----
    _box(ax, 0.93, Y_MERGE - 0.05, 0.06, 0.10,
         "34-d\ntrait\nvector",
         fc="#222", ec="#000", fs=8)
    _arrow(ax, (X5 + W / 2, Y_MERGE), (0.93, Y_MERGE))

    # ---- Bottom panels ----
    y_head = 0.14; y_body = 0.10

    fig.text(0.04, y_head, "Training (heads)",
             fontsize=10, fontweight="bold", color="#333")
    fig.text(0.04, y_body,
             "• 803 train / 100 val / 101 test stimuli, seed-0 split\n"
             "• AdamW, cosine LR, batch 64, ≤ 400 epochs\n"
             "• Variance-weighted MSE: w = median(std) / per-cell std\n"
             "• Early stop on val mean Pearson r (patience 40)",
             fontsize=8, color="#222", va="top")

    fig.text(0.36, y_head, "Sweep & selection",
             fontsize=10, fontweight="bold", color="#333")
    fig.text(0.36, y_body,
             "• 62 head configs per backbone  (hidden × drop × lr × wd × seed)\n"
             "• Top-10 chosen by VALIDATION mean r  (no test peeking)\n"
             "• ViT-L fine-tune: unfreeze last block, head_lr = 10× backbone_lr\n"
             "• Unweighted averaging  (val-weighted gave Δ < 10⁻⁴)",
             fontsize=8, color="#222", va="top")

    fig.text(0.69, y_head, "Test results  (n = 101 stimuli, held out)",
             fontsize=10, fontweight="bold", color="#333")
    fig.text(0.69, y_body,
             "ViT-B baseline              0.7567\n"
             "ViT-L top-10 ensemble       0.8208\n"
             "ViT-G-224 top-10 ensemble   0.8361\n"
             "ViT-L fine-tune             0.8352\n"
             "ViT-G-518 top-10 (var-wt)   0.8475\n"
             "FINAL ensemble  (w/ TTA)    0.858",
             fontsize=8, color="#222", va="top", family="monospace")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
