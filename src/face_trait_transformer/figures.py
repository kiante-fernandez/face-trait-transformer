"""Optional diagnostic figure for a single-image prediction.

Imports matplotlib lazily — only pulled in if the user actually asks for a
figure. Install via `pip install face-trait-transformer[figures]`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def render_single_prediction_figure(
    image: str | Path | Image.Image,
    attr_names: list[str],
    values: np.ndarray,
    out_path: str | Path | None = None,
    show: bool = False,
):
    """Render image + radar + sorted-bar chart, save to out_path if given.

    Returns the matplotlib Figure. Values are in 0–100.
    """
    import matplotlib.pyplot as plt

    pil = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.8, 1.6], wspace=0.35)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(pil); ax_img.axis("off")
    title = str(image) if not isinstance(image, Image.Image) else "<PIL image>"
    ax_img.set_title(title, fontsize=11)

    # radar
    ax_r = fig.add_subplot(gs[0, 1], projection="polar")
    n = len(attr_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vals = np.clip(values, 0, 100)
    ac = np.concatenate([angles, angles[:1]])
    vc = np.concatenate([vals, vals[:1]])
    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)
    ax_r.set_ylim(0, 100); ax_r.set_yticks([50, 100]); ax_r.set_yticklabels([])
    ax_r.set_xticks(angles); ax_r.set_xticklabels([])
    ax_r.fill(ac, vc, color="#d62728", alpha=0.2, linewidth=0)
    ax_r.plot(ac, vc, "-", color="#d62728", lw=1.5)
    ax_r.plot(angles, vals, "o", color="#d62728", markersize=3)
    for ang, name in zip(angles, attr_names):
        deg = np.degrees(ang); rot = 90 - deg
        if rot > 90: rot -= 180
        elif rot < -90: rot += 180
        ax_r.text(ang, 112, name, rotation=rot, rotation_mode="anchor",
                  ha="center", va="center", fontsize=7)
    ax_r.set_title("predicted-trait radar (0–100)", fontsize=10, pad=18)

    # sorted bars
    order = np.argsort(-vals)
    sorted_attrs = [attr_names[i] for i in order]
    sorted_vals = vals[order]
    ax_b = fig.add_subplot(gs[0, 2])
    y = np.arange(len(attr_names))
    ax_b.barh(y, sorted_vals, 0.7, color="#d62728", edgecolor="#000", linewidth=0.5)
    for yi, v in enumerate(sorted_vals):
        ax_b.text(v + 1, yi, f"{v:.0f}", va="center", fontsize=7)
    ax_b.set_yticks(y); ax_b.set_yticklabels(sorted_attrs, fontsize=7)
    ax_b.invert_yaxis(); ax_b.set_xlim(0, 110)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.set_xlabel("predicted rating (0–100)", fontsize=8)
    ax_b.grid(axis="x", alpha=0.2); ax_b.axvline(0, color="#000", lw=0.6)
    ax_b.set_title("predictions, sorted high → low", fontsize=10)

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    return fig
