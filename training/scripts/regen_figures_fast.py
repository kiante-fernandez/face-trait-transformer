"""Fast figure regeneration using cached features (no live predictor call).

Pulls predictions from:
  * the 518-vw ViT-G head ensemble (via cached features → heads)
  * the fine-tune model (must be pre-computed; stored once in finetune_test_preds.npy)

Average of the two groups is the final ensemble.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr

from face_trait_transformer.data import load_labels, load_splits
from face_trait_transformer.features import build_transform, pick_device
from face_trait_transformer.model import TraitHead


def _ensemble_from_features(ckpt_dir: Path, X: np.ndarray, k: int) -> np.ndarray:
    cks = []
    for p in sorted(ckpt_dir.glob("*.pt")):
        rj = ckpt_dir / (p.stem + ".result.json")
        d = json.loads(rj.read_text())
        cks.append((d["val_mean_r"], p))
    cks.sort(reverse=True)
    preds = []
    for _, p in cks[:k]:
        m = torch.load(p, map_location="cpu", weights_only=False)
        c = m["config"]
        h = TraitHead(in_dim=c["in_dim"], out_dim=c["out_dim"],
                      hidden=c["hidden"], dropout=c["dropout"])
        h.load_state_dict(m["state_dict"]); h.eval()
        with torch.inference_mode():
            preds.append(h(torch.from_numpy(X).float()).numpy())
    return np.stack(preds).mean(0)


def _ft_preds_cached(cache: Path, ckpt: Path, images_dir: Path,
                     test_ids: list[int]) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    device = pick_device()
    meta = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = meta["config"]
    backbone = torch.hub.load("facebookresearch/dinov2", cfg["backbone"])
    head = TraitHead(in_dim=cfg["in_dim"], out_dim=cfg["out_dim"],
                     hidden=cfg["hidden"], dropout=cfg["dropout"])

    class Wrap(torch.nn.Module):
        def __init__(self, b, h):
            super().__init__(); self.backbone = b; self.head = h
        def forward(self, x):
            return self.head(self.backbone(x))

    mod = Wrap(backbone, head).to(device)
    mod.load_state_dict(meta["state_dict"]); mod.eval()
    tf = build_transform()
    preds = []
    with torch.inference_mode():
        for sid in test_ids:
            img = Image.open(images_dir / f"{sid}.jpg").convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            preds.append(mod(x).cpu().numpy()[0])
    arr = np.stack(preds)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def _radar(ax, attr, y_true, y_pred, title=""):
    n = len(attr)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    tv = np.clip(y_true, 0, 100); pv = np.clip(y_pred, 0, 100)
    ac = np.concatenate([angles, angles[:1]])
    tc = np.concatenate([tv, tv[:1]]); pc = np.concatenate([pv, pv[:1]])
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 100); ax.set_yticks([50, 100]); ax.set_yticklabels([])
    ax.set_xticks(angles); ax.set_xticklabels([])
    ax.fill(ac, tc, color="#333", alpha=0.18, linewidth=0)
    ax.plot(ac, tc, "-", color="#333", lw=1.4, label="observed")
    ax.plot(angles, tv, "o", color="#333", markersize=3)
    ax.fill(ac, pc, color="#d62728", alpha=0.18, linewidth=0)
    ax.plot(ac, pc, "-", color="#d62728", lw=1.4, label="predicted")
    ax.plot(angles, pv, "o", color="#d62728", markersize=3)
    for ang, name in zip(angles, attr):
        deg = np.degrees(ang); rot = 90 - deg
        if rot > 90: rot -= 180
        elif rot < -90: rot += 180
        ax.text(ang, 112, name, rotation=rot, rotation_mode="anchor",
                ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10, pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=2, fontsize=8, frameon=False)


def _bars(ax, attr, y_true, y_pred, title=""):
    n = len(attr); y = np.arange(n); bh = 0.38
    tv = np.clip(y_true, 0, 100); pv = np.clip(y_pred, 0, 100)
    ax.barh(y - bh / 2, tv, bh, color="#555", ec="#000", lw=0.5, label="observed")
    ax.barh(y + bh / 2, pv, bh, color="#d62728", ec="#000", lw=0.5, label="predicted")
    for yi, (t, p) in enumerate(zip(tv, pv)):
        ax.text(max(t, p) + 2, yi, f"{t:.0f} / {p:.0f}", va="center", fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels(attr, fontsize=7); ax.invert_yaxis()
    ax.set_xlim(0, 110); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("rating (0-100)  —  labels show observed / predicted", fontsize=8)
    ax.axvline(0, color="#000", lw=0.6); ax.grid(axis="x", alpha=0.2)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7, loc="lower right", frameon=False)


def _plot_one(image_path, attr, y_true, y_pred, title, out):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.8, 1.6], wspace=0.35)
    ax_i = fig.add_subplot(gs[0, 0]); ax_i.imshow(Image.open(image_path).convert("RGB")); ax_i.axis("off"); ax_i.set_title(title, fontsize=11)
    ax_r = fig.add_subplot(gs[0, 1], projection="polar"); _radar(ax_r, attr, y_true, y_pred, "perceived-trait radar (0-100)")
    ax_b = fig.add_subplot(gs[0, 2]); _bars(ax_b, attr, y_true, y_pred, "per-attribute observed vs predicted")
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)


def _plot_overview(attr, y_true, y_pred, out):
    n = y_true.shape[1]; ncols = 6; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    for j in range(n):
        r, _ = pearsonr(y_true[:, j], y_pred[:, j])
        ax = axes.flat[j]
        ax.scatter(y_true[:, j], y_pred[:, j], s=6, alpha=0.55, color="#d62728")
        ax.plot([0, 100], [0, 100], color="#333", lw=0.7, ls="--")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
        ax.tick_params(labelsize=6); ax.set_title(f"{attr[j]}  r={r:.2f}", fontsize=8)
        if j % ncols == 0: ax.set_ylabel("predicted", fontsize=7)
        if j // ncols == nrows - 1: ax.set_xlabel("observed", fontsize=7)
    for k in range(n, nrows * ncols): axes.flat[k].axis("off")
    fig.suptitle("Observed vs predicted per attribute (test split, ViT-G-518-vw + ViT-L fine-tune)",
                 fontsize=11, y=1.00)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("artifacts/h2_features/features_vitg14_518.npy"))
    ap.add_argument("--ids", type=Path, default=Path("artifacts/h2_features/stimulus_ids_vitg14_518.npy"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/h2_sweep_vitg518_vw_ckpts"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--ft-ckpt", type=Path, default=Path("artifacts/bundle/finetune/finetune_vitl_lb1.pt"))
    ap.add_argument("--ft-cache", type=Path, default=Path("artifacts/ft_test_preds.npy"))
    ap.add_argument("--labels", type=Path, default=Path("attribute_means.csv"))
    ap.add_argument("--splits", type=Path, default=Path("artifacts/splits.json"))
    ap.add_argument("--images-dir", type=Path, default=Path("images"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/figures"))
    ap.add_argument("--n-faces", type=int, default=16)
    args = ap.parse_args()

    ids, Y, attr = load_labels(args.labels)
    splits = load_splits(args.splits)
    id_to_row = {int(i): k for k, i in enumerate(ids)}
    test_ids = [int(s) for s in splits["test"]]
    Y_test = np.stack([Y[id_to_row[s]] for s in test_ids]) * 100

    # 518-vw head ensemble
    X = np.load(args.features); fi = np.load(args.ids)
    fpos = {int(i): k for k, i in enumerate(fi)}
    X_test = np.stack([X[fpos[s]] for s in test_ids])
    head_preds = _ensemble_from_features(args.ckpt_dir, X_test, args.top_k)

    # Fine-tune predictions (cached after first run)
    ft_preds = _ft_preds_cached(args.ft_cache, args.ft_ckpt, args.images_dir, test_ids)

    # Final ensemble: average in [0,1] then rescale
    Y_pred = np.clip((head_preds + ft_preds) / 2 * 100, 0, 100)

    # Per-face plots spanning error quantiles
    mae = np.mean(np.abs(Y_pred - Y_test), axis=1)
    order = np.argsort(mae)
    picks_idx = np.linspace(0, len(order) - 1, num=args.n_faces, dtype=int)
    per_face = args.out_dir / "per_face"; per_face.mkdir(parents=True, exist_ok=True)
    image_paths = [args.images_dir / f"{s}.jpg" for s in test_ids]
    for pi in picks_idx:
        i = order[pi]; sid = test_ids[i]
        title = f"stimulus {sid}  |  mean |err| = {mae[i]:.1f} / 100"
        out = per_face / f"stimulus_{sid:04d}.png"
        _plot_one(image_paths[i], attr, Y_test[i], Y_pred[i], title, out)
        print(f"wrote {out}")

    overview = args.out_dir / "overview_scatter.png"
    _plot_overview(attr, Y_test, Y_pred, overview)
    print(f"wrote {overview}")


if __name__ == "__main__":
    main()
