"""Train a trait-regression head on cached DINOv2 features.

Example:
    python -m scripts.train \
        --features artifacts/features.npy --ids artifacts/stimulus_ids.npy \
        --labels attribute_means.csv \
        --head mlp --hidden 512 --dropout 0.2 \
        --epochs 200 --lr 3e-4 --wd 1e-4 \
        --out artifacts/checkpoints/mlp_v1.pt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from face_trait_transformer.data import index_by_id, load_labels, load_splits, make_splits, save_splits
from face_trait_transformer.features import pick_device
from face_trait_transformer.metrics import per_attribute_pearson
from face_trait_transformer.model import TraitHead


def _make_loaders(
    X: np.ndarray, Y: np.ndarray, ids: np.ndarray, splits: dict, batch_size: int,
    W: np.ndarray | None = None,
):
    loaders = {}
    for name in ("train", "val", "test"):
        idx = index_by_id(ids, splits[name])
        x = torch.from_numpy(X[idx]).float()
        y = torch.from_numpy(Y[idx]).float()
        if W is not None:
            w = torch.from_numpy(W[idx]).float()
            ds = TensorDataset(x, y, w)
        else:
            ds = TensorDataset(x, y)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"))
    return loaders


def _run_epoch(model, loader, device, optimizer=None, weighted: bool = False):
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    preds: list[np.ndarray] = []
    targs: list[np.ndarray] = []
    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for batch in loader:
            if weighted:
                xb, yb, wb = batch
                xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
                pred = model(xb)
                # Weighted MSE per-element; weights already normalized to mean 1.
                loss = (wb * (pred - yb).pow(2)).mean()
            else:
                xb, yb = batch
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = (pred - yb).pow(2).mean()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            preds.append(pred.detach().cpu().numpy())
            targs.append(yb.detach().cpu().numpy())
    return (
        float(np.mean(losses)),
        np.concatenate(preds),
        np.concatenate(targs),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--ids", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--splits", type=Path, default=Path("artifacts/splits.json"))
    ap.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stats", type=Path, default=None,
                    help="path to attribute_stats.npz for variance-weighted loss "
                         "(if set, loss is per-cell weighted by 1/(std/median_std)+eps)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    X = np.load(args.features)
    feature_ids = np.load(args.ids)
    ids, Y, attr_names = load_labels(args.labels)

    # Align features to labels by stimulus id
    id_to_row = {int(i): k for k, i in enumerate(ids)}
    order = np.array([id_to_row[int(i)] for i in feature_ids], dtype=np.int64)
    Y_aligned = Y[order]
    ids_aligned = ids[order]
    assert X.shape[0] == Y_aligned.shape[0] == ids_aligned.shape[0]

    if args.splits.exists():
        splits = load_splits(args.splits)
        print(f"Loaded splits from {args.splits}")
    else:
        splits = make_splits(ids_aligned, seed=args.seed)
        save_splits(splits, args.splits)
        print(f"Wrote splits to {args.splits}")

    W_aligned = None
    if args.stats is not None:
        stats = np.load(args.stats, allow_pickle=True)
        s_ids = stats["ids"]; std = stats["std"]
        # Reorder to match ids_aligned (stimulus order of the features file)
        pos = {int(s): k for k, s in enumerate(s_ids)}
        idx = np.array([pos[int(s)] for s in ids_aligned], dtype=np.int64)
        std = std[idx]  # (N, 34) in feature-id order
        med = float(np.nanmedian(std[std > 0]))
        # Normalize so mean weight across cells is 1 — scale-invariant
        w_raw = med / (std + 1e-6)
        w_raw = np.nan_to_num(w_raw, nan=1.0)  # no-data cells → default weight 1
        w_raw = w_raw * (w_raw.size / w_raw.sum())
        W_aligned = w_raw.astype(np.float32)
        print(f"Variance-weighted loss enabled: weight range [{W_aligned.min():.3f}, {W_aligned.max():.3f}], mean={W_aligned.mean():.3f}")

    loaders = _make_loaders(X, Y_aligned, ids_aligned, splits, args.batch_size, W=W_aligned)

    hidden = None if args.head == "linear" else args.hidden
    model = TraitHead(
        in_dim=X.shape[1], out_dim=Y.shape[1], hidden=hidden, dropout=args.dropout
    ).to(device)
    print(f"Head: {args.head} (hidden={hidden}), params={sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    weighted = W_aligned is not None

    best_val_r = -math.inf
    best_state = None
    epochs_no_improve = 0
    log_lines: list[str] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = _run_epoch(model, loaders["train"], device, optimizer, weighted=weighted)
        val_loss, val_pred, val_true = _run_epoch(model, loaders["val"], device, None, weighted=weighted)
        val_rs, _ = per_attribute_pearson(val_true, val_pred)
        val_mean_r = float(np.mean(val_rs))
        scheduler.step()

        line = (
            f"epoch {epoch:3d}/{args.epochs} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_mean_r={val_mean_r:.4f}"
        )
        print(line)
        log_lines.append(line)

        if val_mean_r > best_val_r + 1e-6:
            best_val_r = val_mean_r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stop at epoch {epoch} (best val_mean_r={best_val_r:.4f})")
                break

    assert best_state is not None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "config": {
            "in_dim": int(X.shape[1]),
            "out_dim": int(Y.shape[1]),
            "hidden": hidden,
            "dropout": args.dropout,
            "head": args.head,
            "backbone": "dinov2_vitb14",
            "attr_names": attr_names,
            "seed": args.seed,
            "best_val_mean_r": best_val_r,
        },
    }
    torch.save(ckpt, args.out)
    log_path = args.out.with_suffix(".log")
    log_path.write_text("\n".join(log_lines) + "\n")
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(ckpt["config"], indent=2))
    print(f"Saved checkpoint -> {args.out}")
    print(f"Saved log        -> {log_path}")


if __name__ == "__main__":
    main()
