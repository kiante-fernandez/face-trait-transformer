"""Evaluate a trained trait-regression head on a held-out split.

Example:
    python -m scripts.eval --ckpt artifacts/checkpoints/mlp_v1.pt --split test \
        --features artifacts/features.npy --ids artifacts/stimulus_ids.npy \
        --labels attribute_means.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from face_trait_transformer.data import index_by_id, load_labels, load_splits
from face_trait_transformer.features import pick_device
from face_trait_transformer.metrics import summary
from face_trait_transformer.model import TraitHead


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--ids", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--splits", type=Path, default=Path("artifacts/splits.json"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TraitHead(
        in_dim=cfg["in_dim"],
        out_dim=cfg["out_dim"],
        hidden=cfg["hidden"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X = np.load(args.features)
    feature_ids = np.load(args.ids)
    ids, Y, attr_names = load_labels(args.labels)
    assert attr_names == cfg["attr_names"], "attribute order mismatch between csv and checkpoint"

    id_to_row = {int(i): k for k, i in enumerate(ids)}
    order = np.array([id_to_row[int(i)] for i in feature_ids], dtype=np.int64)
    Y_aligned = Y[order]
    ids_aligned = ids[order]

    splits = load_splits(args.splits)
    idx = index_by_id(ids_aligned, splits[args.split])
    x = torch.from_numpy(X[idx]).float().to(device)
    y_true = Y_aligned[idx]

    with torch.inference_mode():
        y_pred = model(x).cpu().numpy()

    df = summary(y_true, y_pred, attr_names)
    df = df.sort_values("pearson_r", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"Mean pearson_r over 34 attributes: {df['pearson_r'].mean():.4f}")
    print(f"Median pearson_r:                  {df['pearson_r'].median():.4f}")

    out_csv = args.ckpt.with_suffix(f".{args.split}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
