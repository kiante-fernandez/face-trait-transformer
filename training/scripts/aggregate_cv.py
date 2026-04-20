"""Aggregate a 10-fold CV sweep: for each fold pick val-top-k heads, average
their predictions on the fold's test stimuli, concatenate across folds, and
compute per-attribute R² (the metric Peterson et al. 2022 Fig. 2 reports).

Expects the directory layout produced by hoffman2/run_sweep_task_cv.py:

    cv_out/
        fold_00/
            task001_....pt, .result.json, .meta.json
            task001_....test.csv      (from scripts.eval)
            ...
        fold_01/
            ...

All 1004 stimuli show up exactly once as test predictions across the 10 folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from face_trait_transformer.data import load_labels, load_splits
from face_trait_transformer.model import TraitHead


def _group_predict(ckpt_dir: Path, X_test: np.ndarray, top_k: int):
    """Load the top-k heads (by val_mean_r) and average their test predictions."""
    scored = []
    for rj in sorted(ckpt_dir.glob("*.result.json")):
        d = json.loads(rj.read_text())
        ck = ckpt_dir / Path(d["checkpoint"]).name
        if not ck.exists():
            continue
        scored.append((d["val_mean_r"], ck))
    scored.sort(reverse=True)
    preds = []
    for _, p in scored[:top_k]:
        m = torch.load(p, map_location="cpu", weights_only=False)
        c = m["config"]
        head = TraitHead(in_dim=c["in_dim"], out_dim=c["out_dim"],
                         hidden=c["hidden"], dropout=c["dropout"])
        head.load_state_dict(m["state_dict"])
        head.eval()
        with torch.inference_mode():
            preds.append(head(torch.from_numpy(X_test).float()).numpy())
    return np.stack(preds).mean(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv-out", required=True, type=Path,
                    help="directory containing fold_00/, fold_01/, …")
    ap.add_argument("--splits-dir", required=True, type=Path,
                    help="directory with fold_XX.json")
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--ids", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-csv", type=Path, default=Path("artifacts/cv_per_attribute.csv"))
    args = ap.parse_args()

    ids, Y, attr = load_labels(args.labels)
    id_to_row = {int(i): k for k, i in enumerate(ids)}

    X_all = np.load(args.features)
    feat_ids = np.load(args.ids)
    fpos = {int(i): k for k, i in enumerate(feat_ids)}

    all_true = []
    all_pred = []
    for fold_dir in sorted(args.cv_out.glob("fold_*")):
        fold = int(fold_dir.name.split("_")[1])
        splits_path = args.splits_dir / f"fold_{fold:02d}.json"
        splits = json.loads(splits_path.read_text())
        test_ids = [int(s) for s in splits["test"]]
        X_test = np.stack([X_all[fpos[s]] for s in test_ids])
        Y_test = np.stack([Y[id_to_row[s]] for s in test_ids]) * 100

        y_hat = _group_predict(fold_dir, X_test, args.top_k) * 100
        y_hat = np.clip(y_hat, 0, 100)
        all_true.append(Y_test)
        all_pred.append(y_hat)
        print(f"fold {fold:02d}: {len(test_ids)} test  "
              f"mean_r={np.mean([pearsonr(Y_test[:, j], y_hat[:, j])[0] for j in range(34)]):.4f}")

    Y_true = np.concatenate(all_true, axis=0)
    Y_pred = np.concatenate(all_pred, axis=0)
    print(f"\nConcatenated: {Y_true.shape[0]} stimuli (expected 1004)")

    rows = []
    for j, name in enumerate(attr):
        r, _ = pearsonr(Y_true[:, j], Y_pred[:, j])
        ss_res = ((Y_true[:, j] - Y_pred[:, j]) ** 2).sum()
        ss_tot = ((Y_true[:, j] - Y_true[:, j].mean()) ** 2).sum()
        rows.append({
            "attribute": name,
            "pearson_r": r,
            "R2": 1 - ss_res / ss_tot,
            "mae": np.mean(np.abs(Y_true[:, j] - Y_pred[:, j])),
            "rmse": np.sqrt(np.mean((Y_true[:, j] - Y_pred[:, j]) ** 2)),
        })
    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(f"MEAN  pearson_r={df.pearson_r.mean():.3f}  R2={df.R2.mean():.3f}")
    print(f"MEDIAN pearson_r={df.pearson_r.median():.3f}  R2={df.R2.median():.3f}")
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
