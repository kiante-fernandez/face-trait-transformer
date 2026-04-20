"""Aggregate the 1.4M raw trial-level ratings into per-(stim, attr) statistics.

Produces an artifact used by the variance-weighted training loss:

    attribute_stats.npz
        ids        : (N,)        stimulus ids (sorted ascending, matches images/)
        mean       : (N, 34)     per-cell mean rating (0-100)
        std        : (N, 34)     per-cell sample std across raters
        n_raters   : (N, 34)     per-cell rater count

For cells with no raw data (shouldn't happen on the main experiment) fall back
to attribute_means.csv with std=NaN and n=0 so the weighted loss can skip them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratings", type=Path, default=Path("attribute_ratings.csv"))
    ap.add_argument("--labels", type=Path, default=Path("attribute_means.csv"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/attribute_stats.npz"))
    args = ap.parse_args()

    df_raw = pd.read_csv(args.ratings)
    # Keep only main-experiment stimuli (1..1004, numeric) — drop validation ids.
    df_raw = df_raw[pd.to_numeric(df_raw["stimulus"], errors="coerce").notna()]
    df_raw = df_raw.assign(stimulus=df_raw["stimulus"].astype(int))
    df_raw = df_raw[df_raw["stimulus"].between(1, 1004)]
    print(f"Kept {len(df_raw)} raw trials across {df_raw.stimulus.nunique()} stimuli")

    df_means = pd.read_csv(args.labels)
    attr_names = [c for c in df_means.columns if c != "stimulus"]
    ids = np.sort(df_means["stimulus"].to_numpy(dtype=np.int64))
    id_idx = {int(s): k for k, s in enumerate(ids)}
    attr_idx = {a: k for k, a in enumerate(attr_names)}

    N = len(ids); A = len(attr_names)
    mean = np.full((N, A), np.nan, dtype=np.float32)
    std = np.full((N, A), np.nan, dtype=np.float32)
    nr = np.zeros((N, A), dtype=np.int32)

    grouped = df_raw.groupby(["stimulus", "attribute"])["rating"]
    for (sid, attr), series in grouped:
        if attr not in attr_idx or int(sid) not in id_idx:
            continue
        i = id_idx[int(sid)]; j = attr_idx[attr]
        mean[i, j] = float(series.mean())
        std[i, j] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        nr[i, j] = int(len(series))

    # Cross-check with attribute_means.csv (they may differ slightly if fill-in trials treated differently)
    csv_mean = df_means.set_index("stimulus").loc[ids, attr_names].to_numpy(dtype=np.float32)
    diff = np.nanmean(np.abs(mean - csv_mean))
    print(f"Mean |raw-mean - csv-mean| = {diff:.3f} on a 0-100 scale")

    coverage = np.mean(nr > 0) * 100
    print(f"Cell coverage: {coverage:.1f}%  (mean n_raters={np.nanmean(nr[nr>0]):.1f})")
    print(f"Median per-cell std across attributes: "
          f"{np.nanmedian(std[nr > 0]):.2f}   (iqr 25-75 = "
          f"{np.nanpercentile(std[nr>0], 25):.2f}-"
          f"{np.nanpercentile(std[nr>0], 75):.2f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=ids, mean=mean, std=std, n_raters=nr,
             attr_names=np.array(attr_names))
    print(f"Wrote {args.out}  shape mean={mean.shape} std={std.shape}")


if __name__ == "__main__":
    main()
