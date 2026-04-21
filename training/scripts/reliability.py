"""Compute per-attribute split-half reliability from the raw OMI trial ratings.

Matches Peterson et al. 2022 Fig. 2's red reliability markers:

> we computed the split-half reliability for each attribute, averaging the
> squared correlations between the averages of 100 random splits of the
> ratings for each map.

So for each attribute, we randomly partition the raters of each (stimulus, attribute)
cell into two halves, compute the per-stimulus mean in each half, correlate those
two N-length vectors across stimuli, square it, and average the squared correlation
over random partitions. This is the "ceiling" R² a model could achieve if it were
only trying to predict the mean of half the raters from the mean of the other half.

Output: a CSV with columns [attribute, split_half_R2, split_half_r, n_iters].

Example:
    python -m training.scripts.reliability \
        --ratings attribute_ratings.csv \
        --out     training/results/split_half_reliability.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def compute(ratings_csv: Path, n_iters: int = 100, seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(ratings_csv)
    # Keep only main-experiment stimuli (1..1004) with numeric ids.
    df = df[pd.to_numeric(df["stimulus"], errors="coerce").notna()]
    df = df.assign(stimulus=df["stimulus"].astype(int))
    df = df[df["stimulus"].between(1, 1004)]

    attrs = sorted(df["attribute"].unique())
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for attr in attrs:
        sub = df[df["attribute"] == attr]
        groups = sub.groupby("stimulus")["rating"].apply(list).to_dict()
        stimuli = sorted(groups.keys())
        rsq_list: list[float] = []
        for _ in range(n_iters):
            a_means, b_means = [], []
            for s in stimuli:
                ratings = np.asarray(groups[s], dtype=float)
                rng.shuffle(ratings)
                mid = len(ratings) // 2
                if mid < 2:  # not enough raters for a split
                    a_means.append(np.nan)
                    b_means.append(np.nan)
                else:
                    a_means.append(ratings[:mid].mean())
                    b_means.append(ratings[mid:2 * mid].mean())
            a = np.asarray(a_means); b = np.asarray(b_means)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 10:
                continue
            r_half, _ = pearsonr(a[mask], b[mask])
            rsq_list.append(r_half ** 2)
        mean_r2 = float(np.mean(rsq_list)) if rsq_list else float("nan")
        mean_r = float(np.sqrt(mean_r2)) if np.isfinite(mean_r2) else float("nan")
        rows.append({
            "attribute": attr,
            "split_half_R2": mean_r2,
            "split_half_r": mean_r,
            "n_iters": len(rsq_list),
        })
        print(f"  {attr:<18} R2={mean_r2:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratings", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Computing split-half reliability with {args.n_iters} iterations...")
    df = compute(args.ratings, n_iters=args.n_iters, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print()
    print(df.sort_values("split_half_r", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
