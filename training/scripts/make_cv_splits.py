"""Generate 10 deterministic CV splits for the 1,004 OMI stimuli.

Each splits file contains train/val/test where test is the held-out fold,
val is 100 stimuli sampled from the remaining 904, train is the remaining 804.

Writes artifacts/cv_splits/fold_<k>.json for k in 0..9.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-stim", type=int, default=1004)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/cv_splits"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(args.n_stim) + 1  # stimulus ids are 1..N
    # Split perm into 10 roughly-equal folds: 4 of size 101, 6 of size 100
    folds = []
    start = 0
    fold_sizes = [101] * (args.n_stim % args.n_folds) + [args.n_stim // args.n_folds] * (
        args.n_folds - (args.n_stim % args.n_folds)
    )
    for sz in fold_sizes:
        folds.append(perm[start : start + sz].tolist())
        start += sz
    assert sum(len(f) for f in folds) == args.n_stim

    for k in range(args.n_folds):
        test = sorted(folds[k])
        rest = [sid for j, f in enumerate(folds) if j != k for sid in f]
        # Sample val from rest deterministically
        rng_k = np.random.default_rng(args.seed + 1000 + k)
        rest_perm = rng_k.permutation(len(rest))
        val = sorted([rest[i] for i in rest_perm[: args.val_size]])
        train = sorted([rest[i] for i in rest_perm[args.val_size :]])
        out = {
            "train": train,
            "val": val,
            "test": test,
            "seed": args.seed,
            "fold": k,
            "n_folds": args.n_folds,
        }
        p = args.out_dir / f"fold_{k:02d}.json"
        p.write_text(json.dumps(out, indent=2))
        print(f"fold {k}: train={len(train)} val={len(val)} test={len(test)}  -> {p}")


if __name__ == "__main__":
    main()
