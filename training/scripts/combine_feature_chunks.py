"""Concatenate per-chunk DINOv2 feature files into a single features + ids array.

Expects chunk_*.npy (features, shape [n_i, D]) and chunk_*.ids.npy (ids, shape [n_i]).

Example:
    python -m scripts.combine_feature_chunks \
        --chunk-dir $SCRATCH/omi/features/vitg_chunks \
        --out       $SCRATCH/omi/features/features_vitg14.npy \
        --ids-out   $SCRATCH/omi/features/stimulus_ids_vitg14.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ids-out", required=True, type=Path)
    args = ap.parse_args()

    feat_files = sorted(args.chunk_dir.glob("chunk_*.npy"))
    feat_files = [p for p in feat_files if not p.name.endswith(".ids.npy")]
    if not feat_files:
        raise SystemExit(f"no chunk_*.npy files under {args.chunk_dir}")

    feats = []
    ids = []
    for fp in feat_files:
        ip = fp.with_suffix(".ids.npy")
        if not ip.exists():
            raise SystemExit(f"missing ids file for {fp}: {ip}")
        feats.append(np.load(fp))
        ids.append(np.load(ip))
        print(f"  {fp.name}: {feats[-1].shape}")

    feats = np.concatenate(feats, axis=0)
    ids = np.concatenate(ids, axis=0)
    # Sort by id for stability with the label loader
    order = np.argsort(ids)
    feats = feats[order]
    ids = ids[order]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, feats)
    np.save(args.ids_out, ids)
    print(f"Wrote {feats.shape} -> {args.out}")
    print(f"Wrote {ids.shape} -> {args.ids_out}")


if __name__ == "__main__":
    main()
