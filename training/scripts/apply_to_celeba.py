"""Apply the OMI trait model to the CelebA external-validity set.

Dataset layout (from the original CelebA validation release):
    <root>/CelebA/*.jpg                     500 images
    <root>/FaceImageIndex.csv               ImageName -> FaceIndex (1..500)
    <root>/CelebA_Image_Code_new.mat        im_code: (1, 500) identity labels (1..50)

The ordering in im_code matches the order of files as listed in FaceImageIndex.csv
(which is the filesystem enumeration order used by the original experiment).

Output: a CSV `predictions.csv` with per-image predictions plus identity metadata.

Example:
    python -m training.scripts.apply_to_celeba \
        --root /Users/kiante/Downloads/CelebA-validation \
        --out  examples/celeba_validation/predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from face_trait_transformer import TraitPredictor


def _load_identity_labels(root: Path) -> pd.DataFrame:
    """Return DataFrame(image_name, face_index, identity) for all 500 stimuli."""
    index_csv = root / "FaceImageIndex.csv"
    mat = root / "CelebA_Image_Code_new.mat"
    if not index_csv.exists() or not mat.exists():
        raise SystemExit(f"missing {index_csv} or {mat}")
    df = pd.read_csv(index_csv).rename(columns={"ImageName": "image_name", "FaceIndex": "face_index"})
    im_code = sio.loadmat(mat)["im_code"].flatten().astype(int)
    if len(im_code) != len(df):
        raise SystemExit(f"length mismatch: FaceImageIndex has {len(df)} rows, im_code has {len(im_code)}")
    df["identity"] = im_code
    return df.sort_values("face_index").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bundle", type=Path, default=None,
                    help="optional local bundle; otherwise pulls from HF Hub via from_pretrained()")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tta", action="store_true",
                    help="enable horizontal-flip TTA (doubles inference time)")
    args = ap.parse_args()

    meta = _load_identity_labels(args.root)
    image_dir = args.root / "CelebA"
    paths: list[Path] = []
    for name in meta["image_name"]:
        p = image_dir / name
        if not p.exists():
            raise SystemExit(f"image missing: {p}")
        paths.append(p)
    print(f"Loaded {len(paths)} images across {meta.identity.nunique()} identities "
          f"(expected 10 per identity; actual counts range "
          f"{meta.identity.value_counts().min()}..{meta.identity.value_counts().max()}).")

    if args.bundle is not None:
        predictor = TraitPredictor.from_bundle(args.bundle)
    else:
        predictor = TraitPredictor.from_pretrained()

    print(f"Predicting (tta={args.tta}, batch_size={args.batch_size})...")
    preds = predictor.predict(
        [str(p) for p in paths],
        batch_size=args.batch_size,
        tta=args.tta,
        return_dataframe=True,
        progress=True,
    )
    preds = preds.drop(columns=["filename"])
    out = pd.concat([meta.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {out.shape[0]} rows × {out.shape[1]} cols -> {args.out}")


if __name__ == "__main__":
    main()
