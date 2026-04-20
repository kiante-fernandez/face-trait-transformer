"""Apply the OMI trait model to the GIRAF aging-faces dataset.

Discovers images under a root directory, extracts metadata from the path
(age group, emotion, demographic code, gender, subject id) and produces a
single CSV with per-image predictions plus those metadata columns.

Expected layout (GIRAF):
    <root>/GIRAF_emotion_expressions/core_set/images/{young,middle,old}/
           {happy,sad,fear,surprise,neutral,angry,disgust}/<code>_<g><id>_<emotion>.png
    <root>/GIRAF_original_images/{Younger Adults, Middle-aged Adults, Older Adults}/
           <code>_<g><id>.png

Example:
    python -m scripts.apply_to_aging \
        --root /Users/kiante/Downloads/aging_images \
        --bundle artifacts/bundle \
        --out aging_validation/predictions.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from face_trait_transformer import TraitPredictor


# Demographic code → human-readable label used in the file naming convention.
_DEMOGRAPHIC = {
    "af": "african",
    "as": "asian",
    "in": "indian",
    "lat": "latin",
    "me": "middle_eastern",
    "wh": "white",
}


def _parse_metadata(image_path: Path, root: Path) -> dict:
    """Extract (age_group, emotion, demographic, gender, subject) from path."""
    rel = image_path.relative_to(root)
    parts = [p.lower() for p in rel.parts]
    stem = image_path.stem.lower()

    # age group
    age_group = None
    for p in parts:
        if p in ("young", "middle", "old"):
            age_group = p
            break
        if p.startswith("younger"):
            age_group = "young"; break
        if p.startswith("middle"):
            age_group = "middle"; break
        if p.startswith("older"):
            age_group = "old"; break

    # emotion (only present for the emotion-expressions branch)
    emotion = None
    for e in ("happy", "sad", "fear", "surprise", "neutral", "angry", "disgust"):
        if e in parts:
            emotion = e
            break

    # demographic + gender + subject-id from filename
    demographic = gender = subject_id = None
    m = re.match(r"([a-z]+)_([fm])(\d+)(?:_([a-z]+))?$", stem)
    if m:
        demographic = _DEMOGRAPHIC.get(m.group(1), m.group(1))
        gender = {"f": "female", "m": "male"}[m.group(2)]
        subject_id = m.group(3)
    return {
        "age_group": age_group,
        "emotion": emotion,
        "demographic": demographic,
        "gender": gender,
        "subject_id": subject_id,
        "relative_path": str(rel),
    }


def _collect_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    # Only include the two documented branches so we skip reference/replication folders.
    include_subdirs = [
        "GIRAF_emotion_expressions/core_set/images",
        "GIRAF_original_images/Younger Adults",
        "GIRAF_original_images/Middle-aged Adults",
        "GIRAF_original_images/Older Adults",
    ]
    for sub in include_subdirs:
        d = root / sub
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("._"):
                paths.append(p)
    return sorted(paths)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="batch size for prediction (watch memory on large backbones)")
    ap.add_argument("--tta", action="store_true",
                    help="enable horizontal-flip TTA (doubles inference time)")
    args = ap.parse_args()

    paths = _collect_images(args.root)
    if not paths:
        raise SystemExit(f"no images found under {args.root}")
    print(f"Found {len(paths)} images. Parsing metadata...")

    meta_rows = []
    for p in paths:
        meta = _parse_metadata(p, args.root)
        meta["filename"] = str(p)
        meta_rows.append(meta)
    meta_df = pd.DataFrame(meta_rows)
    # Rename so the metadata gender doesn't collide with the predictor's `gender` attribute column.
    meta_df = meta_df.rename(columns={"gender": "gender_coded"})
    missing_age = (meta_df.age_group.isna()).sum()
    missing_demo = (meta_df.demographic.isna()).sum()
    print(f"  parsed {len(meta_df)} rows; "
          f"age_group missing in {missing_age}, demographic missing in {missing_demo}")

    print(f"Loading bundle {args.bundle}...")
    predictor = TraitPredictor.from_bundle(args.bundle)

    print(f"Predicting (tta={args.tta}, batch_size={args.batch_size})...")
    preds = predictor.predict(
        [str(p) for p in paths],
        batch_size=args.batch_size,
        tta=args.tta,
        return_dataframe=True,
    )
    # predictor returns a DataFrame with 'filename' and 34 attribute cols
    preds = preds.drop(columns=["filename"])
    attr_cols = list(preds.columns)
    out = pd.concat([meta_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {out.shape[0]} rows × {out.shape[1]} cols -> {args.out}")
    print(f"Attribute columns: {attr_cols[:5]}…")


if __name__ == "__main__":
    main()
