"""Batch inference over a directory of faces.

Run:
    python examples/batch_inference.py path/to/faces_dir path/to/out.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

from face_trait_transformer import TraitPredictor


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python batch_inference.py <faces_dir> <out.csv>")
        return 1
    faces_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in faces_dir.rglob("*")
                   if p.suffix.lower() in exts and not p.name.startswith("._"))
    if not paths:
        raise SystemExit(f"no images found under {faces_dir}")
    print(f"Running TraitPredictor on {len(paths)} images...")

    predictor = TraitPredictor.from_pretrained()  # defaults to kiante/face-trait-transformer
    df = predictor.predict([str(p) for p in paths], batch_size=8, tta=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {df.shape[0]} × {df.shape[1]} → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
