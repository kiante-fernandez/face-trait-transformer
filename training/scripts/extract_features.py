"""Extract DINOv2 CLS features for every image in a directory and cache to disk.

Example:
    python -m scripts.extract_features \
        --images-dir images --out artifacts/features.npy \
        --ids-out artifacts/stimulus_ids.npy
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from face_trait_transformer.features import build_transform, extract_cls, load_dinov2, pick_device


def _parse_int_id(path: Path) -> int | None:
    m = re.fullmatch(r"(\d+)", path.stem)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ids-out", type=Path, default=None)
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=224,
                    help="crop size fed to the backbone (default 224; DINOv2 also trained at 518)")
    ap.add_argument("--id-range", type=str, default=None,
                    help="slice the sorted image list as START:END (0-indexed, end exclusive), e.g. 0:32")
    args = ap.parse_args()

    paths = [
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() == ".jpg" and not p.name.startswith("._")
    ]
    if not paths:
        raise SystemExit(f"no .jpg files under {args.images_dir}")

    int_ids = [_parse_int_id(p) for p in paths]
    if all(i is not None for i in int_ids):
        order = np.argsort(int_ids)
        paths = [paths[i] for i in order]
        ids = np.array([int_ids[i] for i in order], dtype=np.int64)
    else:
        paths = sorted(paths)
        ids = np.array([p.stem for p in paths])  # string ids, best effort

    if args.id_range:
        s, e = args.id_range.split(":")
        s = int(s); e = int(e)
        paths = paths[s:e]
        ids = ids[s:e]
        if not paths:
            raise SystemExit(f"--id-range {args.id_range} produced empty slice")

    print(f"Loading {args.model}...")
    model, _default_transform, device = load_dinov2(args.model)
    transform = build_transform(image_size=args.image_size)
    print(f"Using device: {device}. Encoding {len(paths)} images at {args.image_size}px...")

    feats = extract_cls(
        model, transform, paths, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, feats)
    ids_out = args.ids_out or args.out.with_name("stimulus_ids.npy")
    np.save(ids_out, ids)
    print(f"Saved features {feats.shape} -> {args.out}")
    print(f"Saved ids      {ids.shape}     -> {ids_out}")


if __name__ == "__main__":
    main()
