"""Build a self-contained inference bundle (heads + manifest) for HF Hub.

Layout:
    bundle/
        manifest.json
        README.md                     (this file's content)
        METHODS.md                    (copied from docs/methods.md)
        head_<group>/*.pt             top-k heads for each frozen backbone
        finetune/*.pt                 optional end-to-end fine-tune checkpoints

The code to load this bundle lives in the `face-trait-transformer` Python
package (`pip install face-trait-transformer`).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from face_trait_transformer.data import load_labels


_BUNDLE_README = """# face-trait-transformer — model bundle

This folder holds the trained weights for the `face-trait-transformer` model.
Code to load and run the bundle lives in the Python package of the same name.

## Install the package

    pip install face-trait-transformer[hub]

## Use the bundle

Either point at this local directory:

    from face_trait_transformer import TraitPredictor
    predictor = TraitPredictor.from_bundle("path/to/this/folder")

…or pull it from HuggingFace Hub (recommended):

    predictor = TraitPredictor.from_pretrained("kiante/face-trait-transformer")

Then:

    row = predictor.predict("face.jpg")           # pandas.Series (34 cols, 0–100)
    df  = predictor.predict(["a.jpg", "b.jpg"])   # pandas.DataFrame

On first run, torch.hub downloads the DINOv2 backbone weights to
`~/.cache/torch/hub/checkpoints/` (~1.2 GB for ViT-G/14).

## What's inside

- `manifest.json` — attribute names, backbone groups, head file list.
- `head_<group>/*.pt` — the top-k MLP head checkpoints per backbone, selected
  by validation Pearson r.
- `finetune/*.pt` — optional end-to-end fine-tune checkpoints (self-contained
  module state; load through `TraitPredictor`, not directly).
- `METHODS.md` — the paper-ready methods section.

## License

Weights are distributed under **Creative Commons BY-NC-SA 4.0**, inherited
from the underlying One Million Impressions dataset (Peterson et al., 2022).
Non-commercial use only; attribute this project and the OMI paper; derivatives
must use the same license.

## Important caveat

Predictions reflect **learned human perceptions / rater stereotypes**, not
ground-truth attributes of the depicted people. This is especially important
for demographic and socially-constructed attributes (asian, black, white,
middle-eastern, hispanic, islander, native, gay, liberal, privileged, godly,
electable, looks-like-you). Frame outputs as "perceived X", not "X".
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Repeatable pairs via dest lists — parse manually.
    ap.add_argument("--backbone", action="append", default=[],
                    help="DINOv2 hub name, e.g. dinov2_vitl14 (can be repeated)")
    ap.add_argument("--ckpt-dir", action="append", default=[], type=Path,
                    help="directory with .pt + matching .result.json (can be repeated)")
    ap.add_argument("--top-k", action="append", default=None,
                    help="top-k heads to include per backbone (can be repeated; default 10 each)")
    ap.add_argument("--image-size", action="append", default=None,
                    help="image size for each backbone (can be repeated; default 224 each)")
    ap.add_argument("--group-name", action="append", default=None,
                    help="custom group key per backbone; defaults to backbone hub name")
    ap.add_argument("--finetune", action="append", default=[], type=Path,
                    help="finetune checkpoint file to include (can be repeated)")
    ap.add_argument("--labels", required=True, type=Path,
                    help="attribute_means.csv to pull canonical attribute names")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if len(args.backbone) != len(args.ckpt_dir):
        raise SystemExit("--backbone and --ckpt-dir must be paired (same count)")
    top_ks = [int(k) for k in (args.top_k or [])]
    while len(top_ks) < len(args.backbone):
        top_ks.append(10)
    image_sizes = [int(s) for s in (args.image_size or [])]
    while len(image_sizes) < len(args.backbone):
        image_sizes.append(224)
    group_names = list(args.group_name or [])
    while len(group_names) < len(args.backbone):
        group_names.append(args.backbone[len(group_names)])

    _, _, attr_names = load_labels(args.labels)
    args.out.mkdir(parents=True, exist_ok=True)

    # Ship the paper-ready methods writeup with the weights.
    repo_root = Path(__file__).resolve().parents[2]
    methods_src = repo_root / "docs" / "methods.md"
    if methods_src.exists():
        shutil.copy2(methods_src, args.out / "METHODS.md")
    # A minimal README for the HF/bundle distribution.
    readme = args.out / "README.md"
    readme.write_text(_BUNDLE_README)

    manifest = {"attr_names": attr_names, "backbones": {}, "bundle_version": 1}

    for bname, ck_dir, k, image_size, gname in zip(
        args.backbone, args.ckpt_dir, top_ks, image_sizes, group_names
    ):
        rjs = sorted(ck_dir.glob("*.result.json"))
        if not rjs:
            raise SystemExit(f"no *.result.json under {ck_dir}")
        scored = []
        for rj in rjs:
            d = json.loads(rj.read_text())
            local = ck_dir / Path(d["checkpoint"]).name
            if local.exists():
                scored.append((d["val_mean_r"], local))
        scored.sort(reverse=True)
        chosen = [p for _, p in scored[:k]]

        head_subdir = args.out / f"head_{gname}"
        head_subdir.mkdir(exist_ok=True)
        head_files: list[str] = []
        for p in chosen:
            dest = head_subdir / p.name
            shutil.copy2(p, dest)
            head_files.append(f"head_{gname}/{p.name}")

        manifest["backbones"][gname] = {
            "base_model": bname,
            "image_size": image_size,
            "top_k": k,
            "head_files": head_files,
        }
        print(f"  {gname} (base={bname}, {image_size}px): copied {len(chosen)} heads -> {head_subdir}")

    if args.finetune:
        ft_dir = args.out / "finetune"
        ft_dir.mkdir(exist_ok=True)
        ft_files: list[str] = []
        for src_pt in args.finetune:
            dest = ft_dir / src_pt.name
            shutil.copy2(src_pt, dest)
            ft_files.append(f"finetune/{src_pt.name}")
            print(f"  finetune: copied {src_pt.name} -> {dest}")
        manifest["finetune_files"] = ft_files

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # README was already written above via _BUNDLE_README; keep behavior idempotent.
    if not (args.out / "README.md").exists():
        (args.out / "README.md").write_text(_BUNDLE_README)

    total_size = sum(p.stat().st_size for p in args.out.rglob("*.pt"))
    print(f"Wrote bundle -> {args.out}  ({total_size / 1e6:.1f} MB of weights)")


if __name__ == "__main__":
    main()
