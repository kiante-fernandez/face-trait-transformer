"""`ftt` command-line interface.

Subcommands:
    ftt predict IMAGE_OR_DIR [--bundle PATH | --repo HF_REPO] [--figure PATH] [--out CSV]
    ftt download              [--repo HF_REPO]               # pre-fetch the bundle to cache
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _collect_images(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(q for q in p.rglob("*") if q.suffix.lower() in exts and not q.name.startswith("._"))
    raise SystemExit(f"input not found: {p}")


def _cmd_predict(args: argparse.Namespace) -> int:
    from . import TraitPredictor
    paths = _collect_images(args.input)
    if args.bundle is not None:
        predictor = TraitPredictor.from_bundle(args.bundle)
    else:
        predictor = TraitPredictor.from_pretrained(args.repo)

    if args.figure and len(paths) == 1:
        row, _ = predictor.predict_with_figure(paths[0], out_path=args.figure)
        print(row.to_string())
        print(f"wrote {args.figure}")
        if args.out is not None:
            df = row.to_frame().T
            df.insert(0, "filename", [str(paths[0])])
            df.to_csv(args.out, index=False)
            print(f"wrote {args.out}")
        return 0

    df = predictor.predict([str(p) for p in paths], return_dataframe=True, tta=not args.no_tta)
    out = args.out or Path("predictions.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}  shape={df.shape}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from .hub import download_from_hub
    local = download_from_hub(repo_id=args.repo)
    print(f"Bundle cached at: {local}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ftt", description="face-trait-transformer CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_predict = sub.add_parser("predict", help="run inference on one image or a directory")
    p_predict.add_argument("input", type=Path, help="image file or directory")
    p_predict.add_argument("--bundle", type=Path, default=None,
                           help="local bundle directory (overrides --repo)")
    p_predict.add_argument("--repo", type=str, default="kiante/face-trait-transformer",
                           help="HuggingFace Hub repo id for the bundle")
    p_predict.add_argument("--figure", type=Path, default=None,
                           help="save 3-panel diagnostic figure (single-image only)")
    p_predict.add_argument("--out", type=Path, default=None,
                           help="output CSV path (default predictions.csv for directories)")
    p_predict.add_argument("--no-tta", action="store_true",
                           help="disable horizontal-flip TTA (~2× faster)")
    p_predict.set_defaults(func=_cmd_predict)

    p_download = sub.add_parser("download", help="pre-download the HF Hub bundle to cache")
    p_download.add_argument("--repo", type=str, default="kiante/face-trait-transformer")
    p_download.set_defaults(func=_cmd_download)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
