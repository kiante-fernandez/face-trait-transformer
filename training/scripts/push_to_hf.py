"""Push a local bundle to HuggingFace Hub as the face-trait-transformer model.

Before first use, authenticate once:

    pip install huggingface_hub
    huggingface-cli login        # paste a write-scope token from https://huggingface.co/settings/tokens

Then:

    python -m training.scripts.push_to_hf \
        --bundle training/bundle \
        --repo   kiante/face-trait-transformer \
        --model-card docs/model_card.md

This will:
  1. Create the repo on HF Hub (no-op if it exists).
  2. Upload the bundle folder.
  3. Write the model card as the repo-root README.md.

Re-run whenever you rebuild the bundle; HF Hub keeps the history.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--repo", default="kiante/face-trait-transformer",
                    help="target HuggingFace Hub repo id")
    ap.add_argument("--model-card", type=Path,
                    default=Path(__file__).resolve().parents[2] / "docs" / "model_card.md",
                    help="Markdown to publish as the HF repo's README.md")
    ap.add_argument("--private", action="store_true",
                    help="create as a private repo (default public)")
    ap.add_argument("--commit-message", default="Publish face-trait-transformer bundle")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. `pip install huggingface_hub` and then "
            "`huggingface-cli login` once."
        ) from e

    if not args.bundle.exists():
        raise SystemExit(f"bundle dir not found: {args.bundle}")

    # Copy the model card into the bundle root as README.md so HF renders it.
    card_dest = args.bundle / "README.md"
    if args.model_card.exists():
        shutil.copy2(args.model_card, card_dest)
        print(f"copied {args.model_card} -> {card_dest}")
    else:
        print(f"warning: model card not found at {args.model_card}; skipping")

    api = HfApi()
    api.create_repo(repo_id=args.repo, private=args.private, exist_ok=True)
    print(f"uploading {args.bundle} -> https://huggingface.co/{args.repo}")
    api.upload_folder(
        folder_path=str(args.bundle),
        repo_id=args.repo,
        commit_message=args.commit_message,
    )
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
