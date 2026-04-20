"""HuggingFace Hub integration: download the shipped bundle on first use."""
from __future__ import annotations

from pathlib import Path

DEFAULT_REPO = "kiante/face-trait-transformer"


def download_from_hub(
    repo_id: str = DEFAULT_REPO,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> Path:
    """Fetch the bundle files from HuggingFace Hub and return the local path.

    Requires `pip install face-trait-transformer[hub]` (i.e. huggingface_hub).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to call from_pretrained(). "
            "Install it with `pip install face-trait-transformer[hub]` "
            "or `pip install huggingface_hub`."
        ) from e
    local = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    return Path(local)
