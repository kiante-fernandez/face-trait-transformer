"""`TraitPredictor` — the public inference API.

A bundle is a directory containing:

    manifest.json           attr_names, backbones, finetune_files
    head_<group>/*.pt       MLP head checkpoints (frozen-backbone branch)
    finetune/*.pt           end-to-end fine-tune checkpoints (carry backbone + head)

See docs/methods.md for the full training recipe.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .features import build_transform, pick_device
from .hub import DEFAULT_REPO, download_from_hub
from .model import TraitHead

logger = logging.getLogger(__name__)


class TraitPredictor:
    """Face → 34-d perceived-trait regression, with cross-model ensembling + TTA.

    Construct via
      - :py:meth:`from_pretrained`  (download from HuggingFace Hub, recommended)
      - :py:meth:`from_bundle`      (use a local bundle directory, offline)
      - direct constructor if you're assembling your own ensemble from scratch.
    """

    def __init__(
        self,
        attr_names: list[str],
        backbones: dict[str, dict] | None = None,
        finetunes: list[Path] | None = None,
        device: torch.device | None = None,
    ):
        self.attr_names = list(attr_names)
        self.device = device or pick_device()
        self._backbone_groups = backbones or {}
        self._base_models: dict[str, tuple[nn.Module, int]] = {}
        self._group_state: dict[str, tuple[object, list[TraitHead]]] = {}
        self._finetune_paths = [Path(p) for p in (finetunes or [])]
        self._finetunes: list[tuple[nn.Module, object]] = []

    # -------- Constructors --------

    @classmethod
    def from_bundle(
        cls, bundle_dir: str | Path, device: torch.device | None = None
    ) -> TraitPredictor:
        """Load a TraitPredictor from a local bundle directory."""
        bundle_dir = Path(bundle_dir)
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        attr_names = manifest["attr_names"]
        backbones: dict[str, dict] = {}
        for gname, spec in manifest.get("backbones", {}).items():
            backbones[gname] = {
                "base_model": spec.get("base_model", gname),
                "image_size": int(spec.get("image_size", 224)),
                "head_files": [bundle_dir / f for f in spec["head_files"]],
            }
        finetunes = [bundle_dir / f for f in manifest.get("finetune_files", [])]
        return cls(
            attr_names=attr_names,
            backbones=backbones,
            finetunes=finetunes,
            device=device,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        device: torch.device | None = None,
    ) -> TraitPredictor:
        """Download the bundle from HuggingFace Hub and construct a TraitPredictor.

        On first call the weights (~1.2 GB) are cached locally;
        subsequent calls are offline.
        """
        local = download_from_hub(repo_id=repo_id, revision=revision,
                                  cache_dir=cache_dir, token=token)
        return cls.from_bundle(local, device=device)

    # -------- Lazy-loaded backbone cache --------

    def _ensure_group_loaded(self, group_name: str) -> None:
        if group_name in self._group_state:
            return
        spec = self._backbone_groups[group_name]
        base = spec["base_model"]
        image_size = int(spec["image_size"])
        if base not in self._base_models:
            logger.info("loading base model %s on %s", base, self.device)
            model = torch.hub.load("facebookresearch/dinov2", base)
            model.eval().to(self.device)
            self._base_models[base] = (model, int(model.norm.weight.shape[0]))
        _, in_dim = self._base_models[base]
        transform = build_transform(image_size=image_size)
        heads: list[TraitHead] = []
        for p in spec["head_files"]:
            ck = torch.load(p, map_location=self.device, weights_only=False)
            cfg = ck["config"]
            if cfg["in_dim"] != in_dim:
                raise ValueError(
                    f"head {Path(p).name} expects in_dim={cfg['in_dim']} but backbone "
                    f"{base} produces {in_dim}"
                )
            if list(cfg["attr_names"]) != self.attr_names:
                raise ValueError(f"head {Path(p).name} attr_names disagree with manifest")
            head = TraitHead(
                in_dim=cfg["in_dim"], out_dim=cfg["out_dim"],
                hidden=cfg["hidden"], dropout=cfg["dropout"],
            )
            head.load_state_dict(ck["state_dict"])
            head.eval().to(self.device)
            heads.append(head)
        self._group_state[group_name] = (transform, heads)

    def _ensure_finetunes_loaded(self) -> None:
        if self._finetunes or not self._finetune_paths:
            return
        for p in self._finetune_paths:
            ck = torch.load(p, map_location=self.device, weights_only=False)
            cfg = ck["config"]
            if list(cfg["attr_names"]) != self.attr_names:
                raise ValueError(f"finetune {p.name} attr_names disagree with manifest")
            logger.info("loading finetune %s on %s", p.name, self.device)
            backbone = torch.hub.load("facebookresearch/dinov2", cfg["backbone"])
            head = TraitHead(
                in_dim=cfg["in_dim"], out_dim=cfg["out_dim"],
                hidden=cfg["hidden"], dropout=cfg["dropout"],
            )

            class _Wrap(nn.Module):
                def __init__(self, b, h):
                    super().__init__()
                    self.backbone = b
                    self.head = h

                def forward(self, x):  # type: ignore[override]
                    return self.head(self.backbone(x))

            mod = _Wrap(backbone, head).to(self.device)
            mod.load_state_dict(ck["state_dict"])
            mod.eval()
            image_size = int(cfg.get("image_size", 224))
            self._finetunes.append((mod, build_transform(image_size=image_size)))

    # -------- Group forward helpers --------

    @torch.inference_mode()
    def _group_forward(self, group_name: str, pils: list[Image.Image]) -> torch.Tensor:
        self._ensure_group_loaded(group_name)
        transform, heads = self._group_state[group_name]
        base = self._backbone_groups[group_name]["base_model"]
        model, _ = self._base_models[base]
        x = torch.stack([transform(p) for p in pils]).to(self.device)
        feats = model(x)
        return torch.stack([h(feats) for h in heads], dim=0).mean(0)

    @torch.inference_mode()
    def _group_forward_flip(self, group_name: str, pils: list[Image.Image]) -> torch.Tensor:
        flipped = [p.transpose(Image.FLIP_LEFT_RIGHT) for p in pils]
        return self._group_forward(group_name, flipped)

    # -------- Prediction --------

    def predict(
        self,
        images: str | Path | Image.Image | Iterable[str | Path | Image.Image],
        batch_size: int = 16,
        return_dataframe: bool = True,
        tta: bool = True,
        progress: bool = True,
    ):
        """Predict the 34-d trait vector for one or more images.

        Parameters
        ----------
        images            : path, PIL image, or iterable of those.
        batch_size        : forward-pass batch size (lower if VRAM/MPS-tight).
        return_dataframe  : return a pandas DataFrame (default) or numpy array.
        tta               : average predictions on image + horizontal flip.
        progress          : show a tqdm progress bar for batched inputs
                            (auto-disabled for single-image calls).

        Returns
        -------
        Single image → pandas.Series (if return_dataframe) or (34,) ndarray.
        Multiple    → pandas.DataFrame (N, 35 with filename col) or (N, 34) ndarray.
        Values are on the 0–100 scale, clamped.
        """
        if isinstance(images, (str, Path, Image.Image)):
            inputs: Sequence = [images]
            single = True
        else:
            inputs = list(images)
            single = False

        pils: list[Image.Image] = []
        filenames: list[str] = []
        for item in inputs:
            if isinstance(item, Image.Image):
                img = item.convert("RGB"); name = "<PIL image>"
            else:
                img = Image.open(item).convert("RGB"); name = str(item)
            pils.append(img); filenames.append(name)

        from tqdm.auto import tqdm as _tqdm
        show_bar = progress and len(pils) > batch_size

        def _chunks(pil_list: list[Image.Image], fwd, desc: str = "predict"):
            outs = []
            starts = range(0, len(pil_list), batch_size)
            iterator = _tqdm(starts, desc=desc, leave=False, disable=not show_bar,
                             total=(len(pil_list) + batch_size - 1) // batch_size)
            for s in iterator:
                outs.append(fwd(pil_list[s : s + batch_size]))
            return torch.cat(outs, dim=0)

        group_preds: list[torch.Tensor] = []
        for gname in self._backbone_groups:
            preds = _chunks(pils, lambda b, g=gname: self._group_forward(g, b))
            if tta:
                flipped = _chunks(pils, lambda b, g=gname: self._group_forward_flip(g, b))
                preds = (preds + flipped) / 2
            group_preds.append(preds)

        self._ensure_finetunes_loaded()
        for mod, tfm in self._finetunes:
            def _run(pl, tfm=tfm, mod=mod, flip=False):
                if flip:
                    pl = [p.transpose(Image.FLIP_LEFT_RIGHT) for p in pl]
                x = torch.stack([tfm(p) for p in pl]).to(self.device)
                with torch.inference_mode():
                    return mod(x)

            preds = _chunks(pils, _run)
            if tta:
                flipped = _chunks(pils, lambda b: _run(b, flip=True))
                preds = (preds + flipped) / 2
            group_preds.append(preds)

        if not group_preds:
            raise RuntimeError("No models to run — bundle is empty.")
        y = torch.stack(group_preds, dim=0).mean(0).detach().cpu().numpy()
        y100 = np.clip(y * 100.0, 0.0, 100.0)

        if return_dataframe:
            import pandas as pd
            df = pd.DataFrame(y100, columns=self.attr_names)
            df.insert(0, "filename", filenames)
            return df.iloc[0] if single else df
        return y100[0] if single else y100

    def predict_with_figure(
        self,
        image: str | Path | Image.Image,
        out_path: str | Path | None = None,
        show: bool = False,
    ):
        """Predict and also render a three-panel diagnostic figure.

        Requires matplotlib (``pip install face-trait-transformer[figures]``).
        Returns ``(pandas.Series, matplotlib.figure.Figure)``.
        """
        from .figures import render_single_prediction_figure

        row = self.predict(image, return_dataframe=True)
        values = row.drop(labels=["filename"]).to_numpy(dtype=float)
        fig = render_single_prediction_figure(
            image, self.attr_names, values, out_path=out_path, show=show
        )
        return row.drop(labels=["filename"]), fig
