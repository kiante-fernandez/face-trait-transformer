"""Shared fixtures for end-to-end tests.

Builds a tiny bundle on the fly: 2 MLP heads + 1 finetune-style checkpoint, all
operating on an 8-dim backbone feature. A monkeypatched torch.hub.load replaces
DINOv2 with a stub that produces random 8-dim vectors, so the test runs in a
few seconds with no network / no DINOv2 weights required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

TINY_ATTR_NAMES = [
    "trustworthy", "attractive", "dominant", "smart", "age", "gender", "weight",
    "typical", "happy", "familiar", "outgoing", "memorable", "well-groomed",
    "long-haired", "smug", "dorky", "skin-color", "hair-color", "alert", "cute",
    "privileged", "liberal", "asian", "middle-eastern", "hispanic", "islander",
    "native", "black", "white", "looks-like-you", "gay", "electable", "godly",
    "outdoors",
]  # 34


class _StubBackbone(nn.Module):
    """Returns a deterministic 8-dim vector per image (same spec as DINOv2's CLS token)."""

    def __init__(self, out_dim: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(out_dim)   # our code reads .norm.weight to get in_dim
        self.proj = nn.Linear(3 * 16 * 16, out_dim)   # ignores input size, just deterministic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W); downsample to 16x16 then linear
        x = torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(1)
        return self.proj(x)


@pytest.fixture
def tiny_bundle(tmp_path: Path, monkeypatch) -> Path:
    """Writes a complete bundle to tmp_path and returns its path.

    The bundle has:
      - manifest.json (attr_names, one backbone group "dinov2_vitb14" at 224, one finetune)
      - head_dinov2_vitb14/head_{0,1}.pt
      - finetune/finetune_stub.pt
    """
    attrs = TINY_ATTR_NAMES
    IN_DIM = 8

    head_dir = tmp_path / "head_dinov2_vitb14"
    head_dir.mkdir()
    for i in range(2):
        head = nn.Sequential(
            nn.Linear(IN_DIM, 16), nn.GELU(), nn.Dropout(0.1), nn.Linear(16, 34),
        )
        torch.save({
            "state_dict": {f"net.{k}": v for k, v in head.state_dict().items()},
            "config": {
                "in_dim": IN_DIM, "out_dim": 34, "hidden": 16, "dropout": 0.1,
                "head": "mlp", "backbone": "dinov2_vitb14",
                "attr_names": attrs, "seed": i, "best_val_mean_r": 0.1,
            },
        }, head_dir / f"head_{i}.pt")

    # Finetune = (StubBackbone + head) wrapped in the _Wrap-compatible format
    ft_dir = tmp_path / "finetune"
    ft_dir.mkdir()
    stub_bb = _StubBackbone(IN_DIM)
    head = nn.Sequential(
        nn.Linear(IN_DIM, 16), nn.GELU(), nn.Dropout(0.1), nn.Linear(16, 34),
    )
    # Mirror the _Wrap structure: module keys prefixed "backbone." and "head.net.<k>"
    wrap_state: dict[str, torch.Tensor] = {}
    for k, v in stub_bb.state_dict().items():
        wrap_state[f"backbone.{k}"] = v
    for k, v in head.state_dict().items():
        wrap_state[f"head.net.{k}"] = v
    torch.save({
        "state_dict": wrap_state,
        "config": {
            "in_dim": IN_DIM, "out_dim": 34, "hidden": 16, "dropout": 0.1,
            "head": "mlp", "backbone": "dinov2_vitb14", "image_size": 224,
            "attr_names": attrs, "seed": 0, "best_val_mean_r": 0.1, "finetune": True,
        },
    }, ft_dir / "finetune_stub.pt")

    manifest = {
        "attr_names": attrs,
        "backbones": {
            "dinov2_vitb14": {
                "base_model": "dinov2_vitb14",
                "image_size": 224,
                "top_k": 2,
                "head_files": [f"head_dinov2_vitb14/head_{i}.pt" for i in range(2)],
            },
        },
        "finetune_files": ["finetune/finetune_stub.pt"],
        "bundle_version": 1,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Monkeypatch torch.hub.load to return our stub backbone.
    def _fake_hub_load(repo: str, model: str, *args, **kwargs):
        return _StubBackbone(IN_DIM)

    monkeypatch.setattr(torch.hub, "load", _fake_hub_load)
    return tmp_path
