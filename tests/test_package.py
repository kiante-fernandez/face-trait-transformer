"""Sanity tests that don't require downloading model weights."""
from __future__ import annotations

import numpy as np
import torch

import face_trait_transformer
from face_trait_transformer import TraitPredictor
from face_trait_transformer.features import build_transform, pick_device
from face_trait_transformer.metrics import per_attribute_pearson, summary
from face_trait_transformer.model import TraitHead


def test_package_exposes_version_and_predictor():
    assert isinstance(face_trait_transformer.__version__, str)
    assert TraitPredictor is not None


def test_build_transform_shape():
    tfm = build_transform(image_size=224)
    from PIL import Image
    dummy = Image.new("RGB", (300, 300), color="white")
    x = tfm(dummy)
    assert x.shape == (3, 224, 224)

    tfm518 = build_transform(image_size=518)
    x518 = tfm518(dummy)
    assert x518.shape == (3, 518, 518)


def test_pick_device_returns_torch_device():
    dev = pick_device()
    assert isinstance(dev, torch.device)


def test_trait_head_forward():
    head = TraitHead(in_dim=128, out_dim=34, hidden=64, dropout=0.1)
    x = torch.randn(5, 128)
    y = head(x)
    assert y.shape == (5, 34)
    # Linear probe
    linear = TraitHead(in_dim=128, out_dim=34, hidden=None)
    y2 = linear(x)
    assert y2.shape == (5, 34)


def test_metrics_on_fake_data():
    rng = np.random.default_rng(0)
    y = rng.random((50, 34))
    rs, ps = per_attribute_pearson(y, y)
    assert np.allclose(rs, 1.0)
    # summary has the right columns
    df = summary(y, y, [f"a{i}" for i in range(34)])
    assert list(df.columns) == ["attribute", "pearson_r", "pearson_p", "R2", "mae", "rmse"]
    assert np.allclose(df["R2"].to_numpy(), 1.0)


def test_predictor_requires_at_least_one_model(tmp_path):
    # With no backbones and no finetunes, predict should raise cleanly.
    p = TraitPredictor(attr_names=["a1"], backbones={}, finetunes=[])
    import pytest
    from PIL import Image
    img = Image.new("RGB", (10, 10), color="white")
    with pytest.raises((RuntimeError, SystemExit)):
        p.predict(img)
