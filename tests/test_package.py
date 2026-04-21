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


def test_end_to_end_from_bundle(tiny_bundle):
    """Load a complete (tiny) bundle and run a prediction end-to-end."""
    from PIL import Image

    predictor = TraitPredictor.from_bundle(tiny_bundle, device=torch.device("cpu"))
    img = Image.new("RGB", (64, 64), color=(120, 80, 90))

    # Single image, no TTA
    row = predictor.predict(img, tta=False)
    assert "filename" in row.index
    # 34 attributes + filename
    assert len(row) == 35
    # Numeric columns are in 0–100
    vals = row.drop(labels=["filename"]).to_numpy(dtype=float)
    assert vals.shape == (34,)
    assert (vals >= 0).all() and (vals <= 100).all()

    # Batch of two images, with TTA
    df = predictor.predict([img, img], tta=True)
    assert df.shape == (2, 35)
    # Deterministic: same image twice → same prediction (up to numeric noise)
    np.testing.assert_allclose(
        df.iloc[0].drop(labels=["filename"]).to_numpy(float),
        df.iloc[1].drop(labels=["filename"]).to_numpy(float),
        atol=1e-5,
    )


def test_end_to_end_predict_with_figure(tiny_bundle, tmp_path):
    from PIL import Image

    predictor = TraitPredictor.from_bundle(tiny_bundle, device=torch.device("cpu"))
    img = Image.new("RGB", (64, 64), color=(120, 80, 90))
    out_png = tmp_path / "diag.png"
    row, fig = predictor.predict_with_figure(img, out_path=out_png)
    assert out_png.exists() and out_png.stat().st_size > 1000
    assert len(row) == 34
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure)


def test_bootstrap_mean_metric_shapes():
    from face_trait_transformer.metrics import bootstrap_mean_metric
    rng = np.random.default_rng(0)
    y_true = rng.random((30, 5)) * 100
    y_pred = y_true + rng.normal(0, 5, y_true.shape)
    res = bootstrap_mean_metric(y_true, y_pred, metric="pearson_r", n_boot=200)
    assert 0 <= res["point"] <= 1
    assert res["ci_lo"] <= res["point"] <= res["ci_hi"]
    assert res["n_boot"] == 200
