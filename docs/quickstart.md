# Quickstart

## Install

```bash
pip install face-trait-transformer[hub,figures]
```

Optional extras:
- `hub` adds `huggingface_hub` for `from_pretrained(...)`.
- `figures` adds `matplotlib` for `predict_with_figure(...)`.

## One image → trait vector

```python
from face_trait_transformer import TraitPredictor

m = TraitPredictor.from_pretrained("kiante/face-trait-transformer")
row = m.predict("face.jpg")         # pandas.Series, 34 attributes, 0–100
print(row["age"], row["trustworthy"], row["electable"])
```

First call downloads the 1.2 GB bundle to HuggingFace Hub's cache
(`~/.cache/huggingface/…`). Also on first call, `torch.hub` downloads the
DINOv2 backbone(s) it needs (~1.2 GB for ViT-G/14).

## Diagnostic figure

```python
row, fig = m.predict_with_figure("face.jpg", out_path="diag.png")
```

Produces a three-panel figure: the face on the left, a radar plot of the 34
predicted traits, and a bar chart sorted high → low. See
`figures/per_face/stimulus_0582.png` for an example.

## Batch

```python
df = m.predict(
    ["face1.jpg", "face2.jpg", "face3.jpg"],
    batch_size=4,
    tta=True,
)
# df has columns [filename, trustworthy, attractive, …] — 35 cols total
df.to_csv("predictions.csv", index=False)
```

## Command-line

```bash
# single image with diagnostic figure
ftt predict face.jpg --figure diag.png

# directory
ftt predict faces_dir/ --out predictions.csv --no-tta

# pre-fetch the bundle to the local cache (useful for air-gapped nodes)
ftt download
```

## Offline / local bundle

If you've already downloaded the bundle (or trained your own via
`training/scripts/export_bundle.py`):

```python
m = TraitPredictor.from_bundle("/path/to/bundle")
```

## Notes

- Values are returned on a **0–100** scale and clamped. 0 = least / most-feminine /
  darkest / etc.; 100 = most of the attribute.
- These are **perceived traits** from rater judgments, not ground-truth
  attributes. Frame outputs accordingly, especially for demographic and
  socially-constructed dimensions.
- Device: `from_pretrained()` auto-selects `cuda > mps > cpu`. Override with
  `device=torch.device("cpu")`.
