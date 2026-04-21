# External validity check on CelebA (50 identities × 10 photos)

A second worked example of running `TraitPredictor` on an independent dataset
and measuring whether the predictions behave sensibly. This set has a
different structure from the [aging_validation](../aging_validation/README.md)
case: no age / emotion / demographic metadata per image, but we know each of
the 500 photos belongs to one of 50 celebrities (10 photos per identity). That
lets us ask a question the aging dataset can't: **does the model produce
face-level predictions, or just image-level noise?**

**Dataset.** A 500-image slice of CelebA curated into 50 identities × 10
images for single-neuron face-perception experiments. Provided by the original
release as `CelebA/*.jpg` + `FaceImageIndex.csv` + `CelebA_Image_Code_new.mat`
(we don't redistribute; see the original release for access).

**Protocol.** Run `TraitPredictor.from_pretrained("kiante/face-trait-transformer")`
over every image; join identity labels from the .mat file; decompose variance
into within-identity and between-identity components.

## Headline result

**Mean ICC(1,1) across 34 attributes = 0.619** — i.e. ~62 % of the variance in
the model's trait predictions is explained by *who* the person is, not *which
photo*. The split between "identity-defining" and "state-dependent" attributes
is exactly the one a well-calibrated face-perception model should produce:

### Highest ICC — identity-defining, stable across photos

| attribute | ICC(1,1) | within-ID SD | between-ID SD |
|---|---|---|---|
| gender | **0.95** | 5.8 | 28.1 |
| long-haired | **0.90** | 5.2 | 18.1 |
| black | **0.88** | 6.7 | 20.6 |
| cute | **0.85** | 5.3 | 13.6 |
| skin-color | **0.83** | 5.8 | 14.0 |
| age | **0.82** | 4.2 | 9.9 |
| white | **0.81** | 10.0 | 22.8 |
| hair-color | **0.77** | 10.3 | 20.2 |

These are all attributes that *shouldn't* change across photos of the same
person (unless someone radically restyles their hair). The model treats them
that way.

### Lowest ICC — state/context-dependent, varies across photos

| attribute | ICC(1,1) | within-ID SD | between-ID SD |
|---|---|---|---|
| outdoors | **0.16** | 13.8 | 7.9 |
| alert | 0.36 | 5.6 | 4.9 |
| happy | **0.39** | 11.6 | 10.5 |
| smart | 0.41 | 5.2 | 5.1 |
| outgoing | 0.45 | 8.6 | 8.6 |

`outdoors` hitting 0.16 is the single clearest validity result: the model has
learned that whether a photo was taken outdoors is a **photo property**, not
an **identity property**. Same for `happy` — within-identity SD is actually
*larger* than between-identity SD, i.e. how happy a celebrity looks varies
more across their 10 photos than across different celebrities. That's the
correct behavior when evaluating a dataset of candid celebrity photos.

### Middle band

Demographic attributes (asian, hispanic, native, middle-eastern, islander) and
higher-level social inferences (liberal, gay, privileged, dominant, godly) fall
in the 0.50–0.70 range — still reliably identity-linked, but with more
per-photo variability than the bodily attributes.

## Per-identity profile plot

`identity_profiles.png` shows the 10-photo min/mean/max per attribute for 6
sampled celebrities (spanning the distinctiveness range). Visualizes stability:
when the shaded band is narrow the prediction is stable across photos.

## Trait-space PCA

`trait_pca.png` projects each of the 500 images into the top-2 PCs of the
34-d trait space and colors/labels by identity. PC1 + PC2 together capture
~59 % of variance; same-identity photos visibly cluster.

## Example diagnostics

Four per-face diagnostic panels (image + 34-attribute radar + sorted bar chart)
for a sample of four identities spanning the distinctiveness range, plus a
within-identity overlay figure visualizing how stable the trait vector is
across a single celebrity's 10 photos.

| File | What it shows |
|---|---|
| `diagnostics/identity_08_001693.png` | identity 8 (near the grand mean of all predictions) |
| `diagnostics/identity_44_013251.png` | identity 44 |
| `diagnostics/identity_34_032470.png` | identity 34 |
| `diagnostics/identity_15_091612.png` | identity 15 (most distinctive in trait space) |
| `diagnostics/within_identity_overlay.png` | identity 15 — 10 individual photos as translucent red lines, with the identity mean in black |

The overlay is the key within-identity stability visualization: tightly
clustered red lines where the black mean sits mean the model gives the same
answer across this person's 10 photos (identity-defining traits); wider
spread means the model lets that attribute vary photo-to-photo (state-
dependent traits). Matches what the ICC table numerically says.

## Reproducing this example

```bash
# 1. predict
python -m training.scripts.apply_to_celeba \
    --root /path/to/CelebA-validation \
    --out  examples/celeba_validation/predictions.csv \
    --batch-size 4

# 2. analyze
python -m training.scripts.analyze_celeba \
    --predictions examples/celeba_validation/predictions.csv \
    --out-dir     examples/celeba_validation
```

The full `predictions.csv` (500 rows × 37 cols: image_name, face_index,
identity, and 34 traits) and four validity figures live alongside this README.

## What to read it as

This test complements the GIRAF aging-faces analysis: GIRAF checked that
predictions track **explicit labels** (age group, emotion, gender, demographic
code). This one checks that predictions behave like **coherent face-level
traits** — consistent within a person, discriminating between people, and
appropriately noisy on attributes that genuinely depend on the specific photo.
Together they're much stronger evidence of external validity than either alone.
