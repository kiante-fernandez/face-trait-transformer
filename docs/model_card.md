---
language: en
license: cc-by-nc-sa-4.0
library_name: face-trait-transformer
tags:
  - face
  - perception
  - trait-prediction
  - dinov2
  - omi
datasets:
  - jcpeterson/omi
pipeline_tag: image-feature-extraction
---

# face-trait-transformer

**Predict 34-dimensional perceived-trait vectors from face images.** Trained on
the One Million Impressions (OMI) dataset (Peterson et al., 2022, *PNAS*).

## Quick inference

```python
from face_trait_transformer import TraitPredictor

m = TraitPredictor.from_pretrained("kiante/face-trait-transformer")
row = m.predict("face.jpg")            # pandas.Series, 34 attributes, 0–100 scale
df  = m.predict(["a.jpg", "b.jpg"])    # pandas.DataFrame (N × 35, filename + 34 attrs)
```

## Model summary

- **Architecture.** Cross-model ensemble of (1) 10 MLP heads over frozen DINOv2
  ViT-G/14 CLS features at 518 × 518, trained with variance-weighted MSE, and
  (2) a partial end-to-end fine-tune of DINOv2 ViT-L/14 (last transformer block
  + head). At inference we apply horizontal-flip test-time augmentation.
- **Inputs.** Any face image. Preprocessing: bicubic resize → center-crop →
  ImageNet normalization (handled internally).
- **Outputs.** 34-dim float vector on a 0–100 scale, one value per OMI
  attribute: *trustworthy, attractive, dominant, smart, age, gender, weight,
  typical, happy, familiar, outgoing, memorable, well-groomed, long-haired,
  smug, dorky, skin-color, hair-color, alert, cute, privileged, liberal,
  asian, middle-eastern, hispanic, islander, native, black, white,
  looks-like-you, gay, electable, godly, outdoors.*

## Performance

Evaluated on a held-out 101-stimulus test split of OMI (seed-0 80/10/10).

| Metric | Value |
|---|---|
| Mean Pearson r (34 attrs) | **0.858** |
| Mean R² | **0.738** |
| Median Pearson r | 0.897 |
| Per-attribute r range | 0.63 – 0.98 |

Under 10-fold cross-validation (frozen-backbone branch only, matching the
Peterson 2022 protocol), we get **mean R² = 0.734** — outperforming the
Peterson et al. 2022 reported value (~0.55) on **33 of 34 attributes**. The
largest gains are on attributes they flagged as hardest (`gay` +0.50, `Black`
+0.33, `believes in god` +0.31, `looks-like-you` +0.39). See
[docs/methods.md](../blob/main/docs/methods.md) for the full comparison.

## Intended use

- Research on face perception, social cognition, and stereotyping.
- Computational social-science analyses that need perceived-trait covariates
  for arbitrary face photographs.
- Augmenting behavioral datasets with machine estimates of how raters *would*
  have rated new stimuli.

## Out-of-scope / misuse

- **Do not use as ground-truth demographic or personality assessment.** These
  are perceived traits from rater judgments, not identity properties.
- **Not validated for high-stakes decisions** (hiring, credit, legal). Use on
  identifiable people without consent may be unlawful under local privacy
  regimes.
- The OMI dataset is CC BY-NC-SA 4.0 — non-commercial only.

## Biases & limitations

- Training raters were predominantly White and US-based; predictions inherit
  their stereotypes and in-group biases.
- Demographic attribute predictions trend toward "white" on intermediate-
  phenotype groups (Latin, Middle-Eastern); see validity analysis on the
  GIRAF aging-faces dataset (`examples/aging_validation/`).
- No explicit South-Asian category in OMI — Indian-coded stimuli get high
  probability on "hispanic"/"asian" rather than a dedicated label.
- Subjective attributes (looks-like-you, familiar, memorable, typical) have
  low inter-rater reliability; model ceiling is correspondingly lower.

## Training data

1,004 synthetic face images (StyleGAN2 on FFHQ, 1024 × 1024) with ~1.3 M
trial-level ratings from ~4,100 Amazon Mechanical Turk participants,
aggregated to per-(stimulus, attribute) means. Dataset:
https://github.com/jcpeterson/omi.

## Citation

If you use this model, please cite the underlying dataset:

```bibtex
@article{peterson2022omi,
  title={Deep models of superficial face judgments},
  author={Peterson, Joshua C and Uddenberg, Stefan and Griffiths, Thomas L
          and Todorov, Alexander and Suchow, Jordan W},
  journal={Proceedings of the National Academy of Sciences},
  volume={119}, number={17}, pages={e2115228119}, year={2022},
  doi={10.1073/pnas.2115228119}
}
```

## License

Weights are **CC BY-NC-SA 4.0** (inheriting from OMI). The loading code is
**MIT**-licensed.
