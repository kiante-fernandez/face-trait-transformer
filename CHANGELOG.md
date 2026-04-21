# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-20

Initial public release.

### Added

- `TraitPredictor` public API with `.from_pretrained("kiante/face-trait-transformer")`
  and `.from_bundle(local_dir)` constructors; `.predict(...)` for single images
  or batches and `.predict_with_figure(...)` for a three-panel diagnostic plot.
- `ftt` CLI with `predict` and `download` subcommands.
- HuggingFace Hub integration via `from_pretrained`.
- Horizontal-flip test-time augmentation (enabled by default).
- tqdm progress bar on batched `predict()` (auto-disabled for single-image
  calls; suppress with `progress=False`).
- stdlib `logging` instead of raw `print()` inside the library. Silence with
  `logging.getLogger("face_trait_transformer").setLevel(logging.WARNING)`.
- `torchvision.transforms.v2` pipeline for image preprocessing.
- End-to-end fixture test (`tests/conftest.py::tiny_bundle`) that exercises
  `predict` and `predict_with_figure` without downloading weights.
- `training/` reproducibility pipeline: 62-config head sweep, variance-weighted
  MSE against raw trial ratings, ViT-L/14 last-block fine-tune, 10-fold CV
  matching Peterson et al. (2022), split-half reliability ceiling, bootstrap
  confidence intervals, and aging-dataset validity analysis.
- `docs/methods.md` — paper-ready methods section.
- `docs/model_card.md` — HuggingFace-style model card with biases,
  limitations, intended use, and citation guidance.
- `examples/quickstart.py`, `examples/batch_inference.py`,
  `examples/quickstart.ipynb` (Colab-ready), and
  `examples/aging_validation/` (a worked external-validity case study).

### Model performance

- Test mean Pearson r = **0.857** (95 % CI 0.842–0.870), mean R² = **0.738**
  (95 % CI 0.707–0.755) on a held-out 101-stimulus test split of OMI.
- 10-fold CV mean R² = **0.734**, which is **99 % of the split-half reliability
  ceiling** averaged across 34 attributes.
- Beats the Peterson et al. 2022 R² on 33 of 34 attributes, with the largest
  gains concentrated on the attributes they flagged as hardest
  (`gay` +0.50, `looks-like-you` +0.39, `Black` +0.33, `believes in god` +0.31).

### License

- Code: MIT (`LICENSE`).
- Model weights: CC BY-NC-SA 4.0, inherited from the OMI dataset
  (`LICENSE-WEIGHTS`).
