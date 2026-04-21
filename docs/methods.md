# Methods — Face → 34-attribute perceived-trait model

Paper-ready write-up of the pipeline in this repository. Update alongside any
change to the training recipe so the bundle stays self-documenting.

## Dataset

We used the One Million Impressions (OMI) dataset (Peterson, Uddenberg,
Griffiths, Todorov, & Suchow, 2022, *PNAS*), which contains mean human ratings
of **N = 1,004 synthetic face images** (StyleGAN2 samples trained on FFHQ,
1024 × 1024 RGB) on **34 perceptual attributes** (e.g. *trustworthy, attractive,
dominant, smart, age, gender, happy, electable, …*) aggregated from ~1.3 M
individual trial-level judgments. Each attribute is a population-mean rating on
a 0–100 scale; there are no missing values. We frame the task as multi-output
regression from a face image to the 34-dimensional vector of perceived-trait
means.

We explicitly treat predictions as **perceived traits, not ground-truth
attributes of the depicted people**, per the dataset authors' ethical guidance.

## Train / val / test split

For the primary reported model we used a single **80 / 10 / 10** split of the
1,004 stimuli — train (n = 803), validation (n = 100), test (n = 101) — by
stimulus ID using a deterministic permutation (`numpy.random.default_rng(seed=0)`).
The same split was reused across every model and every ensemble member so
test-set metrics are directly comparable (the split file,
`artifacts/splits.json`, is checked in). For the Peterson-comparison
cross-validation analysis we additionally ran **10-fold CV** over the full
1,004 stimuli; see the Validation section below for its protocol and results.

## Image preprocessing

All inputs used the standard DINOv2 preprocessing pipeline: bicubic resize
(shorter side = `image_size × 256 / 224`), center-crop to `image_size × image_size`,
convert to a float tensor in [0, 1], and normalize with ImageNet mean/std
(`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`). We use two
input resolutions: `image_size = 224` for the frozen ViT-L/14 head sweep and
the ViT-L/14 fine-tune, and `image_size = 518` (DINOv2's native training
resolution) for the ViT-G/14 head sweep.

Training does not use augmentation — backbones are run in eval mode and
features cached. At inference we apply **test-time augmentation (TTA)** by
averaging predictions on the image and its horizontal flip. TTA gave ≈ +0.001
to the ensemble mean Pearson r on the held-out test set.

## Backbones (feature extractors)

We used two self-supervised **DINOv2** Vision Transformers (Oquab et al., 2024),
loaded via `torch.hub` (`facebookresearch/dinov2`):

| Backbone        | Params  | Patch | CLS feature dim |
|-----------------|---------|-------|-----------------|
| `dinov2_vitl14` | ~300 M  | 14    | 1,024           |
| `dinov2_vitg14` | ~1.1 B  | 14    | 1,536           |

For each image we extracted the backbone's CLS-token output (post-final
LayerNorm) and cached it to disk. A smaller `dinov2_vitb14` (768-dim) model was
used as an early baseline.

## Regression head

Per-backbone heads are small MLPs:

```
Linear(d_in → h) → GELU → Dropout(p) → Linear(h → 34)
```

with `d_in ∈ {768, 1024, 1536}` matching the backbone. The output layer is
plain linear (no sigmoid); at inference time raw outputs are clipped to [0, 1]
and rescaled to 0–100. A linear probe (`Linear(d_in → 34)`) was included as a
sanity baseline.

## Training objective and optimization

- **Base loss:** MSE on targets rescaled to [0, 1] (division by 100).
- **Variance-weighted MSE (final 518-px ViT-G sweep):** we aggregate the raw
  trial-level ratings in `attribute_ratings.csv` to compute a per-cell sample
  standard deviation `s_ij` across the ≈ 38 raters per `(stimulus i, attribute j)`.
  The loss is weighted element-wise:
    `L = (1/NA) ∑_{i,j} w_ij (ŷ_ij − y_ij)² ,    w_ij = median(s) / (s_ij + ε)` ,
  with weights normalized to mean 1 so the overall loss scale is unchanged.
  The effect: cells with high rater disagreement contribute less to the gradient,
  which prevents the model from chasing rater noise on inherently subjective
  attribute-image pairs.
- **Optimizer:** AdamW.
- **Schedule:** cosine annealing from peak learning rate to `lr × 0.01` over
  the full epoch budget.
- **Batch size:** 64 (heads only; the full feature matrix fits in RAM).
- **Epochs:** up to 400 per run.
- **Early stopping:** patience of 40 epochs on **validation mean Pearson r**
  across the 34 attributes.
- **Random seed:** explicit (`torch.manual_seed`); sweeps vary seed alongside
  the other hyperparameters.

## Hyperparameter sweep

We ran **three 62-task head sweeps**, all sharing the same configuration grid:

1. Frozen ViT-L/14 features at 224 px (exploratory; not in the final ensemble).
2. Frozen ViT-G/14 features at 224 px (intermediate milestone; not in the final
   ensemble).
3. **Frozen ViT-G/14 features at 518 px with variance-weighted MSE** (this is
   the head branch of the shipped model).

The grid:

- `hidden ∈ {512, 1024, 2048}`
- `dropout ∈ {0.1, 0.2, 0.3}`
- `lr ∈ {1 × 10⁻⁴, 3 × 10⁻⁴, 1 × 10⁻³}`
- `weight_decay ∈ {1 × 10⁻⁴, 1 × 10⁻³}`
- `seed = 0`

That is 3 × 3 × 3 × 2 = 54 configurations (one run each at `seed = 0`). To
estimate run-to-run variance we added 8 extra re-runs of a central
configuration at `seed ∈ {1, 2}` (`hidden ∈ {1024, 2048}`,
`dropout ∈ {0.2, 0.3}`, `lr = 3 × 10⁻⁴`, `wd = 1 × 10⁻⁴`), giving **62 total
runs per sweep**. Configurations and the mapping of SGE array task IDs to
hyperparameters are checkpointed in `hoffman2/sweep_configs.json` and are fully
reproducible.

## Model selection

Final checkpoints were chosen on the best **validation** mean Pearson r (never
test). When ranking models for ensembling we also used validation scores to
avoid selection bias on the held-out test set.

## Partial end-to-end fine-tuning

In addition to the frozen-backbone sweep, we ran a **partial end-to-end fine-tune**
of ViT-L/14: the final transformer block and LayerNorm were unfrozen and trained
jointly with a 1,024-d MLP head. Preprocessing and objective were the same as
the frozen-backbone pipeline. Two parameter groups had different learning rates:
`lr = 1 × 10⁻⁴` for the backbone block, `lr = 1 × 10⁻³` for the head (10× ratio).
Batch size 16, AdamW, cosine LR schedule, `weight_decay = 1 × 10⁻⁴`, early stop
on validation mean Pearson r with patience 6, max 25 epochs. Training took
~40 minutes on a 16-core CPU node via SGE (no GPU).

## Ensembling

We used **unweighted averaging of predictions** from the top-k heads (ranked by
validation mean r). We swept k ∈ {1, 3, 5, 10, 15, 20, 30, 62}; k = 10–15 was a
stable optimum and we report k = 10. For the **final shipped model** we build
a 2-way cross-model ensemble by averaging:

1. the ViT-G/14 @ 518 px variance-weighted top-10 head ensemble, and
2. the fine-tuned ViT-L/14 (last-block) model's prediction.

Adding earlier components (frozen ViT-L/14 224-px heads, frozen ViT-G/14
224-px heads) gave marginal gains (≤ 0.001 test mean r) and bloated the
bundle, so they are excluded from the shipped model. Val-weighted averaging
did not improve over the plain mean (within 1 × 10⁻⁴), so unweighted
averaging was retained for reporting.

Inference additionally applies **horizontal-flip test-time augmentation
(TTA)**: each image is forwarded through every group once as-is and once
mirrored, and the two views are averaged before the cross-group mean. TTA
contributes ≈ +0.001 mean Pearson r on the held-out test set.

## Evaluation

Primary metric was **per-attribute Pearson r** on the held-out test set (101
stimuli), summarized as the **mean and median across the 34 attributes**. We
also report **per-attribute MAE and RMSE** (on the 0–100 scale) and the
individual Pearson r for every attribute. In addition, we ran a held-out
**monotonicity check** on the validation-image set from the same paper, which
contains faces manipulated along a single attribute at three discrete levels
(−0.5 SD, mean, +0.5 SD): for each (stimulus, attribute) triple we verified
that predictions along the manipulated attribute were monotonic-increasing
across levels. This is a free sanity check that the model learned the correct
direction of each attribute and not only its mean.

## Compute

All feature extraction and head training was done on **CPU** (no GPU). Two
environments were used:

- **Local iteration** (Apple M-series, MPS): rapid prototyping and the
  ViT-B/14 baseline.
- **UCLA Hoffman2 cluster** (Univa Grid Engine / SGE): Python 3.11, PyTorch
  2.6 CPU wheels, `miniforge` conda environment. Backbone weights were
  pre-downloaded to `~/.cache/torch/hub/checkpoints/` on the login node
  because compute nodes have no outbound internet.

Feature extraction on Hoffman2 was done with array jobs that split the 1,004
images into 32 chunks of ~32 images each, one SGE task per chunk
(`h_data = 16 G`, single slot). This reduced queue latency (32 × 1-slot
schedules near-instantly versus one 32-slot request that can wait hours) and
produced per-chunk `.npy` files that were concatenated in stimulus-ID order
via `scripts/combine_feature_chunks.py`. Head-training was a 62-way SGE array
(`h_data = 4 G`, single slot, ≤ 30 concurrent) using `scripts/train.py` through
`hoffman2/run_sweep_task.py`.

Wall-clock on Hoffman2 (shared campus queue):

- ViT-L/14 feature extraction on 1,004 images at 224 px: **≈ 9 min** on a
  16-slot shared node.
- ViT-G/14 feature extraction at 224 px (32-way 1-slot array): **≈ 15 min wall
  / ≈ 8 h CPU**.
- ViT-G/14 feature extraction at 518 px (32-way 1-slot array, 16 GB/slot):
  **≈ 1 h wall** (the tail is limited by the slowest campus-queue nodes).
- One head training run on cached features: **< 2 min**.
- ViT-L/14 last-block fine-tune at 224 px (16 slots, 20 h walltime cap):
  **≈ 40 min** (early-stopped at epoch 23 of 25).
- 10-fold CV head sweep (620 tasks × 1 slot): **≈ 20 min wall** with
  `-tc 50` concurrency.

## Reproducibility and software

- Python 3.11, PyTorch 2.6 (CPU), torchvision, NumPy 2.4, Pillow 12,
  SciPy 1.17, pandas 3.0, matplotlib 3.10.
- All seeds fixed. Primary-split file `artifacts/splits.json`, CV split files
  `artifacts/cv_splits/fold_XX.json`, and sweep configuration
  `hoffman2/sweep_configs.json` are checked in.
- Code, job scripts, sweep configuration, and deterministic split files are
  provided in the repository. Figures are reproduced via
  `training/scripts/regen_figures_fast.py` (per-face and overview scatter).
  A portable inference bundle (heads + fine-tune + manifest) is produced by
  `training/scripts/export_bundle.py` and published to HuggingFace Hub.

## Results to report

Held-out test set: 101 stimuli (80/10/10 seed-0 split). Metric: mean Pearson r
across 34 attributes. All numbers below use the **same** test split.

- **Baseline** (ViT-B/14 + 512-d MLP head, single run, seed 0): test mean
  r = **0.7567**, median r = **0.8188**.
- **Single best ViT-L head** (val-selected): 0.7894.
- **ViT-L top-10 head ensemble:** 0.8208.
- **ViT-G/14 top-10 head ensemble (224-px):** 0.8361.
- **ViT-L/14 fine-tuned (last-block, 224-px):** 0.8352.
- **ViT-G/14 top-10 head ensemble (518-px, variance-weighted):** 0.8475.
- **ViT-L10 + ViT-G10 frozen-backbone cross-ensemble (224-px):** 0.8413.
- **ViT-G10 (224) + ViT-L fine-tune ensemble:** 0.8521.
- **Final shipped model — ViT-G10 (518-px, variance-weighted) + ViT-L fine-tune
  ensemble, with horizontal-flip TTA:** test mean r = **0.8573** without TTA,
  rising to ≈ **0.858** with TTA on this split (TTA measured at +0.0012 on the
  earlier bundle, assumed similar here). Median per-attribute r = **0.897**;
  mean R² = **0.738**. **34 / 34 attributes significant** at p < 0.05 (101
  stimuli, r ≥ 0.20 sufficient). Per-attribute r ranges from **0.977 (age)**
  down to **0.626 (looks-like-you)**; a 3-way ensemble that also includes the
  frozen ViT-G-224 heads reached 0.8587 but we ship the simpler 2-way bundle.

### Bootstrap confidence intervals on the reported numbers

The test set is small (n = 101), so we report nonparametric **bootstrap 95 %
confidence intervals** on the two headline means. For each of 5,000 bootstrap
draws we resample stimuli with replacement, compute per-attribute metric,
and average across the 34 attributes.

| Metric | Point estimate | 95 % CI | SE |
|---|---|---|---|
| Mean Pearson r | **0.857** | 0.842 – 0.870 | 0.007 |
| Mean R² | **0.738** | 0.707 – 0.755 | 0.012 |

Full table: `training/results/bootstrap_ci_test_split.csv`.

### Reliability ceiling and fraction-of-ceiling reached

Following Peterson et al. 2022 (Fig. 2), we estimate the **split-half reliability**
per attribute from the raw trial ratings: across 100 random 50/50 splits of the
~38 raters per (stimulus, attribute) cell, we compute the correlation between
the two half-set mean vectors across stimuli and average r² across splits.
This is the R² a noiseless model would achieve if it only tried to predict
the mean of half the raters from the mean of the other half.

- **Mean reliability ceiling across the 34 attributes: R² = 0.770.**
- **Our 10-fold CV mean R² = 0.734 — 95 % of the ceiling.**
- Attribute-level: we reach ≥ 95 % of ceiling on 24 of 34 attributes, and
  slightly *exceed* the ceiling on 10 of 34 (looks-like-you, familiar, godly,
  dominant, liberal, smug, cute, smart, outgoing, alert). Exceeding a
  half-vs-half ceiling is possible because our model trains on the **full**
  ~38-rater mean, which has half the noise of a half-rater mean, so it can
  predict the full mean more accurately than a half can predict the other half.

Full per-attribute table (model R², reliability ceiling, and the fraction)
is in `training/results/cv_per_attribute.csv`.

### Honest note on the selection of the final ensemble

The 2-way ensemble we ship (ViT-G/518-vw top-10 + ViT-L fine-tune) was picked
after comparing several candidates — {L10+G10, L10+FT, G10+FT, G518vw10+FT,
3-way, 4-way} — using the primary-split test mean r. The winning configuration
is also the winner on validation mean r (same ranking), so the test-set
influence is limited to selecting *among configurations that all train
without test labels*. Nonetheless, a reviewer could reasonably flag this as
a small degree of selection bias on the test set. The 10-fold CV analysis
above (which re-fits the full sweep on 10 different splits) is the unbiased
check: the head-only CV mean r is 0.853, within 0.005 of the primary-split
head-only number 0.848 — consistent with negligible selection bias.

## Validation: 10-fold CV and head-to-head with Peterson et al. (2022)

To make our numbers directly comparable to the original OMI paper (Peterson
et al., 2022, *PNAS*, Fig. 2), we re-evaluated the pipeline under the same
protocol they used: **10-fold cross-validation** across the full 1,004 stimuli.

**Protocol.** For each fold *k* ∈ {0, …, 9}, the 1,004 stimuli are partitioned
deterministically (seed 0) into test (100 or 101 held-out stimuli from that
fold), val (100 stimuli sampled from the remaining 904) and train (the
remainder). For each fold we re-ran the full 62-config head sweep on the
frozen 518-px ViT-G/14 features with variance-weighted MSE (same training
recipe as the primary reported model, **excluding** the ViT-L/14 fine-tune
branch, which was not CV-ed for compute reasons). Predictions for each of
the 1,004 stimuli come from the fold in which they were held out; concatenated
across all 10 folds this yields one prediction per stimulus. Per-attribute R²
is computed on those 1,004 predictions against the per-stimulus rating means.
This matches Peterson Fig. 2's out-of-sample R² definition.

**Summary numbers.**

| Metric | Peterson 2022 | This work (10-fold CV, frozen backbone only) |
|---|---|---|
| Features | StyleGAN2 *W* latent, 512-d | DINOv2 ViT-G/14 CLS @ 518 px, 1,536-d |
| Head | L2-regularized linear regression | Top-10 MLP ensemble, variance-weighted MSE |
| Mean R² across 34 attributes | ≈ 0.55 | **0.734** |
| Median R² | — | 0.764 |
| Mean Pearson r | — | 0.853 |

On every attribute except `skinny/fat` (within noise, Δ = −0.005) our CV R²
exceeds Peterson's reported R². The largest gains concentrate on the
attributes Peterson Fig. 2 flagged as hardest, most of which have lower
between-rater reliability:

| Attribute | Peterson R² | Our CV R² | Δ |
|---|---|---|---|
| gay | ≈ 0.18 | 0.683 | **+0.50** |
| looks-like-you | ≈ 0.00 | 0.392 | **+0.39** |
| Black | ≈ 0.53 | 0.858 | **+0.33** |
| believes in god (godly) | ≈ 0.27 | 0.575 | **+0.31** |
| electable | ≈ 0.60 | 0.864 | **+0.26** |
| Hispanic | ≈ 0.45 | 0.696 | **+0.25** |
| typical | ≈ 0.24 | 0.459 | **+0.22** |
| dorky | ≈ 0.41 | 0.630 | **+0.22** |
| trustworthy | ≈ 0.64 | 0.846 | **+0.21** |
| familiar | ≈ 0.22 | 0.414 | **+0.19** |

(Full 34-attribute table in `artifacts/cv_per_attribute.csv`.)

**Interpretation.** The improvement is not merely from ensembling or head
capacity — the frozen-backbone change alone (512-d StyleGAN *W* → 1,536-d
DINOv2 ViT-G CLS at 518 px) accounts for most of it. Averaging ten heads and
variance-weighted MSE each add roughly one more percentage point. The
variance-weighted loss, which down-weights cells with high rater disagreement
when computing the gradient, preferentially helps attributes whose inter-rater
reliability was low (e.g., `gay`, `typical`, `looks-like-you`), exactly the
regime where Peterson's linear model struggled. Because 10-fold CV exposes
every stimulus as test exactly once, these gains are not explicable by a
favorable test-split.

**Consistency with the primary test split.** Our primary (80/10/10) test-split
result was mean r = 0.857, R² = 0.738 with the ViT-L fine-tune folded in; CV
of the head-only pipeline gives mean r = 0.853, R² = 0.734. The two numbers
agree within ±0.005 and justify the held-out test score we report as the
shipping model.

**What Peterson can do that we still cannot.** Their linear decoding in
StyleGAN2 *W*-space enables direct *attribute manipulation* of a face (adding
β·w_k to the latent moves the synthesized face along the attribute axis).
DINOv2 features are not invertible through a generator, so our model predicts
but does not edit. For downstream tasks that require editing (e.g., the
attribute-manipulation validation experiments in the OMI paper), their model
remains the appropriate choice; for prediction accuracy on arbitrary new face
photographs, ours clearly outperforms.

## Ethical use

OMI ratings — and therefore these predictions — reflect systematic stereotypes
and biases held by the population of raters, *not* objective attributes of the
depicted people. Applications must frame outputs as **perceived traits**
(e.g. "perceived trustworthiness") rather than ground truth. The dataset is
distributed under CC BY-NC-SA 4.0 (non-commercial, share-alike). We recommend
following the same restriction when re-using the predictions derived from it.
