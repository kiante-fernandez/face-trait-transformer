# Training pipeline — reproducing the shipped model

This folder contains everything you need to reproduce the weights published at
[`kiante/face-trait-transformer`](https://huggingface.co/kiante/face-trait-transformer).
Nothing in `training/` is installed by `pip install face-trait-transformer` — it
exists purely so the numbers in the paper can be regenerated end-to-end.

## Prerequisites

- The OMI dataset in a sibling folder (the `attribute_means.csv`, `images/`,
  and `attribute_ratings.zip` from https://github.com/jcpeterson/omi).
- Python 3.9+ with `pip install -e ".[training]"` from the repo root.
- ~6 GB free disk for cached features + sweep checkpoints.
- CPU is sufficient; a GPU will be a lot faster for the ViT-G/14 feature
  extraction but nothing else strictly requires it.

## What's already here

- **`splits/splits.json`** — primary 80/10/10 seed-0 stimulus split.
- **`splits/cv_splits/fold_XX.json`** — the 10 CV folds used for the
  Peterson-comparison analysis.
- **`splits/attribute_stats.npz`** — per-(stimulus, attribute) rater mean /
  std / n_raters aggregated from the raw trial-level ratings. This is the
  input to the variance-weighted MSE loss.
- **`results/cv_per_attribute.csv`** — 10-fold CV per-attribute R² / r
  reported in `docs/methods.md`.
- **`results/per_attribute_r_r2.csv`** — primary-split per-attribute R² / r.

## Reproduce a number end-to-end

### 1. Point the scripts at your OMI checkout

```
export OMI_DIR=/path/to/omi        # contains attribute_means.csv + images/ + attribute_ratings.zip
```

### 2. Extract the frozen ViT-G/14 features at 518 px (one-time)

```
python -m training.scripts.extract_features \
    --images-dir $OMI_DIR/images \
    --out       training/cache/features_vitg14_518.npy \
    --ids-out   training/cache/stimulus_ids_vitg14_518.npy \
    --model     dinov2_vitg14 \
    --image-size 518 \
    --batch-size 4 \
    --num-workers 4
```

If you already have `attribute_ratings.csv` unzipped, otherwise run:
```
unzip $OMI_DIR/attribute_ratings.zip -d $OMI_DIR/
python -m training.scripts.build_raw_targets \
    --ratings $OMI_DIR/attribute_ratings.csv \
    --labels  $OMI_DIR/attribute_means.csv \
    --out     training/splits/attribute_stats.npz
```

### 3. Sweep the 62 head configs (variance-weighted MSE)

Run each config (or distribute across a cluster). One config looks like:

```
python -m training.scripts.train \
    --features training/cache/features_vitg14_518.npy \
    --ids      training/cache/stimulus_ids_vitg14_518.npy \
    --labels   $OMI_DIR/attribute_means.csv \
    --splits   training/splits/splits.json \
    --stats    training/splits/attribute_stats.npz \
    --head mlp --hidden 2048 --dropout 0.3 --lr 1e-3 --wd 1e-4 \
    --epochs 400 --patience 40 --seed 0 \
    --out training/sweep_out/task053.pt
```

The full 62-config grid is in `hoffman2/sweep_configs.json` (ignore the
`hoffman2/` naming — it's just a portable JSON).

### 4. Partial end-to-end fine-tune of ViT-L/14 (last block)

```
python -m training.scripts.finetune \
    --images-dir $OMI_DIR/images \
    --labels     $OMI_DIR/attribute_means.csv \
    --splits     training/splits/splits.json \
    --model      dinov2_vitl14 \
    --unfreeze-blocks 1 \
    --hidden 1024 --dropout 0.2 \
    --lr 1e-4 --head-lr 1e-3 --wd 1e-4 \
    --epochs 25 --patience 6 \
    --batch-size 16 --num-workers 8 \
    --out training/checkpoints/finetune_vitl_lb1.pt
```

### 5. Assemble the publishable bundle

```
python -m training.scripts.export_bundle \
    --backbone dinov2_vitg14 --ckpt-dir training/sweep_out --top-k 10 \
        --image-size 518 --group-name vitg_518_vw \
    --finetune training/checkpoints/finetune_vitl_lb1.pt \
    --labels $OMI_DIR/attribute_means.csv \
    --out training/bundle
```

At that point `TraitPredictor.from_bundle("training/bundle")` works exactly
like `TraitPredictor.from_pretrained(...)`.

## 10-fold CV (the Peterson-comparison analysis)

```
# 1. split generation (already committed under training/splits/cv_splits)
python -m training.scripts.make_cv_splits

# 2. sweep 62 configs × 10 folds = 620 runs (distribute somehow)
#    each config/fold writes into training/sweep_cv_out/fold_XX/...

# 3. aggregate into a single per-attribute R² / r table
python -m training.scripts.aggregate_cv \
    --cv-out     training/sweep_cv_out \
    --splits-dir training/splits/cv_splits \
    --features   training/cache/features_vitg14_518.npy \
    --ids        training/cache/stimulus_ids_vitg14_518.npy \
    --labels     $OMI_DIR/attribute_means.csv \
    --top-k 10 \
    --out-csv    training/results/cv_per_attribute.csv
```

## Cluster notes

The original runs used UCLA Hoffman2 (Univa Grid Engine). The `hoffman2/`
folder in the old repo contained site-specific `qsub` wrappers for
(1) chunked feature extraction as an SGE array, (2) the 62-way head sweep
and the 620-task CV sweep as SGE arrays, and (3) the fine-tune as an 8–16
slot CPU job with a 20 h walltime cap. These are deliberately *not* shipped
here — they're portable enough to translate to SLURM/PBS/your favorite
scheduler. The pure-Python scripts above are what matters.
