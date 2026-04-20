# External validity check on the GIRAF aging-faces dataset

A worked example of running `TraitPredictor` on a never-seen dataset and
using its metadata to check the predictions' construct validity.

**Dataset.** GIRAF (Generatively Inferred Representations of Aging Faces) —
~1,088 real and synthetic face photographs across three age groups (young,
middle, old), seven emotion labels, and six demographic codes embedded in the
filename. Downloadable separately; we don't redistribute here.

**Protocol.** Run `TraitPredictor.from_pretrained("kiante/face-trait-transformer")`
across every image; compare predicted traits to the metadata labels.

## Headline results

| Check | Outcome |
|---|---|
| Predicted `age` ↑ with age group | Spearman ρ = **0.920**, ANOVA F = 3178, p ≈ 0 |
| Predicted `happy` ↑ on happy-emotion images vs others | t (happy vs neutral) = **54.1**; ANOVA F = 937 |
| Predicted `gender` matches coded gender | t (male vs female) = **102.6**; clean separation |
| Predicted demographic attrs match coded groups | white/African clean; Latin/Middle-Eastern biased toward "white" (known OMI rater bias) |

## Reproducing this example

```bash
# 1. predict
python -m training.scripts.apply_to_aging \
    --root /path/to/aging_images \
    --bundle /path/to/bundle \
    --out examples/aging_validation/predictions.csv \
    --batch-size 4

# 2. analyze
python -m training.scripts.analyze_aging \
    --predictions examples/aging_validation/predictions.csv \
    --out-dir examples/aging_validation
```

The full `predictions.csv` (1,088 rows, 39 cols) and four validity figures live
alongside this README.

## What to read it as

This is a *validity* check — does the model produce sensible outputs on an
independent dataset whose metadata we can use as a sanity check? Yes for age,
emotion, gender; the demographic predictions surface real biases in OMI's
rater pool. Treat the demographic outputs with the caveats in
`docs/model_card.md`.
