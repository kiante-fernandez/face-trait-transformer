"""Within/between-identity validity analysis of trait predictions on CelebA.

500 photos of 50 distinct celebrities (10 per identity). No per-image age /
emotion / demographic labels, but we have the identity grouping. The core
validity test: a good model should

  (a) produce consistent traits for the 10 photos of the same person
      (intra-identity std low, especially for identity-defining attributes
      like age, gender, skin-color, hair-color, ethnic categories);
  (b) discriminate between different people (inter-identity std high on
      those same attributes).

We quantify this with:

  - **Intra-class correlation coefficient (ICC)** per attribute: fraction of
    variance explained by identity. ICC ≈ 1 means "depends on who the person
    is, not which photo"; ICC ≈ 0 means "image noise dominates".
  - **Intra vs inter std** bars per attribute.
  - **Within-identity profile plot** for a few celebrities, showing how
    stable the trait predictions are across their 10 photos.
  - **Trait-space PCA** colored by identity, to see if a nearest-neighbor in
    trait space is usually the same person.

Outputs to --out-dir:

    intra_vs_inter.png          per-attribute intra/inter std comparison
    icc_per_attribute.png       barh of ICC(1,1) per attribute, sorted
    identity_profiles.png       radar strips for 6 sample identities
    trait_pca.png               2-D PCA projection colored by identity
    summary_stats.csv           numeric summaries (per-attr ICC, F-stat, p)

Example:
    python -m training.scripts.analyze_celeba \
        --predictions examples/celeba_validation/predictions.csv \
        --out-dir     examples/celeba_validation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist


# Order attributes the same way the predictor produces them (alphabetically by
# how OMI lists them). At analysis time we just read whatever columns exist.
def _attr_columns(df: pd.DataFrame) -> list[str]:
    non_attr = {"image_name", "face_index", "identity", "filename"}
    return [c for c in df.columns if c not in non_attr]


def _icc_one_way(values: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """One-way random-effects ICC(1,1) and the corresponding F-test.

    values : (N,) array of a single attribute across all images.
    groups : (N,) array of identity labels.

    Returns (ICC1, F_stat, p_value).
    ICC1 = (MS_between − MS_within) / (MS_between + (k − 1) MS_within)
    """
    uniq = np.unique(groups)
    k_counts = np.array([np.sum(groups == g) for g in uniq])
    # If cell sizes are balanced use k = common size; otherwise harmonic mean.
    k = k_counts.mean()
    grand = values.mean()
    ss_between = np.sum([n * (values[groups == g].mean() - grand) ** 2
                         for g, n in zip(uniq, k_counts)])
    ss_within = np.sum([np.sum((values[groups == g] - values[groups == g].mean()) ** 2)
                        for g in uniq])
    df_between = len(uniq) - 1
    df_within = len(values) - len(uniq)
    ms_between = ss_between / df_between
    ms_within = ss_within / max(df_within, 1)
    if ms_within <= 0 or (ms_between + (k - 1) * ms_within) <= 0:
        return float("nan"), float("nan"), float("nan")
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    f_stat = ms_between / ms_within
    p = 1.0 - f_dist.cdf(f_stat, df_between, df_within)
    return float(icc), float(f_stat), float(p)


def _plot_intra_vs_inter(df: pd.DataFrame, attrs: list[str], out: Path) -> pd.DataFrame:
    rows = []
    for a in attrs:
        intra = df.groupby("identity")[a].std(ddof=1).mean()
        identity_means = df.groupby("identity")[a].mean()
        inter = identity_means.std(ddof=1)
        rows.append({"attribute": a, "intra_id_std": intra, "inter_id_std": inter,
                     "inter_over_intra": inter / intra if intra > 0 else np.inf})
    out_df = pd.DataFrame(rows).sort_values("inter_over_intra", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(out_df))
    ax.barh(y - 0.22, out_df["inter_id_std"], 0.42, color="#2a5dab", label="between-identity SD")
    ax.barh(y + 0.22, out_df["intra_id_std"], 0.42, color="#d62728", label="within-identity SD")
    ax.set_yticks(y); ax.set_yticklabels(out_df["attribute"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("standard deviation of predicted rating (0–100 scale)", fontsize=9)
    ax.set_title("Within- vs between-identity variability per attribute "
                 "(500 photos of 50 celebrities)", fontsize=10)
    ax.grid(axis="x", alpha=0.2); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out_df


def _plot_icc(summary: pd.DataFrame, out: Path) -> None:
    df = summary.sort_values("icc", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 9))
    y = np.arange(len(df))
    colors = ["#2a5dab" if v > 0.75 else "#c4a83f" if v > 0.5 else "#c44e52"
              for v in df["icc"]]
    ax.barh(y, df["icc"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y); ax.set_yticklabels(df["attribute"], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(-0.05, 1.05)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("ICC(1,1) — fraction of variance explained by identity", fontsize=9)
    ax.set_title("Per-attribute intraclass correlation across 50 identities × 10 photos",
                 fontsize=10)
    for yi, v in enumerate(df["icc"]):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def _plot_identity_profiles(df: pd.DataFrame, attrs: list[str], out: Path,
                            n_ids: int = 6) -> None:
    """For a few identities, plot the 10-photo min/mean/max per attribute."""
    # Pick identities spanning the range of "distinctiveness" (Euclidean distance of
    # per-id mean vector from the grand mean).
    id_means = df.groupby("identity")[attrs].mean()
    grand = id_means.mean()
    dist = np.sqrt(((id_means - grand) ** 2).sum(axis=1)).sort_values()
    picks = list(dist.iloc[np.linspace(0, len(dist) - 1, n_ids, dtype=int)].index)

    fig, axes = plt.subplots(n_ids, 1, figsize=(11, 1.15 * n_ids), sharex=True)
    if n_ids == 1:
        axes = [axes]
    x = np.arange(len(attrs))
    for ax, ident in zip(axes, picks):
        sub = df[df["identity"] == ident][attrs]
        mn = sub.min().to_numpy()
        mx = sub.max().to_numpy()
        mean = sub.mean().to_numpy()
        ax.fill_between(x, mn, mx, color="#c44e52", alpha=0.18, step="mid",
                        label="per-photo range")
        ax.plot(x, mean, "-o", color="#c44e52", markersize=3, lw=1.2,
                label="identity mean")
        ax.set_ylim(0, 100); ax.set_ylabel(f"ID {ident}", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(attrs, rotation=60, ha="right", fontsize=7)
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(
        "Trait profiles across the 10 photos of 6 sampled identities "
        "(shaded band = per-photo min/max; line = mean)", fontsize=10, y=1.00)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


def _plot_pca(df: pd.DataFrame, attrs: list[str], out: Path) -> None:
    from numpy.linalg import svd

    X = df[attrs].to_numpy(dtype=float)
    Xc = X - X.mean(axis=0)
    U, S, Vt = svd(Xc, full_matrices=False)
    comp = Xc @ Vt[:2].T  # (N, 2)
    var_explained = (S ** 2 / (S ** 2).sum())[:2]
    ids = df["identity"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    uniq = np.unique(ids)
    # tab20 gives 20 distinct colors; with 50 ids we cycle — that's fine since we also mark with size.
    cmap = plt.get_cmap("tab20")
    for i, ident in enumerate(uniq):
        mask = ids == ident
        ax.scatter(comp[mask, 0], comp[mask, 1],
                   s=16, color=cmap(i % 20), alpha=0.75, edgecolors="none")
        cx, cy = comp[mask].mean(axis=0)
        ax.text(cx, cy, str(ident), fontsize=6, ha="center", va="center",
                color="black", weight="bold")
    ax.set_xlabel(f"PC1  ({var_explained[0] * 100:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC2  ({var_explained[1] * 100:.1f}% var)", fontsize=9)
    ax.set_title("Trait-space PCA colored + labeled by identity (tight clusters = within-ID consistency)",
                 fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    attrs = _attr_columns(df)
    if "identity" not in df.columns:
        raise SystemExit("expected 'identity' column; re-run apply_to_celeba.py")
    print(f"Loaded {len(df)} rows across {df.identity.nunique()} identities, "
          f"{len(attrs)} attributes.")

    # 1. ICC + F-test per attribute
    rows = []
    for a in attrs:
        icc, f_stat, p = _icc_one_way(df[a].to_numpy(), df["identity"].to_numpy())
        rows.append({"attribute": a, "icc": icc, "F": f_stat, "p": p})
    icc_df = pd.DataFrame(rows)

    # 2. Intra/inter std per attribute
    iv = _plot_intra_vs_inter(df, attrs, args.out_dir / "intra_vs_inter.png")

    # 3. ICC bar
    _plot_icc(icc_df, args.out_dir / "icc_per_attribute.png")

    # 4. Identity profile strips
    _plot_identity_profiles(df, attrs, args.out_dir / "identity_profiles.png", n_ids=6)

    # 5. PCA
    _plot_pca(df, attrs, args.out_dir / "trait_pca.png")

    # 6. Summary CSV
    summary = icc_df.merge(iv[["attribute", "intra_id_std", "inter_id_std", "inter_over_intra"]],
                           on="attribute")
    summary = summary.sort_values("icc", ascending=False).reset_index(drop=True)
    summary.to_csv(args.out_dir / "summary_stats.csv", index=False)

    # Headline numbers
    mean_icc = summary["icc"].mean()
    median_icc = summary["icc"].median()
    n_high = int((summary["icc"] > 0.75).sum())
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(f"Mean ICC across {len(attrs)} attributes: {mean_icc:.3f}  (median {median_icc:.3f})")
    print(f"{n_high} / {len(attrs)} attributes have ICC > 0.75 (strong identity effect)")
    print(f"Wrote figures + summary to {args.out_dir}")


if __name__ == "__main__":
    main()
