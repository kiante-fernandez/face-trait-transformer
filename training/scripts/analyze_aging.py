"""Validity analysis of the OMI model's predictions on GIRAF aging images.

Checks implemented:
  1. Predicted `age` ↑ with age_group (young → middle → old).
  2. Predicted `happy` ↑ on happy-emotion images vs the other emotion labels.
  3. Predicted `gender` attr (0 = feminine, 100 = masculine) matches coded gender.
  4. Predicted demographic attrs highest on matching coded demographic.

Writes into the --out-dir:
    age_by_group.png            box+strip plot of predicted age per age_group
    happy_by_emotion.png        box+strip plot of predicted happy per emotion
    gender_by_coded.png         box+strip plot of predicted gender per coded gender
    demographic_confusion.png   heatmap of mean predicted demographic attr per coded group
    summary_stats.csv           numeric summaries + statistical tests
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, spearmanr, ttest_ind


AGE_ORDER = ["young", "middle", "old"]
EMOTION_ORDER = ["happy", "surprise", "neutral", "sad", "disgust", "fear", "angry"]
GENDER_ORDER = ["female", "male"]
DEMOGRAPHIC_ORDER = ["african", "asian", "indian", "latin", "middle_eastern", "white"]
DEMOGRAPHIC_TO_ATTR = {
    "african": "black",
    "asian": "asian",
    "indian": "asian",        # OMI has no separate South-Asian category
    "latin": "hispanic",
    "middle_eastern": "middle-eastern",
    "white": "white",
}


def _strip_box(ax, sub_df, cat_col, val_col, order, title, ylabel):
    groups = [sub_df.loc[sub_df[cat_col] == k, val_col].dropna().to_numpy() for k in order]
    ax.boxplot(groups, positions=range(len(order)), widths=0.55,
               showfliers=False, medianprops=dict(color="#d62728", lw=1.4),
               boxprops=dict(lw=1), whiskerprops=dict(lw=1), capprops=dict(lw=1))
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        xs = i + rng.uniform(-0.15, 0.15, size=len(g))
        ax.scatter(xs, g, s=8, alpha=0.35, color="#2a5dab")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.2)


def _age_analysis(df: pd.DataFrame, out: Path) -> list[dict]:
    sub = df.dropna(subset=["age_group", "age"])
    if sub.empty:
        return []
    rows = [{"test": "age_vs_group", "group": g,
             "n": int((sub.age_group == g).sum()),
             "mean": float(sub.loc[sub.age_group == g, "age"].mean()),
             "std": float(sub.loc[sub.age_group == g, "age"].std()),
             "median": float(sub.loc[sub.age_group == g, "age"].median())}
            for g in AGE_ORDER]
    cat = sub["age_group"].map({k: i for i, k in enumerate(AGE_ORDER)})
    rho, p_rho = spearmanr(cat, sub["age"])
    stat, p_f = f_oneway(*[sub.loc[sub.age_group == g, "age"].to_numpy() for g in AGE_ORDER])
    rows.append({"test": "age_vs_group", "group": "spearman", "n": len(sub),
                 "mean": float(rho), "std": float(p_rho)})
    rows.append({"test": "age_vs_group", "group": "anova",    "n": len(sub),
                 "mean": float(stat), "std": float(p_f)})

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _strip_box(ax, sub, "age_group", "age", AGE_ORDER,
               f"Predicted age vs coded age group  (Spearman ρ = {rho:.3f}, p = {p_rho:.1e})",
               "predicted age (0–100)")
    fig.tight_layout(); fig.savefig(out / "age_by_group.png", dpi=160); plt.close(fig)
    return rows


def _happy_analysis(df: pd.DataFrame, out: Path) -> list[dict]:
    sub = df.dropna(subset=["emotion", "happy"])
    if sub.empty:
        return []
    rows = [{"test": "happy_vs_emotion", "group": e,
             "n": int((sub.emotion == e).sum()),
             "mean": float(sub.loc[sub.emotion == e, "happy"].mean()),
             "std": float(sub.loc[sub.emotion == e, "happy"].std()),
             "median": float(sub.loc[sub.emotion == e, "happy"].median())}
            for e in EMOTION_ORDER if (sub.emotion == e).any()]
    if (sub.emotion == "happy").any() and (sub.emotion == "neutral").any():
        t, p = ttest_ind(sub.loc[sub.emotion == "happy", "happy"],
                         sub.loc[sub.emotion == "neutral", "happy"], equal_var=False)
        rows.append({"test": "happy_vs_emotion", "group": "happy_vs_neutral_t",
                     "n": len(sub), "mean": float(t), "std": float(p)})
    groups = [sub.loc[sub.emotion == e, "happy"].to_numpy() for e in EMOTION_ORDER if (sub.emotion == e).any()]
    if len(groups) >= 2:
        stat, p_f = f_oneway(*groups)
        rows.append({"test": "happy_vs_emotion", "group": "anova",
                     "n": len(sub), "mean": float(stat), "std": float(p_f)})

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _strip_box(ax, sub, "emotion", "happy", EMOTION_ORDER,
               "Predicted happy vs coded emotion", "predicted happy (0–100)")
    fig.tight_layout(); fig.savefig(out / "happy_by_emotion.png", dpi=160); plt.close(fig)
    return rows


def _gender_analysis(df: pd.DataFrame, out: Path) -> list[dict]:
    if "gender_coded" not in df.columns or "gender" not in df.columns:
        return []
    sub = df.dropna(subset=["gender_coded", "gender"])
    if sub.empty:
        return []
    rows = [{"test": "gender_vs_coded", "group": g,
             "n": int((sub.gender_coded == g).sum()),
             "mean": float(sub.loc[sub.gender_coded == g, "gender"].mean()),
             "std": float(sub.loc[sub.gender_coded == g, "gender"].std()),
             "median": float(sub.loc[sub.gender_coded == g, "gender"].median())}
            for g in GENDER_ORDER if (sub.gender_coded == g).any()]
    if (sub.gender_coded == "female").any() and (sub.gender_coded == "male").any():
        t, p = ttest_ind(sub.loc[sub.gender_coded == "male",   "gender"],
                         sub.loc[sub.gender_coded == "female", "gender"], equal_var=False)
        rows.append({"test": "gender_vs_coded", "group": "male_vs_female_t",
                     "n": len(sub), "mean": float(t), "std": float(p)})
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    _strip_box(ax, sub, "gender_coded", "gender", GENDER_ORDER,
               "Predicted gender (0 = feminine, 100 = masculine) vs coded gender",
               "predicted gender attr (0–100)")
    fig.tight_layout(); fig.savefig(out / "gender_by_coded.png", dpi=160); plt.close(fig)
    return rows


def _demographic_analysis(df: pd.DataFrame, out: Path) -> list[dict]:
    sub = df.dropna(subset=["demographic"])
    if sub.empty:
        return []
    attrs = [a for a in ["black", "white", "asian", "hispanic",
                          "middle-eastern", "islander", "native"] if a in sub.columns]
    heat = sub.groupby("demographic")[attrs].mean().reindex(DEMOGRAPHIC_ORDER).dropna(how="all")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index)
    ax.set_xlabel("predicted OMI attribute (0–100)")
    ax.set_ylabel("coded demographic")
    ax.set_title("Mean predicted demographic attribute per coded demographic")
    for i, row in enumerate(heat.values):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v < 50 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(out / "demographic_confusion.png", dpi=160); plt.close(fig)

    rows = []
    for code, row in heat.iterrows():
        expected = DEMOGRAPHIC_TO_ATTR.get(code)
        argmax_attr = row.idxmax()
        rows.append({
            "test": "demographic_vs_coded",
            "group": code,
            "n": int((sub.demographic == code).sum()),
            "mean": float(row.get(expected, float("nan"))) if expected else float("nan"),
            "std": float(row.max()),
            "median": None,
            "note": f"expected_attr={expected}  argmax_attr={argmax_attr}",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    rows: list[dict] = []
    rows += _age_analysis(df, args.out_dir)
    rows += _happy_analysis(df, args.out_dir)
    rows += _gender_analysis(df, args.out_dir)
    rows += _demographic_analysis(df, args.out_dir)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "summary_stats.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"Figures + summary saved to {args.out_dir}")


if __name__ == "__main__":
    main()
