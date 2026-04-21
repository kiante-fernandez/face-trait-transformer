"""Per-attribute regression metrics + bootstrap resampling for CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def per_attribute_pearson(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (r, p) arrays of length y_true.shape[1]."""
    n_attr = y_true.shape[1]
    rs = np.empty(n_attr, dtype=np.float64)
    ps = np.empty(n_attr, dtype=np.float64)
    for j in range(n_attr):
        r, p = pearsonr(y_true[:, j], y_pred[:, j])
        rs[j] = r
        ps[j] = p
    return rs, ps


def bootstrap_mean_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "pearson_r",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap the mean-across-attributes of a metric over stimuli.

    Resamples stimuli (rows) with replacement, computes per-attribute metric,
    then averages across attributes. Returns a dict with point estimate, mean,
    SE, and symmetric CI bounds.

    Parameters
    ----------
    metric : one of {"pearson_r", "R2"}.
    """
    assert y_true.shape == y_pred.shape
    n = y_true.shape[0]
    n_attr = y_true.shape[1]
    rng = np.random.default_rng(seed)

    def _per_attr(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        if metric == "pearson_r":
            rs = np.empty(n_attr)
            for j in range(n_attr):
                r, _ = pearsonr(yt[:, j], yp[:, j])
                rs[j] = r
            return rs
        elif metric == "R2":
            ss_res = np.sum((yt - yp) ** 2, axis=0)
            tot = yt.mean(axis=0)
            ss_tot = np.sum((yt - tot) ** 2, axis=0)
            return 1 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)
        raise ValueError(f"unknown metric {metric}")

    point = float(np.mean(_per_attr(y_true, y_pred)))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(_per_attr(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(boot_means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "metric": metric,
        "point": point,
        "boot_mean": float(boot_means.mean()),
        "boot_se": float(boot_means.std(ddof=1)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci": ci,
        "n_stimuli": n,
        "n_attrs": n_attr,
        "n_boot": n_boot,
    }


def summary(
    y_true: np.ndarray, y_pred: np.ndarray, attr_names: list[str]
) -> pd.DataFrame:
    """DataFrame with columns: attribute, pearson_r, pearson_p, R2, mae, rmse."""
    rs, ps = per_attribute_pearson(y_true, y_pred)
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)
    return pd.DataFrame(
        {
            "attribute": attr_names,
            "pearson_r": rs,
            "pearson_p": ps,
            "R2": r2,
            "mae": mae,
            "rmse": rmse,
        }
    )
