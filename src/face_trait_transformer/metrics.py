"""Per-attribute regression metrics matching what we report in the paper."""
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
