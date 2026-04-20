"""OMI label loading and deterministic train/val/test splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_labels(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read an OMI-style attribute_means.csv.

    Returns
    -------
    ids        : int64 array, shape (N,)
    Y          : float32 array, shape (N, n_attr), values in [0, 1] (rescaled from 0-100)
    attr_names : list of attribute column names in file order
    """
    df = pd.read_csv(csv_path)
    attr_names = [c for c in df.columns if c != "stimulus"]
    ids = df["stimulus"].to_numpy(dtype=np.int64)
    Y = df[attr_names].to_numpy(dtype=np.float32) / 100.0
    return ids, Y, attr_names


def make_splits(
    ids: np.ndarray,
    seed: int = 0,
    fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[int]]:
    """Deterministic 80/10/10 shuffle by stimulus id."""
    assert abs(sum(fracs) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_train = int(round(fracs[0] * len(ids)))
    n_val = int(round(fracs[1] * len(ids)))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return {
        "train": ids[train_idx].tolist(),
        "val": ids[val_idx].tolist(),
        "test": ids[test_idx].tolist(),
        "seed": seed,
    }


def save_splits(splits: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def index_by_id(ids: np.ndarray, wanted_ids: list[int]) -> np.ndarray:
    """Return array indices of wanted_ids within ids, preserving order."""
    pos = {int(i): k for k, i in enumerate(ids)}
    return np.array([pos[int(i)] for i in wanted_ids], dtype=np.int64)
