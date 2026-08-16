"""Evaluation metrics and residual/segmentation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonal: int = 7) -> float:
    y_train = np.asarray(y_train, dtype=float)
    naive_err = np.mean(np.abs(y_train[seasonal:] - y_train[:-seasonal]))
    if naive_err == 0:
        return np.nan
    return float(mae(y_true, y_pred) / naive_err)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train),
    }


def error_segmentation(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, by: str
) -> pd.DataFrame:
    """MAE/RMSE grouped by a categorical column (promo, weekday, holiday...)."""
    t = pd.DataFrame({"seg": df[by].values, "y": y_true, "p": y_pred})
    rows = []
    for seg, g in t.groupby("seg", dropna=False):
        e = np.abs(g["y"] - g["p"])
        rows.append({
            "segment": seg, "n": len(g), "MAE": e.mean(),
            "RMSE": np.sqrt((e**2).mean()),
            "mean_actual": g["y"].mean(),
            "MAPE": mape(g["y"].values, g["p"].values),
        })
    return pd.DataFrame(rows).sort_values("MAE", ascending=False).reset_index(drop=True)
