"""Multi-horizon forecasting: recursive ML forecasts, scenarios and confidence bands."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .features import GBM_FEATURE_COLS, build_future_features


def build_promo_scenario(hist_df: pd.DataFrame, horizon: int, mode: str = "repeat") -> list[int]:
    """Promo flags for the forecast window.

    mode="repeat": repeat the store's most recent weekly promo pattern.
    mode="none":   no promotions (conservative scenario).
    mode="full":   promotion every day (optimistic scenario).
    """
    pattern = hist_df["Promo"].values[-7:]
    if mode == "repeat":
        return [int(pattern[i % 7]) for i in range(horizon)]
    if mode == "none":
        return [0] * horizon
    return [1] * horizon


def build_future_frame(
    hist_df: pd.DataFrame, store_row: pd.Series, cfg: Config,
    horizon: int, promo_mode: str = "repeat",
) -> pd.DataFrame:
    """Future exogenous/calendar feature frame for `horizon` days after the data end."""
    start = hist_df["Date"].max() + pd.Timedelta(days=1)
    fdf = build_future_features(start, horizon, store_row)
    fdf["Promo"] = build_promo_scenario(hist_df, horizon, promo_mode)
    for c in GBM_FEATURE_COLS:
        if c.startswith("log_sales_lag") or c.startswith("log_sales_roll"):
            fdf[c] = np.nan
    return fdf


def recursive_gbm_forecast(
    model, log_history: np.ndarray, fdf: pd.DataFrame, cfg: Config,
) -> np.ndarray:
    """Recursive multi-step forecast for the GBM in log scale."""
    log_h = list(np.asarray(log_history, dtype=float))
    preds = np.empty(len(fdf))
    for i in range(len(fdf)):
        row = fdf.index[i]
        for lag in cfg.lag_days:
            fdf.loc[row, f"log_sales_lag_{lag}"] = log_h[-lag] if len(log_h) >= lag else np.nan
        for w in cfg.roll_days:
            window = log_h[-w:]
            fdf.loc[row, f"log_sales_roll_mean_{w}"] = float(np.mean(window))
            fdf.loc[row, f"log_sales_roll_std_{w}"] = float(np.std(window))
        x = fdf.loc[[row], GBM_FEATURE_COLS]
        pred_log = float(model.predict(x)[0])
        preds[i] = pred_log
        log_h.append(pred_log)
    return preds


def residual_quantiles(residuals_log: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    """Empirical quantiles of log-scale residuals used to build prediction bands."""
    return np.quantile(np.asarray(residuals_log), list(quantiles))


def bands_from_log(pred_log: np.ndarray, residual_q: np.ndarray,
                   quantiles: tuple[float, ...]) -> pd.DataFrame:
    """Expand point forecasts into the deliverable band format (original units)."""
    out = pd.DataFrame({"predicted_sales": np.expm1(pred_log)})
    for q in quantiles:
        if q < 0.5:
            out[f"lower_{int((1 - 2 * q) * 100)}"] = np.expm1(pred_log + residual_q[q])
        else:
            out[f"upper_{int((2 * q - 1) * 100)}"] = np.expm1(pred_log + residual_q[q])
    return out


def forecast_table(dates, pred_sales, residuals_log, quantiles=(0.025, 0.10, 0.90, 0.975)) -> pd.DataFrame:
    """Build the standard forecast dataframe: date + point forecast + 80/95% bands."""
    q = residual_quantiles(residuals_log, quantiles)
    pred_log = np.log1p(pred_sales)
    bands = bands_from_log(pred_log, dict(zip(quantiles, q)), quantiles)
    bands.insert(0, "date", np.asarray(dates))
    return bands
