"""End-to-end orchestration: data -> features -> models -> evaluation -> forecast."""

from __future__ import annotations

import json
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch

from .config import Config
from .eda import run_eda
from .evaluate import compute_metrics, error_segmentation
from .features import GBM_FEATURE_COLS, LSTM_FEATURE_COLS, build_features
from .forecast import build_future_frame, forecast_table, recursive_gbm_forecast
from .ingestion import load_data, prepare_store_data
from .models import GbmForecaster, LstmForecaster, SarimaForecaster

MODEL_ORDER = ["SARIMA", "LSTM", "LightGBM"]
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

ProgressCallback = Callable[[int, str], None]


def _to_sales(log_pred: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(log_pred, dtype=float))


def _compute_store_insights(df: pd.DataFrame) -> dict:
    dow = df.groupby("day_of_week")["Sales"].mean()
    promo0 = df[df["Promo"] == 0]["Sales"].mean()
    promo1 = df[df["Promo"] == 1]["Sales"].mean()
    h0 = df[df["holiday_binary"] == 0]["Sales"].mean()
    h1 = df[df["holiday_binary"] == 1]["Sales"].mean()
    return {
        "dow_mean": dow,
        "peak_dow": DOW_NAMES[int(dow.idxmax())],
        "trough_dow": DOW_NAMES[int(dow.idxmin())],
        "avg_sales": float(df["Sales"].mean()),
        "promo_lift_pct": float(100 * (promo1 / promo0 - 1)),
        "promo_days_pct": float(100 * df["Promo"].mean()),
        "holiday_delta_pct": float(100 * (h1 / h0 - 1)),
        "holiday_days_pct": float(100 * df["holiday_binary"].mean()),
    }


def _fit_sarima(train_log: pd.Series) -> SarimaForecaster:
    return SarimaForecaster().fit(train_log)


def _save_summary(res: dict, cfg: Config) -> None:
    """Persist a compact machine-readable summary alongside the model artifacts."""
    quantiles = {
        m: np.quantile(r, list(cfg.quantiles)).tolist()
        for m, r in res["residuals"].items()
    }
    summary = {
        "store_id": int(res["store_id"]),
        "split_ratio": float(cfg.split_ratio),
        "split_idx": int(res["split_idx"]),
        "test_start": str(pd.to_datetime(res["dates_test"][0]).date()),
        "test_end": str(pd.to_datetime(res["dates_test"][-1]).date()),
        "best_model": res["best_model"],
        "metrics": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in res["metrics"].items()
        },
        "insights": {
            "avg_sales": float(res["insights"]["avg_sales"]),
            "peak_dow": res["insights"]["peak_dow"],
            "trough_dow": res["insights"]["trough_dow"],
            "promo_lift_pct": float(res["insights"]["promo_lift_pct"]),
            "promo_days_pct": float(res["insights"]["promo_days_pct"]),
            "holiday_delta_pct": float(res["insights"]["holiday_delta_pct"]),
            "holiday_days_pct": float(res["insights"]["holiday_days_pct"]),
            "dow_mean": {int(k): float(v) for k, v in res["insights"]["dow_mean"].items()},
        },
        "feature_importance": {
            k: float(v) for k, v in res["feature_importance"].head(20).items()
        },
        "segmentation": {
            k: v.to_dict(orient="records") for k, v in res["segmentation"].items()
        },
        "residual_quantiles": quantiles,
        "promo_scenario": {
            "none_avg": float(np.mean(res["promo_scenario"]["none"])),
            "full_avg": float(np.mean(res["promo_scenario"]["full"])),
        },
        "dates_test": [str(d) for d in res["dates_test"]],
    }
    path = cfg.models_dir / f"store_{res['store_id']}_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_store(store_id: int, cfg: Config, progress: ProgressCallback | None = None) -> dict:
    """Run the full pipeline for one store and return structured results."""
    def report(percent: int, msg: str) -> None:
        if progress is not None:
            progress(percent, msg)

    cfg.ensure_dirs()
    report(2, "Loading data")
    train, store = load_data(cfg)
    store_row = store[store["Store"] == store_id].iloc[0]
    df = prepare_store_data(train, store, store_id)
    df = build_features(df, store_row, cfg)

    report(12, "Exploratory data analysis")
    eda_paths = run_eda(df, store_id, cfg)

    split_idx = int(len(df) * cfg.split_ratio)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    y_test_sales = test_df["Sales"].values
    y_train_sales = train_df["Sales"].values
    test_dates = test_df["Date"].values

    results: dict = {
        "store_id": store_id,
        "store_row": store_row,
        "df": df,
        "split_idx": split_idx,
        "dates_test": test_dates,
        "y_test_sales": y_test_sales,
        "y_train_sales": y_train_sales,
        "figures": eda_paths,
        "metrics": {},
        "residuals": {},
        "insights": _compute_store_insights(df),
    }

    report(20, "Fitting SARIMA baseline")
    sarima = _fit_sarima(train_df["log_sales"])
    sarima_fc_log, _ = sarima.predict(len(test_df))
    sarima_pred_sales = _to_sales(sarima_fc_log)
    sarima_resid = (test_df["log_sales"].values - sarima_fc_log)
    results["metrics"]["SARIMA"] = compute_metrics(
        y_test_sales, sarima_pred_sales, y_train_sales
    )
    results["metrics"]["SARIMA_recursive"] = results["metrics"]["SARIMA"]
    results["residuals"]["SARIMA"] = sarima_resid
    results["residuals"]["SARIMA_recursive"] = sarima_resid
    sarima.update(test_df["log_sales"].values)
    joblib.dump(sarima, cfg.models_dir / f"store_{store_id}_sarima.joblib")

    valid = df["log_sales_lag_30"].notna()
    feat = df.loc[valid, GBM_FEATURE_COLS].reset_index(drop=True)
    target = df.loc[valid, "log_sales"].reset_index(drop=True)
    valid_positions = np.where(valid.values)[0]
    first_valid = valid_positions[0]
    split_in_valid = int(np.searchsorted(valid_positions, split_idx))
    X_train = feat.iloc[:split_in_valid]
    y_train = target.iloc[:split_in_valid]
    X_test = feat.iloc[split_in_valid:]

    report(45, "Tuning + fitting LightGBM")
    gbm = GbmForecaster().fit(X_train, y_train, tune=True)
    gbm_one_step_log = gbm.predict(X_test)
    gbm_one_step_sales = _to_sales(gbm_one_step_log)
    gbm_test_log_actual = target.iloc[split_in_valid:].values
    gbm_one_step_resid = gbm_test_log_actual - gbm_one_step_log
    results["metrics"]["LightGBM"] = compute_metrics(
        y_test_sales, gbm_one_step_sales, y_train_sales
    )
    results["residuals"]["LightGBM"] = gbm_one_step_resid

    test_exog = df.iloc[split_idx:].copy()
    for c in GBM_FEATURE_COLS:
        if c.startswith("log_sales_lag") or c.startswith("log_sales_roll"):
            test_exog[c] = np.nan
    gbm_rec_log = recursive_gbm_forecast(gbm, train_df["log_sales"].values, test_exog, cfg)
    gbm_rec_sales = _to_sales(gbm_rec_log)
    results["metrics"]["LightGBM_recursive"] = compute_metrics(
        y_test_sales, gbm_rec_sales, y_train_sales
    )
    results["residuals"]["LightGBM_recursive"] = (
        test_df["log_sales"].values - gbm_rec_log
    )
    results["feature_importance"] = gbm.feature_importance()
    results["best_params"] = gbm.best_params
    joblib.dump(gbm, cfg.models_dir / f"store_{store_id}_lgb.joblib")
    results["feature_importance"].to_csv(cfg.models_dir / f"store_{store_id}_lgb_importance.csv")

    report(70, "Training LSTM")
    lstm = LstmForecaster(cfg.lstm_params, seed=cfg.seed)
    lstm.fit(df, LSTM_FEATURE_COLS, "log_sales", train_end=split_idx)
    lstm_mask = np.zeros(len(df), dtype=bool)
    lstm_mask[split_idx:] = True
    lstm_one_step_log = lstm.predict(df, lstm_mask)
    test_rows = np.arange(split_idx, len(df))
    lstm_one_step_log_test = lstm_one_step_log[test_rows]
    lstm_one_step_sales = _to_sales(lstm_one_step_log_test)
    lstm_one_step_resid = test_df["log_sales"].values - lstm_one_step_log_test
    results["metrics"]["LSTM"] = compute_metrics(
        y_test_sales, lstm_one_step_sales, y_train_sales
    )
    results["residuals"]["LSTM"] = lstm_one_step_resid

    lstm_test_exog = test_df[LSTM_FEATURE_COLS].copy()
    lstm_test_exog["log_sales"] = np.nan
    lstm_rec_log = lstm.forecast(train_df, lstm_test_exog)
    lstm_rec_sales = _to_sales(lstm_rec_log)
    results["metrics"]["LSTM_recursive"] = compute_metrics(
        y_test_sales, lstm_rec_sales, y_train_sales
    )
    results["residuals"]["LSTM_recursive"] = test_df["log_sales"].values - lstm_rec_log
    torch.save(
        {"state": lstm.model.state_dict(), "scaler": lstm.scaler,
         "feature_cols": lstm.feature_cols, "target_col": lstm.target_col,
         "lookback": lstm.lookback, "params": cfg.lstm_params},
        cfg.models_dir / f"store_{store_id}_lstm.pt",
    )

    seg_promo = error_segmentation(test_df, y_test_sales, gbm_one_step_sales, by="Promo")
    seg_dow = error_segmentation(test_df, y_test_sales, gbm_one_step_sales, by="day_of_week")
    seg_hol = error_segmentation(test_df, y_test_sales, gbm_one_step_sales, by="holiday_binary")
    results["segmentation"] = {"Promo": seg_promo, "day_of_week": seg_dow, "holiday": seg_hol}
    results["recursive_predictions"] = {
        "SARIMA": sarima_pred_sales,
        "LightGBM": gbm_rec_sales,
        "LSTM": lstm_rec_sales,
    }

    horizon_forecasts = {}
    for horizon in cfg.horizons:
        fdf = build_future_frame(df, store_row, cfg, horizon, promo_mode="repeat")
        fc_dates = fdf["Date"].values
        fc = {}

        gbm_log = recursive_gbm_forecast(gbm, df["log_sales"].values, fdf.copy(), cfg)
        gbm_q = np.quantile(gbm_one_step_resid, list(cfg.quantiles))
        fc["LightGBM"] = forecast_table(fc_dates, _to_sales(gbm_log),
                                        gbm_one_step_resid, cfg.quantiles)

        lstm_exog = fdf[LSTM_FEATURE_COLS].copy()
        lstm_exog["log_sales"] = np.nan
        lstm_log = lstm.forecast(df, lstm_exog)
        fc["LSTM"] = forecast_table(fc_dates, _to_sales(lstm_log),
                                    lstm_one_step_resid, cfg.quantiles)

        sarima_log = sarima.forecast_quantiles(horizon)
        sarima_pred = np.column_stack(
            [_to_sales(sarima_log[:, i]) for i in range(sarima_log.shape[1])]
        )
        sarima_df = pd.DataFrame({"date": fc_dates})
        sarima_df["predicted_sales"] = _to_sales((sarima_log[:, 1] + sarima_log[:, 2]) / 2)
        sarima_df["lower_80"] = sarima_pred[:, 1]
        sarima_df["upper_80"] = sarima_pred[:, 2]
        sarima_df["lower_95"] = sarima_pred[:, 0]
        sarima_df["upper_95"] = sarima_pred[:, 3]
        fc["SARIMA"] = sarima_df
        horizon_forecasts[horizon] = fc

    results["forecasts"] = horizon_forecasts

    fdf_none = build_future_frame(df, store_row, cfg, 30, promo_mode="none")
    gbm_none_log = recursive_gbm_forecast(gbm, df["log_sales"].values, fdf_none.copy(), cfg)
    fdf_full = build_future_frame(df, store_row, cfg, 30, promo_mode="full")
    gbm_full_log = recursive_gbm_forecast(gbm, df["log_sales"].values, fdf_full.copy(), cfg)
    results["promo_scenario"] = {
        "none": _to_sales(gbm_none_log),
        "full": _to_sales(gbm_full_log),
    }

    if "LightGBM_recursive" in results["metrics"]:
        results["best_model"] = min(
            ["SARIMA", "LSTM", "LightGBM"],
            key=lambda m: results["metrics"][m + "_recursive"]["RMSE"],
        )
    else:
        results["best_model"] = min(
            ["SARIMA", "LSTM", "LightGBM"],
            key=lambda m: results["metrics"][m]["RMSE"],
        )
    report(95, "Saving artifacts and summary")
    _save_summary(results, cfg)
    report(100, "Done")
    return results


def forecast_sales(data, horizon: int = 30, **kwargs) -> pd.DataFrame:
    """Public function interface.

    Accepts either a store_id (int) or a per-store DataFrame with columns
    Store/Date/Sales/Promo/StateHoliday/SchoolHoliday/Open, and returns the
    forecast dataframe (date, predicted_sales, lower_80, upper_80,
    lower_95, upper_95) for the requested horizon using the best model.
    """
    cfg = Config()
    if isinstance(data, int):
        store_id = data
    else:
        store_id = int(data["Store"].iloc[0])
    res = run_store(store_id, cfg)
    best = res["best_model"]
    fc = res["forecasts"][horizon][best]
    return fc.reset_index(drop=True)


def run_pipeline(cfg: Config | None = None) -> dict:
    """Run every configured store and return per-store results."""
    cfg = cfg or Config()
    cfg.ensure_dirs()
    all_results = {}
    for store_id in cfg.stores:
        print(f"[pipeline] Running store {store_id}")
        all_results[store_id] = run_store(store_id, cfg)
        print(f"[pipeline] Store {store_id} complete")
    return all_results
