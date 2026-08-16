"""Service layer: dataset resolution, stored-model forecasting, uploads, retraining."""

from __future__ import annotations

import io
import json
import re
import uuid
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sales_forecast.config import Config
from sales_forecast.features import LSTM_FEATURE_COLS, build_features
from sales_forecast.forecast import build_future_frame, recursive_gbm_forecast
from sales_forecast.ingestion import clean_train, load_store, prepare_store_data
from sales_forecast.models import LstmForecaster
from sales_forecast.models.lstm import _LSTMNet

from .config import BASE_CFG, STORE_META_DEFAULTS, UPLOAD_TRAIN_COLS, UPLOADS_DIR

BAND_ORDER = ["lower_95", "lower_80", "upper_80", "upper_95"]
STORE_META_REQUIRED = [
    "Store", "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
]
COLS_RENAME = {
    "store": "Store", "date": "Date", "sales": "Sales", "open": "Open",
    "promo": "Promo", "stateholiday": "StateHoliday",
    "schoolholiday": "SchoolHoliday",
}

_engineered_cache: dict[tuple[str, int], tuple[pd.DataFrame, pd.Series, Config]] = {}


def resolve_cfg(dataset: str) -> Config:
    if dataset in ("", "demo"):
        return BASE_CFG
    if dataset.startswith("upload:"):
        upload_id = dataset.split(":", 1)[1]
        udir = UPLOADS_DIR / upload_id
        if not udir.is_dir():
            raise ValueError(f"Unknown upload dataset: {dataset}")
        return replace(
            BASE_CFG,
            data_dir=udir,
            output_dir=udir / "outputs",
            figures_dir=udir / "outputs" / "figures",
            forecasts_dir=udir / "outputs" / "forecasts",
            models_dir=udir / "outputs" / "models",
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def dataset_label(dataset: str) -> str:
    if dataset in ("", "demo"):
        return "Rossmann demo data"
    if dataset.startswith("upload:"):
        upload_id = dataset.split(":", 1)[1]
        info = _upload_info(upload_id)
        return info.get("original_filename", upload_id) if info else upload_id
    return dataset


def list_datasets() -> list[dict]:
    datasets = [{
        "dataset": "demo",
        "label": dataset_label("demo"),
        "stores": _stores_for_cfg(BASE_CFG),
    }]
    for udir in sorted(UPLOADS_DIR.glob("*")):
        if not udir.is_dir():
            continue
        info = _upload_info(udir.name)
        if info is None:
            continue
        cfg = resolve_cfg(f"upload:{udir.name}")
        datasets.append({
            "dataset": f"upload:{udir.name}",
            "label": info.get("original_filename", udir.name),
            "created_at": info.get("created_at"),
            "stores": _stores_for_cfg(cfg),
        })
    return datasets


def _stores_for_cfg(cfg: Config) -> list[dict]:
    out = []
    seen = set()
    summaries = sorted(cfg.models_dir.glob("store_*_summary.json"))
    for path in summaries:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        sid = summary["store_id"]
        seen.add(sid)
        out.append({
            "store_id": sid,
            "trained": True,
            "best_model": summary["best_model"],
            "test_start": summary["test_start"],
            "test_end": summary["test_end"],
        })
    if cfg is BASE_CFG:
        for sid in cfg.stores:
            if sid not in seen:
                out.append({
                    "store_id": int(sid),
                    "trained": False,
                    "best_model": None,
                    "test_start": None,
                    "test_end": None,
                })
        return out
    try:
        train = _read_train(cfg)
        trained_ids = {s["store_id"] for s in out}
        for sid in sorted(int(s) for s in train["Store"].unique()):
            if sid not in trained_ids:
                out.append({
                    "store_id": sid,
                    "trained": False,
                    "best_model": None,
                    "test_start": None,
                    "test_end": None,
                })
    except Exception:
        pass
    return out


def _read_train(cfg: Config) -> pd.DataFrame:
    return clean_train(pd.read_csv(cfg.data_dir / "train.csv",
                                   parse_dates=["Date"], low_memory=False))


def load_summary(dataset: str, store_id: int) -> dict:
    cfg = resolve_cfg(dataset)
    path = cfg.models_dir / f"store_{store_id}_summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Store {store_id} in '{dataset_label(dataset)}' has not been trained yet."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _engineered_df(dataset: str, store_id: int) -> tuple[pd.DataFrame, pd.Series, Config]:
    key = (dataset, int(store_id))
    if key in _engineered_cache:
        return _engineered_cache[key]
    cfg = resolve_cfg(dataset)
    train = _read_train(cfg)
    store = load_store(cfg.data_dir / "store.csv")
    store_row = store[store["Store"] == int(store_id)].iloc[0]
    df = prepare_store_data(train, store, int(store_id))
    df = build_features(df, store_row, cfg)
    _engineered_cache[key] = (df, store_row, cfg)
    return _engineered_cache[key]


def overview(dataset: str, store_id: int) -> dict:
    summary = load_summary(dataset, store_id)
    cfg = resolve_cfg(dataset)
    figures = []
    for path in sorted(cfg.figures_dir.glob(f"store_{store_id}_*.png")):
        figures.append({
            "name": path.stem.replace(f"store_{store_id}_", ""),
            "url": media_url(dataset, path),
        })
    forecasts = []
    for path in sorted(cfg.forecasts_dir.glob(f"store_{store_id}_*d_*.csv")):
        m = re.match(rf"store_{store_id}_(\d+)d_(\w+)\.csv$", path.name)
        if m:
            forecasts.append({
                "horizon": int(m.group(1)),
                "model": m.group(2),
                "url": media_url(dataset, path),
            })
    return {
        "summary": summary,
        "figures": figures,
        "forecasts": forecasts,
    }


def media_url(dataset: str, path: Path) -> str:
    if dataset in ("", "demo"):
        return "/outputs/" + str(path.relative_to(BASE_CFG.output_dir)).replace("\\", "/")
    return "/uploads/" + str(path.relative_to(UPLOADS_DIR)).replace("\\", "/")


def history(dataset: str, store_id: int, days: int = 120) -> list[dict]:
    df, _, _ = _engineered_df(dataset, store_id)
    tail = df.tail(days)
    return [{"date": str(d.date()), "sales": float(s)} for d, s in zip(tail["Date"], tail["Sales"])]


def run_forecast(dataset: str, store_id: int, horizon: int,
                 model: str = "best", promo_mode: str = "repeat") -> dict:
    df, store_row, cfg = _engineered_df(dataset, int(store_id))
    summary = load_summary(dataset, int(store_id))
    if model in ("best", ""):
        model = summary["best_model"]
    if model not in summary["metrics"]:
        raise ValueError(f"Model '{model}' not available for store {store_id}")

    fdf = build_future_frame(df, store_row, cfg, horizon, promo_mode=promo_mode)
    fc_dates = fdf["Date"].values

    if model == "SARIMA":
        sarima = joblib.load(cfg.models_dir / f"store_{store_id}_sarima.joblib")
        q = sarima.forecast_quantiles(horizon)
        fc = pd.DataFrame({"date": fc_dates})
        fc["predicted_sales"] = np.expm1((q[:, 1] + q[:, 2]) / 2)
        fc["lower_95"] = np.expm1(q[:, 0])
        fc["lower_80"] = np.expm1(q[:, 1])
        fc["upper_80"] = np.expm1(q[:, 2])
        fc["upper_95"] = np.expm1(q[:, 3])
    elif model == "LSTM":
        lstm = _load_lstm(cfg, int(store_id))
        exog = fdf[LSTM_FEATURE_COLS].copy()
        exog["log_sales"] = np.nan
        pred_log = lstm.forecast(df, exog)
        fc = _band_table(fc_dates, pred_log, summary, model)
    elif model == "LightGBM":
        gbm = joblib.load(cfg.models_dir / f"store_{store_id}_lgb.joblib")
        pred_log = recursive_gbm_forecast(gbm, df["log_sales"].values, fdf.copy(), cfg)
        fc = _band_table(fc_dates, pred_log, summary, model)
    else:
        raise ValueError(f"Model '{model}' not supported")

    cfg.forecasts_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.forecasts_dir / f"store_{store_id}_{horizon}d_{model.lower()}.csv"
    fc.to_csv(out_path, index=False)

    records = []
    for row in fc.to_dict(orient="records"):
        records.append({
            "date": str(row["date"]),
            "predicted_sales": round(float(row["predicted_sales"]), 2),
            "lower_80": round(float(row["lower_80"]), 2),
            "upper_80": round(float(row["upper_80"]), 2),
            "lower_95": round(float(row["lower_95"]), 2),
            "upper_95": round(float(row["upper_95"]), 2),
        })

    return {
        "dataset": dataset,
        "store_id": int(store_id),
        "model": model,
        "horizon": int(horizon),
        "promo_mode": promo_mode,
        "dates": [r["date"] for r in records],
        "predicted_sales": [r["predicted_sales"] for r in records],
        "lower_80": [r["lower_80"] for r in records],
        "upper_80": [r["upper_80"] for r in records],
        "lower_95": [r["lower_95"] for r in records],
        "upper_95": [r["upper_95"] for r in records],
        "download_url": media_url(dataset, out_path),
    }


def _band_table(dates, pred_log: np.ndarray, summary: dict, model: str) -> pd.DataFrame:
    q = np.asarray(summary["residual_quantiles"][model], dtype=float)
    fc = pd.DataFrame({"date": np.asarray(dates)})
    fc["predicted_sales"] = np.expm1(pred_log)
    fc["lower_95"] = np.expm1(pred_log + q[0])
    fc["lower_80"] = np.expm1(pred_log + q[1])
    fc["upper_80"] = np.expm1(pred_log + q[2])
    fc["upper_95"] = np.expm1(pred_log + q[3])
    return fc


def _load_lstm(cfg: Config, store_id: int) -> LstmForecaster:
    pt = torch.load(cfg.models_dir / f"store_{store_id}_lstm.pt",
                    map_location="cpu", weights_only=False)
    params = pt["params"]
    lstm = LstmForecaster(params, seed=cfg.seed)
    lstm.scaler = pt["scaler"]
    lstm.feature_cols = pt["feature_cols"]
    lstm.target_col = pt["target_col"]
    lstm._target_idx = lstm.feature_cols.index(lstm.target_col)
    lstm.lookback = pt["lookback"]
    net = _LSTMNet(len(lstm.feature_cols), lstm.hidden, lstm.layers, lstm.dropout)
    net.load_state_dict(pt["state"])
    net.eval()
    lstm.model = net
    return lstm


def create_upload(filename: str, train_content: bytes,
                  store_filename: str | None, store_content: bytes | None) -> dict:
    upload_id = uuid.uuid4().hex[:8]
    udir = UPLOADS_DIR / upload_id
    udir.mkdir(parents=True, exist_ok=True)
    (udir / "train.csv").write_bytes(train_content)
    if store_content is not None and len(store_content) > 0:
        (udir / "store.csv").write_bytes(store_content)
        store = _normalize_store(pd.read_csv(io.BytesIO(store_content)))
        store.to_csv(udir / "store.csv", index=False)
    else:
        store = None

    train = _read_train_from_bytes(train_content)
    stores = sorted(int(s) for s in train["Store"].unique())

    if store is None:
        default = pd.DataFrame({
            "Store": stores,
            **{k: [v] * len(stores) for k, v in STORE_META_DEFAULTS.items()},
        })
        default.to_csv(udir / "store.csv", index=False)

    info = {
        "upload_id": upload_id,
        "created_at": pd.Timestamp.now().isoformat(),
        "original_filename": filename,
        "store_file": store is not None,
        "stores": stores,
    }
    (udir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return info


def _read_train_from_bytes(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df = _normalize_train_cols(df)
    return clean_train(df)


def _normalize_train_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {c.strip().lower(): c for c in df.columns}
    for key, canon in COLS_RENAME.items():
        if key in mapping and canon not in df.columns:
            df.rename(columns={mapping[key]: canon}, inplace=True)
    missing = [c for c in UPLOAD_TRAIN_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Train file is missing required column(s): {', '.join(missing)}")
    return df


def _normalize_store(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Store" not in df.columns:
        mapping = {c.strip().lower(): c for c in df.columns}
        if "store" in mapping:
            df.rename(columns={mapping["store"]: "Store"}, inplace=True)
        else:
            raise ValueError("Store file must contain a Store column")
    for col, default in STORE_META_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    missing = [c for c in STORE_META_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Store file is missing required column(s): {', '.join(missing)}")
    return df


def _upload_info(upload_id: str) -> dict | None:
    info_path = UPLOADS_DIR / upload_id / "info.json"
    if not info_path.is_file():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def retrain(dataset: str, store_id: int, progress=None) -> dict:
    from sales_forecast.pipeline import run_store

    cfg = resolve_cfg(dataset)
    res = run_store(int(store_id), cfg, progress=progress)
    return {
        "dataset": dataset,
        "store_id": int(store_id),
        "best_model": res["best_model"],
        "metrics": {
            m: {k: round(float(v), 3) for k, v in mm.items()}
            for m, mm in res["metrics"].items()
        },
    }


def clear_cache() -> None:
    _engineered_cache.clear()
