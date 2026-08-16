"""Data ingestion and cleaning for the Rossmann store sales dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

TRAIN_FILE = "train.csv"
STORE_FILE = "store.csv"


def load_train(path) -> pd.DataFrame:
    """Load the raw training frame."""
    return pd.read_csv(path, parse_dates=["Date"], low_memory=False)


def load_store(path) -> pd.DataFrame:
    """Load the store metadata frame."""
    return pd.read_csv(path)


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes and standardise the state-holiday codes."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["StateHoliday"] = (
        df["StateHoliday"].astype(str).str.strip().replace({"nan": "0", "None": "0"})
    )
    df["StateHoliday"] = df["StateHoliday"].where(df["StateHoliday"].isin(["0", "a", "b", "c"]), "0")
    df["Store"] = df["Store"].astype(int)
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0).astype(int)
    if "Customers" not in df.columns:
        df["Customers"] = 0
    df["Customers"] = pd.to_numeric(df["Customers"], errors="coerce").fillna(0).astype(int)
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce").fillna(0).astype(int)
    df["Promo"] = pd.to_numeric(df["Promo"], errors="coerce").fillna(0).astype(int)
    df["SchoolHoliday"] = pd.to_numeric(df["SchoolHoliday"], errors="coerce").fillna(0).astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
    return df


def fill_missing_days(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a continuous daily grid so the time index never has gaps.

    Missing rows are treated as closed days (Open=0, Sales=0).
    """
    df = df.set_index("Date").sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)
    for col in ["Sales", "Customers", "Open", "Promo", "SchoolHoliday"]:
        df[col] = df[col].fillna(0).astype(int)
    df["StateHoliday"] = df["StateHoliday"].fillna("0")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].ffill()
    return df.rename_axis("Date").reset_index()


def prepare_store_data(train: pd.DataFrame, store: pd.DataFrame, store_id: int) -> pd.DataFrame:
    """Return a clean daily frame for one store merged with its metadata."""
    df = train[train["Store"] == store_id].copy()
    df = clean_train(df)
    df = fill_missing_days(df)
    meta = store[store["Store"] == store_id]
    if meta.empty:
        raise ValueError(f"Store {store_id} not found in store metadata")
    df = df.merge(meta, on="Store", how="left")
    df["log_sales"] = np.log1p(df["Sales"])
    return df.reset_index(drop=True)


def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean the full train/store frames once."""
    train = load_train(cfg.data_dir / TRAIN_FILE)
    store = load_store(cfg.data_dir / STORE_FILE)
    return clean_train(train), store
