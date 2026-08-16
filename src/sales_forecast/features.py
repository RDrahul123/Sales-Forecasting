"""Feature engineering: calendar, cyclical, holiday, promo, lag and rolling features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

MONTH_ABBR = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic calendar features (safe to compute for any future date)."""
    d = df.copy()
    d["day_of_week"] = d["Date"].dt.dayofweek
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    d["day_of_month"] = d["Date"].dt.day
    d["month"] = d["Date"].dt.month
    d["quarter"] = d["Date"].dt.quarter
    d["week_of_year"] = d["Date"].dt.isocalendar().week.astype(int)
    d["day_of_year"] = d["Date"].dt.dayofyear
    d["year"] = d["Date"].dt.year
    d["sin_dow"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["cos_dow"] = np.cos(2 * np.pi * d["day_of_week"] / 7)
    d["sin_doy"] = np.sin(2 * np.pi * d["day_of_year"] / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * d["day_of_year"] / 365.25)
    d["sin_month"] = np.sin(2 * np.pi * d["month"] / 12)
    d["cos_month"] = np.cos(2 * np.pi * d["month"] / 12)
    return d


def encode_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Binary and one-hot encodings of the state/school holiday flags."""
    d = df.copy()
    d["holiday_binary"] = (d["StateHoliday"] != "0").astype(int)
    d["holiday_a"] = (d["StateHoliday"] == "a").astype(int)
    d["holiday_b"] = (d["StateHoliday"] == "b").astype(int)
    d["holiday_c"] = (d["StateHoliday"] == "c").astype(int)
    d["school_holiday"] = pd.to_numeric(d["SchoolHoliday"], errors="coerce").fillna(0).astype(int)
    return d


def promo2_active_weeks(store_row: pd.Series) -> set[tuple[int, int]]:
    """Set of (year, week) in which the recurring Promo2 runs for a store.

    Promo2 starts in Promo2SinceWeek of Promo2SinceYear and then runs for one
    week in each month listed in PromoInterval. Approximated by the ISO week
    containing the first day of each listed month.
    """
    if store_row["Promo2"] != 1 or pd.isna(store_row["Promo2SinceWeek"]) or pd.isna(store_row["Promo2SinceYear"]):
        return set()
    start = (int(store_row["Promo2SinceYear"]), int(store_row["Promo2SinceWeek"]))
    months = [MONTH_ABBR[m.strip()] for m in str(store_row["PromoInterval"]).split(",") if m.strip()]
    weeks = set()
    for year in range(start[0], 2018):
        for month in months:
            iso = pd.Timestamp(year=year, month=month, day=1).isocalendar()
            weeks.add((int(iso.year), int(iso.week)))
    return {(y, w) for (y, w) in weeks if (y, w) >= start}


def add_store_features(df: pd.DataFrame, store_row: pd.Series) -> pd.DataFrame:
    """Static store metadata plus time-varying Promo2 flag."""
    d = df.copy()
    comp_dist = store_row["CompetitionDistance"]
    d["log_competition_distance"] = np.log1p(
        comp_dist if not pd.isna(comp_dist) else 0.0
    )
    comp_open = (
        pd.Timestamp(
            int(store_row["CompetitionOpenSinceYear"]),
            int(store_row["CompetitionOpenSinceMonth"]),
            15,
        )
        if not pd.isna(store_row["CompetitionOpenSinceYear"]) and not pd.isna(store_row["CompetitionOpenSinceMonth"])
        else pd.NaT
    )
    d["competition_open_years"] = ((d["Date"] - comp_open).dt.days / 365.25).clip(lower=0).fillna(0)
    weeks = promo2_active_weeks(store_row)
    iso = pd.DataFrame(
        {"year": d["Date"].dt.isocalendar().year.astype(int),
         "week": d["Date"].dt.isocalendar().week.astype(int)}
    )
    d["promo2"] = iso.apply(tuple, axis=1).isin(weeks).astype(int)
    d["store_type"] = store_row["StoreType"]
    d["assortment"] = store_row["Assortment"]
    return d


def add_lag_rolling(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Lag and rolling-window statistics of the log-sales series."""
    d = df.copy()
    for lag in cfg.lag_days:
        d[f"log_sales_lag_{lag}"] = d["log_sales"].shift(lag)
    for w in cfg.roll_days:
        d[f"log_sales_roll_mean_{w}"] = d["log_sales"].rolling(w).mean()
        d[f"log_sales_roll_std_{w}"] = d["log_sales"].rolling(w).std()
    return d


GBM_FEATURE_COLS = [
    "day_of_week", "is_weekend", "day_of_month", "month", "quarter",
    "week_of_year", "day_of_year", "sin_dow", "cos_dow", "sin_doy", "cos_doy",
    "sin_month", "cos_month", "holiday_binary", "holiday_a", "holiday_b",
    "holiday_c", "school_holiday", "Promo", "promo2",
    "log_competition_distance", "competition_open_years",
    "log_sales_lag_1", "log_sales_lag_2", "log_sales_lag_7",
    "log_sales_lag_14", "log_sales_lag_30",
    "log_sales_roll_mean_7", "log_sales_roll_mean_30",
    "log_sales_roll_std_7", "log_sales_roll_std_30",
]

LSTM_FEATURE_COLS = [
    "Promo", "promo2", "holiday_binary", "holiday_a", "holiday_b", "holiday_c",
    "school_holiday", "is_weekend", "sin_dow", "cos_dow", "sin_doy", "cos_doy",
    "log_sales",
]


def build_features(df: pd.DataFrame, store_row: pd.Series, cfg: Config) -> pd.DataFrame:
    """Apply the full feature-engineering stack to one store frame."""
    d = add_date_features(df)
    d = encode_holidays(d)
    d = add_store_features(d, store_row)
    d = add_lag_rolling(d, cfg)
    return d


def build_future_features(
    start: pd.Timestamp, horizon: int, store_row: pd.Series,
) -> pd.DataFrame:
    """Exogenous/calendar features for future dates in the forecast window.

    Lag and rolling features are filled by the recursive forecaster at runtime.
    Promo defaults to the store's most recent weekly pattern (passed separately
    by the caller via `promo_sequence`).
    """
    idx = pd.date_range(start, periods=horizon, freq="D")
    df = pd.DataFrame({"Date": idx})
    df["Store"] = int(store_row["Store"])
    df["Open"] = 1
    from .holidays import state_holiday_for

    df["StateHoliday"] = [state_holiday_for(ts.date()) for ts in idx]
    df["SchoolHoliday"] = 0
    df["Sales"] = np.nan
    df["log_sales"] = np.nan
    df = add_date_features(df)
    df = encode_holidays(df)
    df = add_store_features(df, store_row)
    return df
