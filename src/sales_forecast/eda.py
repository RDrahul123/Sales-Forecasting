"""Exploratory data analysis plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

matplotlib.use("Agg")

from .config import Config  # noqa: E402


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_series(df: pd.DataFrame, store_id: int, cfg: Config) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(df["Date"], df["Sales"], lw=0.8, alpha=0.7, label="Daily sales")
    axes[0].plot(df["Date"], df["Sales"].rolling(7).mean(), lw=1.5, color="crimson", label="7-day rolling mean")
    axes[0].set_ylabel("Sales (units)")
    axes[0].set_title(f"Store {store_id}: daily sales over time")
    axes[0].legend()
    axes[1].plot(df["Date"], df["Sales"].rolling(30).mean(), lw=1.5, color="darkblue")
    axes[1].plot(df["Date"], df["Sales"].rolling(30).std(), lw=1.2, color="orange")
    axes[1].set_ylabel("30-day rolling mean / std")
    axes[1].legend(["rolling mean", "rolling std"])
    axes[1].set_title("Rolling 30-day mean and volatility")
    axes[1].set_xlabel("Date")
    path = cfg.figures_dir / f"store_{store_id}_timeseries.png"
    _save(fig, path)
    return path


def plot_seasonality(df: pd.DataFrame, store_id: int, cfg: Config) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    by_dow = df.groupby("day_of_week")["Sales"].mean()
    axes[0].bar(by_dow.index, by_dow.values, color="steelblue")
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0].set_title("Average sales by day of week")
    axes[0].set_xlabel("Day of week")
    by_month = df.groupby("month")["Sales"].mean()
    axes[1].plot(by_month.index, by_month.values, marker="o", color="seagreen")
    axes[1].set_title("Average sales by month")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xlabel("Month")
    promo = df[df["Promo"] == 1]["Sales"]
    nopromo = df[df["Promo"] == 0]["Sales"]
    axes[2].boxplot([nopromo, promo], labels=["No promo", "Promo"], patch_artist=True)
    axes[2].set_title("Sales distribution by promo flag")
    axes[2].set_ylabel("Sales")
    fig.suptitle(f"Store {store_id}: seasonality and promotion effects", y=1.02)
    path = cfg.figures_dir / f"store_{store_id}_seasonality.png"
    _save(fig, path)
    return path


def plot_acf_pacf(df: pd.DataFrame, store_id: int, cfg: Config) -> Path:
    series = df["log_sales"].dropna()
    diff1 = series.diff().dropna()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    plot_acf(diff1, lags=40, ax=axes[0])
    plot_pacf(diff1, lags=40, ax=axes[1], method="ywm")
    axes[0].set_title(f"Store {store_id}: ACF of first-differenced log sales")
    axes[1].set_title("Partial ACF")
    path = cfg.figures_dir / f"store_{store_id}_acf_pacf.png"
    _save(fig, path)
    return path


def plot_correlation(df: pd.DataFrame, store_id: int, cfg: Config) -> Path:
    cols = [
        "log_sales", "Promo", "promo2", "holiday_binary", "school_holiday",
        "day_of_week", "month", "quarter", "is_weekend",
        "log_competition_distance", "competition_open_years",
        "log_sales_lag_7", "log_sales_roll_mean_7",
    ]
    corr = df[cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Store {store_id}: correlation heatmap")
    path = cfg.figures_dir / f"store_{store_id}_corr.png"
    _save(fig, path)
    return path


def plot_decomposition(df: pd.DataFrame, store_id: int, cfg: Config) -> Path:
    series = df.set_index("Date")["Sales"]
    dec = seasonal_decompose(series, model="additive", period=7)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    dec.observed.plot(ax=axes[0], lw=0.9)
    axes[0].set_title("Observed")
    dec.trend.plot(ax=axes[1], color="crimson")
    axes[1].set_title("Trend")
    dec.seasonal.plot(ax=axes[2], color="seagreen")
    axes[2].set_title("Weekly seasonal component")
    dec.resid.plot(ax=axes[3], color="gray", lw=0.7)
    axes[3].set_title("Residual")
    fig.suptitle(f"Store {store_id}: additive STL-like decomposition (period=7)", y=1.0)
    path = cfg.figures_dir / f"store_{store_id}_decomposition.png"
    _save(fig, path)
    return path


def run_eda(df: pd.DataFrame, store_id: int, cfg: Config) -> list[Path]:
    """Render all EDA figures for a store."""
    paths = [
        plot_series(df, store_id, cfg),
        plot_seasonality(df, store_id, cfg),
        plot_acf_pacf(df, store_id, cfg),
        plot_correlation(df, store_id, cfg),
        plot_decomposition(df, store_id, cfg),
    ]
    return paths
