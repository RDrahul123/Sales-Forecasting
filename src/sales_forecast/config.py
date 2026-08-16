"""Central configuration for the sales forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """Pipeline configuration. All paths are relative to the project root."""

    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    output_dir: Path = ROOT / "outputs"
    figures_dir: Path = ROOT / "outputs" / "figures"
    forecasts_dir: Path = ROOT / "outputs" / "forecasts"
    models_dir: Path = ROOT / "outputs" / "models"

    stores: tuple[int, ...] = (1097, 682, 733)
    split_ratio: float = 0.8
    horizons: tuple[int, ...] = (30, 90, 180)
    seed: int = 42

    target: str = "Sales"
    lag_days: tuple[int, ...] = (1, 2, 7, 14, 30)
    roll_days: tuple[int, ...] = (7, 30)

    quantiles: tuple[float, ...] = (0.025, 0.10, 0.90, 0.975)

    gbm_params: dict = None
    lstm_params: dict = None

    def __post_init__(self) -> None:
        if self.gbm_params is None:
            object.__setattr__(
                self,
                "gbm_params",
                {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
                 "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.8,
                 "random_state": self.seed, "verbosity": -1},
            )
        if self.lstm_params is None:
            object.__setattr__(
                self,
                "lstm_params",
                {"lookback": 30, "hidden": 32, "layers": 2, "dropout": 0.15,
                 "epochs": 40, "batch_size": 32, "lr": 1e-3, "early_stop_patience": 6},
            )

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.output_dir, self.figures_dir,
                  self.forecasts_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)
