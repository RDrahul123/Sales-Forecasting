"""Classical SARIMA baseline built on pmdarima's auto_arima."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pmdarima as pm


class SarimaForecaster:
    """Seasonal ARIMA baseline forecasting the log1p-sales series."""

    def __init__(self, seasonal_period: int = 7, seed: int = 42):
        self.seasonal_period = seasonal_period
        self.seed = seed
        self.model: pm.ARIMA | None = None

    def fit(self, y: pd.Series | np.ndarray) -> "SarimaForecaster":
        y = np.asarray(y, dtype=float)
        self.model = pm.auto_arima(
            y,
            seasonal=True,
            m=self.seasonal_period,
            stepwise=True,
            max_p=3,
            max_q=3,
            max_order=5,
            d=1,
            D=1,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            n_jobs=1,
            random_state=self.seed,
        )
        return self

    def predict(self, steps: int, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Forecast `steps` ahead in log scale, returning (mean, conf_int)."""
        fc, conf = self.model.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
        return np.asarray(fc), np.asarray(conf)

    def update(self, y: pd.Series | np.ndarray) -> "SarimaForecaster":
        """Extend the fitted model with new observations (no refit)."""
        self.model.update(np.asarray(y, dtype=float))
        return self

    def forecast_quantiles(self, steps: int) -> np.ndarray:
        """Return the quantile matrix aligned to cfg.quantiles = (0.025,0.10,0.90,0.975)."""
        mean_95, ci_95 = self.predict(steps, alpha=0.05)
        mean_80, ci_80 = self.predict(steps, alpha=0.20)
        return np.column_stack([ci_95[:, 0], ci_80[:, 0], ci_80[:, 1], ci_95[:, 1]])
