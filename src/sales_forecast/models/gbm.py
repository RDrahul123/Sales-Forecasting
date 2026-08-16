"""Gradient-boosting forecaster (LightGBM) on engineered lag/rolling features."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


class GbmForecaster:
    """LightGBM regressor trained on tabular features with time-aware CV."""

    def __init__(self, base_params: dict | None = None, seed: int = 42):
        self.seed = seed
        self.base_params = base_params or {
            "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
            "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.8,
            "random_state": seed, "verbosity": -1,
        }
        self.model: lgb.LGBMRegressor | None = None
        self.feature_cols: list[str] = []
        self.best_params: dict = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, tune: bool = True) -> "GbmForecaster":
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        self.feature_cols = list(X.columns)
        if tune:
            param_grid = {
                "num_leaves": [15, 31],
                "learning_rate": [0.03, 0.08],
                "max_depth": [4, 6],
            }
            base = lgb.LGBMRegressor(
                n_estimators=self.base_params["n_estimators"],
                subsample=self.base_params["subsample"],
                colsample_bytree=self.base_params["colsample_bytree"],
                random_state=self.seed,
                verbosity=-1,
            )
            tscv = TimeSeriesSplit(n_splits=4)
            gs = GridSearchCV(
                base, param_grid, cv=tscv,
                scoring="neg_mean_absolute_error", n_jobs=1,
            )
            gs.fit(X, y)
            self.model = gs.best_estimator_
            self.best_params = gs.best_params_
        else:
            self.model = lgb.LGBMRegressor(**self.base_params)
            self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)
