"""PyTorch LSTM forecaster with recursive multi-step prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, layers, batch_first=True, dropout=dropout
        )
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def build_windows(values: np.ndarray, lookback: int) -> np.ndarray:
    """Rolling windows of shape (N - lookback + 1, lookback, F)."""
    n = len(values)
    idx = np.arange(lookback)[None, :] + np.arange(n - lookback + 1)[:, None]
    return values[idx]


class LstmForecaster:
    """Sequence model: windows of (log_sales + exogenous) features -> next day log_sales."""

    def __init__(self, params: dict | None = None, seed: int = 42):
        params = params or {}
        self.lookback = params.get("lookback", 30)
        self.hidden = params.get("hidden", 32)
        self.layers = params.get("layers", 2)
        self.dropout = params.get("dropout", 0.15)
        self.epochs = params.get("epochs", 40)
        self.batch_size = params.get("batch_size", 32)
        self.lr = params.get("lr", 1e-3)
        self.patience = params.get("early_stop_patience", 6)
        self.seed = seed
        self.device = torch.device("cpu")
        self.model: _LSTMNet | None = None
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.target_col: str = "log_sales"
        self._target_idx: int = -1

    def _seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _scale_log(self, scaled_values: np.ndarray) -> np.ndarray:
        mean = self.scaler.mean_[self._target_idx]
        std = self.scaler.scale_[self._target_idx]
        return scaled_values * std + mean

    def fit(self, df: pd.DataFrame, feature_cols: list[str], target_col: str,
            train_end: int, val_frac: float = 0.15) -> "LstmForecaster":
        self._seed()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self._target_idx = feature_cols.index(target_col)
        train_df = df.iloc[:train_end]
        self.scaler.fit(train_df[feature_cols].values)
        scaled_all = self.scaler.transform(df[feature_cols].values)
        windows = build_windows(scaled_all, self.lookback)
        ends = np.arange(self.lookback - 1, len(scaled_all))
        targets_scaled = scaled_all[self.lookback - 1:, self._target_idx]

        val_start = int(train_end * (1 - val_frac))
        train_mask = ends < val_start
        val_mask = (ends >= val_start) & (ends < train_end)

        X_tr = torch.tensor(windows[train_mask], dtype=torch.float32)
        y_tr = torch.tensor(targets_scaled[train_mask], dtype=torch.float32)
        X_va = torch.tensor(windows[val_mask], dtype=torch.float32)
        y_va = torch.tensor(targets_scaled[val_mask], dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True
        )
        self.model = _LSTMNet(len(feature_cols), self.hidden, self.layers, self.dropout).to(
            self.device
        )
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        stale = 0
        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                optim.zero_grad()
                pred = self.model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_va).squeeze(-1)
                val_loss = loss_fn(val_pred, y_va).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, df: pd.DataFrame, predict_mask: np.ndarray) -> np.ndarray:
        """One-step-ahead log-scale predictions for rows where predict_mask is True."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        scaled_all = self.scaler.transform(df[self.feature_cols].values)
        windows = build_windows(scaled_all, self.lookback)
        ends = np.arange(self.lookback - 1, len(scaled_all))
        keep = predict_mask[self.lookback - 1:]
        rows = windows[keep]
        out = np.full(len(df), np.nan)
        if len(rows):
            with torch.no_grad():
                preds = (
                    self.model(torch.tensor(rows, dtype=torch.float32))
                    .squeeze(-1)
                    .numpy()
                )
            out[ends[keep]] = self._scale_log(preds)
        return out

    def forecast(self, hist_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
        """Recursive multi-step forecast in log scale for future_df rows."""
        if self.model is None:
            raise RuntimeError("Call fit() before forecast().")
        self._seed()
        hist = hist_df[self.feature_cols].values[-self.lookback:].copy()
        exog_cols = [c for c in self.feature_cols if c != self.target_col]
        future = future_df.reset_index(drop=True)
        preds_log = np.empty(len(future))
        with torch.no_grad():
            for i in range(len(future)):
                row_unscaled = np.empty(len(self.feature_cols))
                exog_vals = future.iloc[i][exog_cols].to_numpy(dtype=float)
                for j, c in enumerate(self.feature_cols):
                    if c == self.target_col:
                        row_unscaled[j] = hist[-1, j]
                    else:
                        row_unscaled[j] = exog_vals[exog_cols.index(c)]
                scaled_row = self.scaler.transform(row_unscaled[None, :])
                window_scaled = self.scaler.transform(hist)
                x = torch.tensor(
                    window_scaled[-self.lookback:][None, :, :], dtype=torch.float32
                )
                pred_scaled = self.model(x).item()
                pred_log = float(self._scale_log(np.array([pred_scaled]))[0])
                preds_log[i] = pred_log
                new_row = row_unscaled.copy()
                new_row[self._target_idx] = pred_log
                hist = np.vstack([hist, new_row])[-self.lookback:]
        return preds_log
