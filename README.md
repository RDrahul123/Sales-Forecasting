# Sales Forecasting — Time-Series Analysis (Rossmann Store Sales)

End-to-end, reproducible forecasting of **daily store sales** using three model families:

| Family | Model | Notes |
|---|---|---|
| Classical | SARIMA (`pmdarima.auto_arima`, period 7) | Baseline with native prediction intervals |
| Machine Learning | LightGBM on lag/rolling/calendar features | Time-series CV + tuning |
| Deep Learning | PyTorch LSTM (sequence windows) | Recursive multi-step forecasting |

Pipeline stages: **data ingestion → EDA → feature engineering → modelling →
evaluation → multi-horizon forecasts (30/90/180 days) with 80%/95% confidence
bands → Markdown report**.

## Data

Public **Rossmann Store Sales** dataset (Kaggle competition), 1,015k rows, 1,115
stores, daily from 2013-01-01 to 2015-07-31. Downloaded into `data/` from a
GitHub mirror (`train.csv`, `store.csv`). Includes `Store, Date, Sales,
Customers, Open, Promo, StateHoliday, SchoolHoliday` plus store metadata
(assortment, competition distance, Promo2).

Three always-open stores are used for the demonstration: **1097** (low volume),
**682** (mid), **733** (high). Any store can be added via `--stores`.

## Project structure

```
sales_forecasting/
├── data/                      # raw train.csv + store.csv
├── src/sales_forecast/
│   ├── config.py              # central configuration
│   ├── holidays.py            # German holiday calendar for the forecast window
│   ├── ingestion.py           # load, clean, reindex (no missing timestamps)
│   ├── features.py            # cyclical, holiday, promo, lag, rolling features
│   ├── eda.py                 # figures: series, seasonality, ACF/PACF, corr, decomposition
│   ├── evaluate.py            # MAE, RMSE, MAPE, MASE + error segmentation
│   ├── forecast.py            # recursive multi-step forecast + confidence bands
│   ├── report.py              # report.md generation
│   ├── models/                # SARIMA, LightGBM, LSTM
│   └── pipeline.py            # run_store / run_pipeline / forecast_sales
├── webapp/                    # FastAPI + single-page frontend
│   ├── main.py, api.py, service.py, jobs.py, schemas.py, config.py
│   └── static/                # index.html, app.js, style.css, vendor/plotly.min.js
├── notebook/
│   ├── sales_forecasting.ipynb   # companion walkthrough
│   └── generate_notebook.py      # regenerates the notebook
├── outputs/
│   ├── figures/               # all EDA/model/forecast plots
│   ├── forecasts/             # CSV per store × horizon × model
│   ├── models/                # trained artifacts (joblib / .pt / summary.json)
│   └── report.md              # the deliverable report
├── run_pipeline.py            # CLI entry point
├── run_web.py                 # web app entry point
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

## Setup

```powershell
pip install -r requirements.txt
```

Python 3.10+ required; verified on 3.14.

## Usage

```powershell
# Full multi-store pipeline + report (all stores in Config)
python run_pipeline.py

# Specific stores / horizons / split
python run_pipeline.py --stores 1 4 85 --horizons 30 90 --split 0.8
```

Public function interface:

```python
from sales_forecast.pipeline import forecast_sales
fc = forecast_sales(1097, horizon=30)      # returns date + predicted_sales + 80/95% bands
```

Or open `notebook/sales_forecasting.ipynb` for the interactive walkthrough.

## Web application

A FastAPI backend + single-page frontend (no build step, offline Plotly) exposes
the whole workflow in a browser: dashboard with EDA figures and model metrics,
on-demand forecasting from the saved models, custom CSV upload and background
retraining with progress.

```powershell
# local (serves http://127.0.0.1:8000)
python run_web.py

# or explicitly
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

Tabs:
- **Dashboard** — pick a dataset + store to see model comparison, insights,
  feature importance, EDA figures and precomputed forecast charts.
- **Forecast** — select store, horizon (30/90/180 days), model and promo
  scenario; the forecast (with 80/95% bands) is generated on demand from the
  saved artifacts in seconds, and the CSV is downloadable.
- **Upload & Train** — upload your own `train.csv` (columns `Store, Date,
  Sales, Open, Promo, StateHoliday, SchoolHoliday`) plus an optional Rossmann
  `store.csv`; train all three models for one store (~2–3 min) with live
  progress, then forecast it.

REST API (see interactive docs at `/docs`):

```
GET  /api/health
GET  /api/datasets
GET  /api/stores/{id}/overview?dataset=demo
GET  /api/stores/{id}/history?dataset=demo&days=120
POST /api/forecast      {dataset, store_id, horizon, model, promo_mode}
POST /api/retrain       {dataset, store_id}     -> {job_id}
GET  /api/jobs/{job_id}                          -> progress
POST /api/upload        multipart train_file (+ store_file) -> {upload_id}
```

`dataset` is `demo` (the bundled Rossmann data) or `upload:<upload_id>`
returned by `/api/upload`. Uploaded datasets live under `webapp_data/uploads/`
and are listed automatically.

## Docker

```powershell
docker build -t sales-forecast .
docker run --rm -p 8000:8000 sales-forecast
# open http://localhost:8000
```

The image uses a CPU-only PyTorch build to keep it lean. Pre-trained artifacts
in `outputs/` are not baked into the image, so the demo stores start untrained —
use the **Upload & Train** tab to train any store (demo data is included).
Persist uploads/retrains across restarts by mounting a volume over
`/app/webapp_data`.

## Evaluation protocol

- Strict **chronological split**: first 80% train, last 20% test.
- Metrics: **MAE, RMSE, MAPE, MASE** (MASE vs seasonal-naive, lag 7).
- One-step-ahead and **recursive multi-step** evaluation on the test window.
- Error segmentation by promotion status, weekday and holiday flag.

## Results (recursive test period)

| Store | Best model | MAE | RMSE | MAPE | MASE |
|---|---|---|---|---|---|
| 1097 | LightGBM | 786 | 1045 | 7.09% | 0.62 |
| 682 | LightGBM | 940 | 1372 | 7.91% | 0.33 |
| 733 | LightGBM | 1337 | 1755 | 8.24% | 0.73 |

The LSTM is very strong one-step-ahead (MAPE ≈ 1%) but compounds error over long
recursive horizons; LightGBM with explicit lag/rolling features is the most robust
multi-step model. See `outputs/report.md` for the full comparison, EDA insights,
promotion impact scenarios and business recommendations.

## Reproducibility

- `requirements.txt` pins exact library versions; `pyproject.toml` allows
  `pip install -e .`.
- The pipeline handles missing timestamps (reindexes to a continuous daily grid)
  and is parameterized over stores — no code changes needed for new stores.
- Forecast outputs follow the standard schema:
  `date, predicted_sales, lower_80, upper_80, lower_95, upper_95`.
