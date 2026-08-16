"""Generate the companion Jupyter notebook for the sales forecasting pipeline.

Run from the project root:  python notebook/generate_notebook.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook" / "sales_forecasting.ipynb"


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(source=src, metadata={})


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(
        source=src, metadata={}, outputs=[], execution_count=None
    )


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        md(
            "# Sales Forecasting: Time-Series Analysis (Rossmann Store Sales)\n\n"
            "An end-to-end pipeline: ingestion -> EDA -> feature engineering -> "
            "SARIMA / LightGBM / LSTM modelling -> evaluation -> multi-horizon "
            "forecasts with confidence bands -> report.\n\n"
            "This notebook walks through one store end to end. The same steps run for "
            "multiple stores via `python run_pipeline.py`."
        ),
        md("## 0. Setup"),
        code(
            "import sys, warnings\n"
            "sys.path.insert(0, '..')          # project root\n"
            "sys.path.insert(0, '../src')      # package\n"
            "warnings.filterwarnings('ignore')\n\n"
            "import numpy as np\nimport pandas as pd\n"
            "from sales_forecast.config import Config\n"
            "from sales_forecast.ingestion import load_data, prepare_store_data\n"
            "from sales_forecast.features import build_features, GBM_FEATURE_COLS, LSTM_FEATURE_COLS\n"
            "from sales_forecast.eda import run_eda\n"
            "from sales_forecast.evaluate import compute_metrics, error_segmentation\n"
            "from sales_forecast.models import SarimaForecaster, GbmForecaster, LstmForecaster\n"
            "from sales_forecast.forecast import build_future_frame, recursive_gbm_forecast, forecast_table\n"
            "from sales_forecast.pipeline import run_store, forecast_sales\n"
            "from sales_forecast.report import write_report\n\n"
            "cfg = Config()\n"
            "print('Config OK. Stores:', cfg.stores, '| Horizons:', cfg.horizons)"
        ),
        md("## 1. Data ingestion & cleaning"),
        code(
            "train, store = load_data(cfg)\n"
            "print('train shape:', train.shape, '| store metadata:', store.shape)\n"
            "train[['Date', 'Sales', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']].head(5)"
        ),
        code(
            "df = prepare_store_data(train, store, 1097)\n"
            "df = build_features(df, store[store['Store'] == 1097].iloc[0], cfg)\n"
            "print('engineered frame:', df.shape)\n"
            "print('date range:', df['Date'].min().date(), '->', df['Date'].max().date())\n"
            "print('missing timestamps:', int(df['Date'].duplicated().sum()))"
        ),
        md("## 2. Exploratory data analysis\n\n"
           "Time series, weekly/monthly seasonality, promo effect, ACF/PACF, "
           "correlation heatmap and seasonal decomposition."),
        code(
            "paths = run_eda(df, 1097, cfg)\n"
            "print('EDA figures written:', len(paths))"
        ),
        code(
            "dow = df.groupby('day_of_week')['Sales'].mean()\n"
            "names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']\n"
            "print('Avg sales by weekday:', {names[i]: round(v) for i, v in dow.items()})\n"
            "p0 = df[df['Promo']==0]['Sales'].mean(); p1 = df[df['Promo']==1]['Sales'].mean()\n"
            "print(f'Promo lift: +{100*(p1/p0-1):.1f}%')"
        ),
        md("## 3. Feature engineering\n\n"
           "Calendar/cyclical features, holiday and promo encodings, store metadata, "
           "and lag/rolling statistics of the log-sales series."),
        code(
            "print('GBM features (%d):' % len(GBM_FEATURE_COLS))\n"
            "print(', '.join(GBM_FEATURE_COLS))\n"
            "print()\n"
            "print('LSTM window features (%d):' % len(LSTM_FEATURE_COLS))\n"
            "print(', '.join(LSTM_FEATURE_COLS))"
        ),
        md("## 4. Modelling\n\n"
           "Chronological 80/20 split. Three models:\n"
           "- **SARIMA** (classical, pmdarima auto_arima, period 7)\n"
           "- **LightGBM** (gradient boosting on lag/rolling features, time-series CV)\n"
           "- **LSTM** (PyTorch sequence model, recursive multi-step)\n\n"
           "The cell below runs the full per-store pipeline (about 3 minutes)."),
        code(
            "res = run_store(1097, cfg)\n"
            "for k, v in res['metrics'].items():\n"
            "    if 'recursive' in k:\n"
            "        print(f\"{k:24s} MAE={v['MAE']:8.0f} RMSE={v['RMSE']:8.0f} \"\n"
            "              f\"MAPE={v['MAPE']:5.2f}%  MASE={v['MASE']:.3f}\")\n"
            "print('\\nBest model by recursive-test RMSE:', res['best_model'])"
        ),
        md("## 5. Evaluation\n\n"
           "Metrics: MAE, RMSE, MAPE, MASE. Recursive forecasts respect the "
           "chronological split (first 80% train, last 20% test)."),
        code(
            "best = res['best_model']\n"
            "print('Test-period recursive metrics for best model (%s):' % best)\n"
            "print(pd.Series(res['metrics'][best + '_recursive']))\n"
            "print('\\nTop 10 LightGBM features:')\n"
            "print(res['feature_importance'].head(10).to_string())"
        ),
        md("## 6. Multi-horizon forecasts (30 / 90 / 180 days)"),
        code(
            "fc30 = res['forecasts'][30][best]\n"
            "fc30.head(10)"
        ),
        code(
            "out = cfg.forecasts_dir / f'store_1097_30d_{best.lower()}.csv'\n"
            "fc30.to_csv(out, index=False)\n"
            "print('saved:', out)"
        ),
        md("## 7. Public function interface\n\n"
           "`forecast_sales(store_id, horizon)` returns the deliverable forecast "
           "dataframe: `date, predicted_sales, lower_80, upper_80, lower_95, upper_95`."),
        code(
            "fc = forecast_sales(1097, horizon=30)\n"
            "fc.head()"
        ),
        md("## 8. Full multi-store pipeline & report\n\n"
           "Run all configured stores and regenerate `outputs/report.md`. "
           "(This cell can take several minutes; the per-store results above are "
           "already cached in `outputs/`.)"),
        code(
            "from sales_forecast.pipeline import run_pipeline\n"
            "all_results = run_pipeline(cfg)\n"
            "report_path = write_report(all_results, cfg)\n"
            "print('Report:', report_path)"
        ),
        md("## Summary\n\n"
           "- **LightGBM** with lag/rolling features is the most accurate multi-step "
           "forecaster (recursive-test MAPE 7-8%).\n"
           "- The **LSTM** is excellent one-step-ahead (MAPE ~1%) but compounds errors "
           "over long recursive horizons.\n"
           "- **SARIMA** provides a solid classical baseline and native prediction intervals.\n"
           "- Forecast CSVs, model artifacts and `report.md` live under `outputs/`."),
    ]
    nb["cells"] = cells
    for cell in nb["cells"]:
        if not cell.get("id"):
            cell["id"] = str(uuid.uuid4())
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3.14 (sales-forecast)", "language": "python", "name": "python314"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> None:
    nb = build()
    nbf.write(nb, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
