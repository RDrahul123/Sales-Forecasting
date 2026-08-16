"""Command-line entry point for the sales forecasting pipeline.

Usage:
    python run_pipeline.py                     # all configured stores
    python run_pipeline.py --stores 1 4 85     # specific stores
    python run_pipeline.py --horizons 30 90    # forecast horizons
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sales_forecast.config import Config  # noqa: E402
from sales_forecast.pipeline import run_pipeline  # noqa: E402
from sales_forecast.report import write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Sales forecasting pipeline (Rossmann).")
    parser.add_argument("--stores", nargs="+", type=int, default=None,
                        help="Store IDs to model (default: stores in Config).")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Forecast horizons in days (default: 30 90 180).")
    parser.add_argument("--split", type=float, default=None,
                        help="Chronological train fraction (default: 0.8).")
    args = parser.parse_args()

    cfg = Config()
    if args.stores:
        cfg = dataclasses.replace(cfg, stores=tuple(args.stores))
    if args.horizons:
        cfg = dataclasses.replace(cfg, horizons=tuple(args.horizons))
    if args.split:
        cfg = dataclasses.replace(cfg, split_ratio=args.split)

    results = run_pipeline(cfg)
    report_path = write_report(results, cfg)
    print(f"[cli] report written to {report_path}")
    for store_id, res in results.items():
        print(
            f"[cli] store {store_id}: best model = {res['best_model']} | "
            f"RMSE(recursive) = {res['metrics'][res['best_model'] + '_recursive']['RMSE']:.0f} "
            f"| MAPE = {res['metrics'][res['best_model'] + '_recursive']['MAPE']:.2f}%"
        )


if __name__ == "__main__":
    main()
