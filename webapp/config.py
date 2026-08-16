"""Web application paths and global settings."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sales_forecast.config import Config as PipelineConfig  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
APP_DATA_DIR = PROJECT_ROOT / "webapp_data"
UPLOADS_DIR = APP_DATA_DIR / "uploads"

BASE_CFG = PipelineConfig()

UPLOAD_TRAIN_COLS = [
    "Store", "Date", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday",
]

STORE_META_DEFAULTS = {
    "StoreType": "a",
    "Assortment": "a",
    "CompetitionDistance": 10000.0,
    "CompetitionOpenSinceYear": None,
    "CompetitionOpenSinceMonth": None,
    "Promo2": 0,
    "Promo2SinceWeek": None,
    "Promo2SinceYear": None,
    "PromoInterval": "",
}
