"""FastAPI application entry point for the sales forecasting web app."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import service
from .api import router
from .config import BASE_CFG, STATIC_DIR, UPLOADS_DIR

app = FastAPI(
    title="Sales Forecasting",
    description="Web interface and API for the SARIMA / LightGBM / LSTM sales forecasting pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

BASE_CFG.ensure_dirs()
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=BASE_CFG.output_dir), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.on_event("startup")
def _warmup() -> None:
    service.clear_cache()
