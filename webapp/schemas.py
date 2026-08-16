"""Request/response models for the REST API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    dataset: str = "demo"
    store_id: int
    horizon: int = Field(default=30, ge=7, le=365)
    model: str = "best"
    promo_mode: str = "repeat"


class RetrainRequest(BaseModel):
    dataset: str = "demo"
    store_id: int


class JobResponse(BaseModel):
    job_id: str
    description: str
    status: str
    progress: int
    message: str
    error: Optional[str] = None
    result: Optional[dict] = None
