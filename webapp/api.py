"""REST API endpoints for the sales forecasting web application."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile

from . import jobs, service
from .schemas import ForecastRequest, JobResponse, RetrainRequest

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/datasets")
def datasets() -> dict:
    return {"datasets": service.list_datasets()}


@router.get("/stores/{store_id}/overview")
def store_overview(store_id: int, dataset: str = "demo") -> dict:
    try:
        return service.overview(dataset, store_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/stores/{store_id}/history")
def store_history(store_id: int, dataset: str = "demo", days: int = 120) -> dict:
    try:
        return {
            "dataset": dataset,
            "store_id": store_id,
            "history": service.history(dataset, store_id, days),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/forecast")
def forecast(req: ForecastRequest) -> dict:
    try:
        return service.run_forecast(
            req.dataset, req.store_id, req.horizon,
            model=req.model, promo_mode=req.promo_mode,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/retrain")
def retrain(req: RetrainRequest) -> JobResponse:
    try:
        service.load_summary(req.dataset, req.store_id)
        trained = True
    except FileNotFoundError:
        trained = False
    job = jobs.submit(
        f"Retrain store {req.store_id} ({service.dataset_label(req.dataset)})",
        lambda progress: service.retrain(req.dataset, req.store_id, progress=progress),
    )
    resp = JobResponse(**job.to_dict())
    resp.result = {"trained_before": trained}
    return resp


@router.post("/upload")
async def upload(train_file: UploadFile, store_file: UploadFile | None = None) -> dict:
    train_content = await train_file.read()
    if len(train_content) == 0:
        raise HTTPException(status_code=400, detail="Train file is empty")
    store_content = await store_file.read() if store_file is not None else None
    try:
        info = service.create_upload(
            train_file.filename or "train.csv",
            train_content,
            store_file.filename if store_file is not None else None,
            store_content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    return info


@router.get("/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: str) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
    return JobResponse(**job.to_dict())


@router.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": [j.to_dict() for j in jobs.list_jobs()]}
