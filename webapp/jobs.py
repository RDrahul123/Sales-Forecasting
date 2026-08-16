"""In-process background job manager with progress tracking for retraining runs."""

from __future__ import annotations

import threading
import uuid
from typing import Callable


class Job:
    def __init__(self, job_id: str, description: str):
        self.job_id = job_id
        self.description = description
        self.status = "queued"
        self.progress = 0
        self.message = "Queued"
        self.error: str | None = None
        self.result: dict | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "description": self.description,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "result": self.result,
        }


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def submit(description: str, fn: Callable) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, description)
    with _lock:
        _jobs[job_id] = job

    def runner() -> None:
        job.status = "running"
        job.progress = 1
        job.message = "Starting"

        def progress(percent: int, message: str) -> None:
            job.progress = percent
            job.message = message

        try:
            result = fn(progress)
            job.result = result
            job.status = "succeeded"
            job.progress = 100
            job.message = "Completed"
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.message = f"{type(exc).__name__}: {exc}"
            job.error = str(exc)

    threading.Thread(target=runner, daemon=True).start()
    return job


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[Job]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.job_id)
