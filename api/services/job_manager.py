from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4


@dataclass
class Job:
    job_id: str
    name: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    result: Any = None
    future: Future | None = None
    logs: list[str] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
        }

    def log(self, message: str, progress: float | None = None) -> None:
        self.message = message
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        self.logs.append(f"{datetime.now().isoformat()} {message}")


class JobManager:
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, Job] = {}

    def submit(self, name: str, fn: Callable[[Job], Any]) -> Job:
        job = Job(job_id=uuid4().hex[:12], name=name)
        self.jobs[job.job_id] = job

        def runner() -> Any:
            job.status = "running"
            job.started_at = datetime.now().isoformat()
            job.log("started", 0.0)
            try:
                job.result = fn(job)
                job.status = "completed"
                job.log("completed", 1.0)
                return job.result
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                job.log(f"failed: {exc}")
                raise
            finally:
                job.finished_at = datetime.now().isoformat()

        job.future = self.executor.submit(runner)
        return job

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.future and job.future.cancel():
            job.status = "cancelled"
            job.finished_at = datetime.now().isoformat()
            job.log("cancelled")
        return job


job_manager = JobManager()
