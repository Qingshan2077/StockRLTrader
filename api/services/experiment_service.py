from __future__ import annotations

from api.services.serialization import json_safe
from layers.experiments import ExperimentManager


class ExperimentService:
    def __init__(self):
        self.manager = ExperimentManager()

    def list(self, status: str | None = None, tag: str | None = None) -> list[dict]:
        return [json_safe(exp) for exp in self.manager.list_experiments(status=status, tag=tag)]

    def create(self, name: str, config: dict, tags: list[str]) -> str:
        return self.manager.create_experiment(name=name, config=config, tags=tags)

    def compare(self, exp_ids: list[str], metrics: list[str] | None = None) -> list[dict]:
        return [json_safe(exp) for exp in self.manager.compare(exp_ids, metrics)]
