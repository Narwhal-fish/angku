"""Tracker that keeps everything in memory. For tests and local dry runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from mlops.domain.experiment import ExperimentSpec
from mlops.ports.tracking import RunStatus


@dataclass
class RecordedRun:
    run_id: str
    experiment_id: str
    spec: ExperimentSpec
    metrics: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    status: RunStatus | None = None


class InMemoryTracker:
    def __init__(self) -> None:
        self.runs: list[RecordedRun] = []
        self._active: RecordedRun | None = None

    @property
    def current(self) -> RecordedRun:
        if self._active is None:
            raise RuntimeError("no active run; call start_run first")
        return self._active

    def start_run(self, *, run_id: str, experiment_id: str, spec: ExperimentSpec, resume: bool = False) -> None:
        if self._active is not None:
            raise RuntimeError(f"run {self._active.run_id} is still active")
        earlier = [r for r in self.runs if r.run_id == run_id] if resume else []
        if earlier:
            self._active = earlier[-1]
            self._active.status = None
            return
        self._active = RecordedRun(run_id=run_id, experiment_id=experiment_id, spec=spec)
        self.runs.append(self._active)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        self.current.metrics.append((step, dict(metrics)))

    def end_run(self, status: RunStatus) -> None:
        self.current.status = status
        self._active = None
