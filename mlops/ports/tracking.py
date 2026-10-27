"""Port: where run metadata and metrics go (MLflow, W&B, a test double).

Every call carries the ids the domain already computed, so an adapter never
has to invent its own notion of experiment or run identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from mlops.domain.experiment import ExperimentSpec

RunStatus = Literal["FINISHED", "FAILED"]


class TrackerError(Exception):
    """Misconfigured backend or calls out of order (log before start, nested start)."""


class Tracker(Protocol):
    def start_run(self, *, run_id: str, experiment_id: str, spec: ExperimentSpec, resume: bool = False) -> None:
        """Open a run. ``resume=True``: continue the backend's earlier run with this ``run_id``
        (a resumed checkpoint) instead of opening a second one; start fresh if there is none."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None: ...

    def end_run(self, status: RunStatus) -> None: ...
