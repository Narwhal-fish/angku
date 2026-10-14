"""Port + local adapter for *where* a run executes.

``run_experiment`` is the in-process path. A ``Launcher`` has the same
signature so the CLI can swap in a distributed backend (Ray Train, and on
Kubernetes a KubeRay RayJob) without the caller noticing.
"""

from __future__ import annotations

from typing import Protocol

from mlops.application.runner import RunResult, run_experiment
from mlops.domain.experiment import ExperimentSpec
from mlops.ports.artifacts import ArtifactStore
from mlops.ports.data import DataProvider
from mlops.ports.environment import EnvironmentProbe
from mlops.ports.tracking import Tracker


class Launcher(Protocol):
    def launch(
        self,
        spec: ExperimentSpec,
        *,
        data: DataProvider,
        tracker: Tracker,
        artifacts: ArtifactStore | None = None,
        attempt: int = 1,
        device: str | None = None,
        probe: EnvironmentProbe | None = None,
        checkpoint_every: int | None = None,
        resume: bool = False,
    ) -> RunResult: ...


class LocalLauncher:
    """world_size 1, this process."""

    def launch(
        self, spec, *, data, tracker, artifacts=None, attempt=1, device=None, probe=None,
        checkpoint_every=None, resume=False,
    ) -> RunResult:
        return run_experiment(
            spec, data=data, tracker=tracker, artifacts=artifacts, attempt=attempt, device=device, probe=probe,
            checkpoint_every=checkpoint_every, resume=resume,
        )
