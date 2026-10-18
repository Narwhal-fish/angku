"""What the UI can do besides reading: injected by the CLI, which owns the adapters
(data provider, tracker, launcher), so the UI never picks one itself. No textual import."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mlops.adapters.kuberay import RayJobStatus
from mlops.application.reproduce import ReproduceReport
from mlops.application.runner import RunResult


@dataclass(frozen=True)
class UiActions:
    reproduce: Callable[[str], ReproduceReport]  # run_id -> report; may take as long as the training
    resume: Callable[[str], RunResult]  # run_id -> the finished run
    rayjobs: Callable[[], list[RayJobStatus]] | None = None  # our RayJobs in the namespace (kubectl)
    rayjob_logs: Callable[[str], str] | None = None  # RayJob name -> the tail of its driver log
