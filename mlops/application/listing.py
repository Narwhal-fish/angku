"""Use case: what is under an artifact root, run by run and experiment by experiment.

Reads ``spec.json`` + ``manifest.json`` only (never weights), so scanning hundreds of
SwinUNETR bundles is cheap. ``group_by_experiment`` answers the question content-addressed
ids exist for: did every FINISHED attempt of this ExperimentID end with the same final
metrics? Exactly by default; ``atol`` > 0 accepts runs whose final metrics are within ``atol``
of each other (max - min), for runs from different launchers. NaN agrees only with NaN.

Statuses: FINISHED / FAILED (from the manifest), RUNNING (no manifest, a live ``progress.json``:
see ``application.progress``; never resumable), INTERRUPTED (no manifest but a
checkpoint.pt or a Ray launch.json: killed, resumable), INCOMPLETE (no manifest, no checkpoint), UNREADABLE
(spec or manifest does not parse; ``error`` says why). Directories without ``spec.json``
are not runs and are skipped.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from mlops.application.bundle import CHECKPOINT, LAUNCH, MANIFEST, SPEC
from mlops.application.progress import PROGRESS, is_live, read_progress
from mlops.domain.best import BestRecord
from mlops.domain.experiment import ExperimentSpec
from mlops.domain.identity import compute_experiment_id
from mlops.domain.manifest import RunManifest
from mlops.ports.artifacts import ArtifactError, ArtifactStore


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    experiment_id: str
    name: str
    status: str
    attempt: int | None = None
    epochs_done: int | None = None
    epochs_total: int | None = None
    final_metrics: dict[str, float] = field(default_factory=dict)
    workers: int | None = None
    resumes: int = 0
    # A checkpoint exists and the run did not finish: `mlops resume` can continue it.
    resumable: bool = False
    error: str | None = None
    best: BestRecord | None = None  # keep-best runs: which epoch best.pt holds
    updated: float | None = None  # RUNNING: unix time of the last heartbeat

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "attempt": self.attempt,
            "epochs_done": self.epochs_done,
            "epochs_total": self.epochs_total,
            "final_metrics": {k: None if math.isnan(v) else v for k, v in self.final_metrics.items()},
            "workers": self.workers,
            "resumes": self.resumes,
            "resumable": self.resumable,
            "error": self.error,
            "best": self.best.model_dump() if self.best is not None else None,
            "updated": self.updated,
        }


@dataclass(frozen=True)
class ExperimentGroup:
    experiment_id: str
    name: str
    runs: list[RunSummary]
    atol: float = 0.0

    @property
    def finished(self) -> int:
        return sum(r.status == "FINISHED" for r in self.runs)

    @property
    def spread(self) -> dict[str, tuple[float, float]]:
        """(min, max) of every final metric that differs (by more than ``atol``) between the
        FINISHED runs. A metric NaN in some runs and defined in others always differs."""
        done = [r.final_metrics for r in self.runs if r.status == "FINISHED"]
        out: dict[str, tuple[float, float]] = {}
        for k in sorted({k for m in done for k in m}):
            values = [m.get(k, math.nan) for m in done]
            defined = [v for v in values if not math.isnan(v)]
            if not defined:
                continue  # NaN everywhere: undefined alike
            lo, hi = min(defined), max(defined)
            if len(defined) < len(values) or hi - lo > self.atol:
                out[k] = (lo, hi)
        return out

    @property
    def metrics_agree(self) -> bool | None:
        """True / False across >= 2 FINISHED runs; None when there is nothing to compare."""
        return None if self.finished < 2 else not self.spread

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "finished": self.finished,
            "metrics_agree": self.metrics_agree,
            "atol": self.atol,
            "spread": {k: list(v) for k, v in self.spread.items()},
            "runs": [r.to_dict() for r in self.runs],
        }


def list_runs(store: ArtifactStore, *, experiment: str | None = None, now: float | None = None) -> list[RunSummary]:
    """Every run in ``store``; ``experiment`` keeps those whose ExperimentID starts with it.
    ``now`` (default: the clock) decides whether a run's progress is still live."""
    now = time.time() if now is None else now
    out = []
    for run_id in store.runs():
        files = store.list(run_id)
        if SPEC not in files:
            continue
        s = _summarise(store, run_id, files, now)
        if experiment is None or s.experiment_id.startswith(experiment):
            out.append(s)
    return sorted(out, key=lambda r: (r.name, r.experiment_id, r.attempt or 0, r.run_id))


def summarise_run(store: ArtifactStore, run_id: str, *, now: float | None = None) -> RunSummary | None:
    """The ``list_runs`` summary of one run; None when ``run_id`` is not a run (gone, or no spec.json)."""
    now = time.time() if now is None else now
    try:
        files = store.list(run_id)
    except ArtifactError:
        return None
    return _summarise(store, run_id, files, now) if SPEC in files else None


def group_by_experiment(runs: list[RunSummary], *, atol: float = 0.0) -> list[ExperimentGroup]:
    groups: dict[str, list[RunSummary]] = {}
    for r in runs:
        groups.setdefault(r.experiment_id, []).append(r)
    return [ExperimentGroup(exp, rs[0].name, rs, atol) for exp, rs in groups.items()]


def _summarise(store: ArtifactStore, run_id: str, files: list[str], now: float) -> RunSummary:
    try:
        spec = ExperimentSpec.model_validate_json(store.get(run_id, SPEC))
        manifest = RunManifest.model_validate_json(store.get(run_id, MANIFEST)) if MANIFEST in files else None
    except (ValidationError, ValueError) as e:
        return RunSummary(run_id=run_id, experiment_id="?", name="?", status="UNREADABLE", error=str(e).splitlines()[0])
    # local runs keep checkpoint.pt here; Ray runs keep theirs in Ray storage and leave launch.json
    has_ckpt = CHECKPOINT in files or LAUNCH in files
    base = dict(
        run_id=run_id,
        experiment_id=compute_experiment_id(spec),
        name=spec.meta.name,
        epochs_total=spec.training.epochs,
        workers=spec.resources.workers,
    )
    if manifest is None:
        progress = read_progress(store, run_id) if PROGRESS in files else None
        if progress is not None and is_live(progress, now=now):
            return RunSummary(
                **base,
                status="RUNNING",
                epochs_done=progress.epoch,
                final_metrics=dict(progress.history[-1]) if progress.history else {},
                updated=progress.updated,
            )
        return RunSummary(**base, status="INTERRUPTED" if has_ckpt else "INCOMPLETE", resumable=has_ckpt)
    return RunSummary(
        **base,
        status=manifest.status,
        attempt=manifest.attempt,
        epochs_done=len(manifest.history),
        final_metrics=dict(manifest.history[-1]) if manifest.history else {},
        resumes=len(manifest.resumes),
        resumable=has_ckpt and manifest.status != "FINISHED",
        best=manifest.best,
    )
