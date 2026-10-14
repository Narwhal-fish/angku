"""Use case: explain why two runs differ, from their bundles alone.

Turns "these runs differ" into "they differ because training.seed changed". Reads
``spec.json`` and ``manifest.json`` only (never the weights), so it is cheap for any model
and works on FAILED runs and on killed ones that left only a spec and a checkpoint.

Sections, in order of how much they matter for identity:

    experiment   fingerprint paths: these are what make the ExperimentIDs differ
    run          RunID-only inputs: resources, attempt
    labels       in no id: meta, tracking, code.repo, environment.image, dataset.version
    environment  what each run actually ran on (manifest.environment.actual)
    metrics      final value of every metric, side by side

Nested dicts and lists of sections (augmentation, metrics) are flattened to dotted paths
(``augmentation.0.params.prob``); a list of scalars is one value. A path present on one side
only is ``ABSENT`` on the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mlops.application.bundle import MANIFEST, SPEC
from mlops.domain.experiment import ExperimentSpec
from mlops.domain.identity import compute_experiment_id
from mlops.domain.manifest import RunManifest
from mlops.ports.artifacts import ArtifactError, ArtifactStore


class _Absent:
    """The path / metric does not exist on that side."""

    _instance: _Absent | None = None

    def __new__(cls) -> _Absent:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ABSENT"


ABSENT: Any = _Absent()


@dataclass(frozen=True)
class Change:
    path: str
    a: Any
    b: Any

    def to_dict(self) -> dict[str, Any]:
        return _sides({"path": self.path}, self.a, self.b)


@dataclass(frozen=True)
class MetricPair:
    name: str
    a: Any  # float or ABSENT
    b: Any

    @property
    def delta(self) -> float | None:
        if self.a is ABSENT or self.b is ABSENT:
            return None
        return self.b - self.a

    def to_dict(self) -> dict[str, Any]:
        return _sides({"name": self.name}, _json_float(self.a), _json_float(self.b))


@dataclass(frozen=True)
class RunDiff:
    run_a: str
    run_b: str
    experiment_id_a: str
    experiment_id_b: str
    status_a: str
    status_b: str
    attempt_a: int | None
    attempt_b: int | None
    experiment: list[Change] = field(default_factory=list)
    run: list[Change] = field(default_factory=list)
    labels: list[Change] = field(default_factory=list)
    environment: list[Change] = field(default_factory=list)
    metrics: list[MetricPair] = field(default_factory=list)

    @property
    def same_experiment(self) -> bool:
        return self.experiment_id_a == self.experiment_id_b

    @property
    def same_run(self) -> bool:
        return self.run_a == self.run_b

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_a": self.run_a,
            "run_b": self.run_b,
            "experiment_id_a": self.experiment_id_a,
            "experiment_id_b": self.experiment_id_b,
            "same_experiment": self.same_experiment,
            "same_run": self.same_run,
            "status_a": self.status_a,
            "status_b": self.status_b,
            "experiment": [c.to_dict() for c in self.experiment],
            "run": [c.to_dict() for c in self.run],
            "labels": [c.to_dict() for c in self.labels],
            "environment": [c.to_dict() for c in self.environment],
            "metrics": [m.to_dict() for m in self.metrics],
        }


def diff_runs(store_a: ArtifactStore, run_a: str, store_b: ArtifactStore, run_b: str) -> RunDiff:
    """Raise ``ArtifactError`` when either run has no ``spec.json``."""
    spec_a, man_a = _read(store_a, run_a)
    spec_b, man_b = _read(store_b, run_b)
    attempt_a = man_a.attempt if man_a else None
    attempt_b = man_b.attempt if man_b else None

    run = _changes(_flatten({"resources": spec_a.resources.model_dump()}),
                   _flatten({"resources": spec_b.resources.model_dump()}))
    if attempt_a is not None and attempt_b is not None and attempt_a != attempt_b:
        run.append(Change("attempt", attempt_a, attempt_b))

    exp_a = compute_experiment_id(spec_a)
    exp_b = compute_experiment_id(spec_b)
    return RunDiff(
        run_a=man_a.run_id if man_a else run_a,
        run_b=man_b.run_id if man_b else run_b,
        experiment_id_a=exp_a,
        experiment_id_b=exp_b,
        status_a=man_a.status if man_a else "no manifest",
        status_b=man_b.status if man_b else "no manifest",
        attempt_a=attempt_a,
        attempt_b=attempt_b,
        experiment=_changes(_flatten(spec_a.fingerprint()), _flatten(spec_b.fingerprint())),
        run=run,
        labels=_changes(_flatten(_labels(spec_a)), _flatten(_labels(spec_b))),
        environment=diff_environments(man_a, man_b),
        metrics=_metrics(man_a, man_b),
    )


# -- helpers -------------------------------------------------------------------


def _read(store: ArtifactStore, run_id: str) -> tuple[ExperimentSpec, RunManifest | None]:
    spec = ExperimentSpec.model_validate_json(store.get(run_id, SPEC))
    try:
        manifest = RunManifest.model_validate_json(store.get(run_id, MANIFEST))
    except ArtifactError:  # killed before any manifest was written
        manifest = None
    return spec, manifest


def _labels(spec: ExperimentSpec) -> dict[str, Any]:
    """Everything in the spec that is in no id (fingerprint and resources are elsewhere)."""
    return {
        "meta": spec.meta.model_dump(),
        "tracking": spec.tracking.model_dump(),
        "code": {"repo": spec.code.repo},
        "environment": {"image": spec.environment.image},
        "dataset": {"version": spec.dataset.version},
    }


def diff_environments(a: RunManifest | None, b: RunManifest | None) -> list[Change]:
    """What differs between the environments two runs recorded (``manifest.environment.actual``)."""
    env_a = a.environment if a else None
    env_b = b.environment if b else None
    if env_a is None and env_b is None:
        return []
    if env_a is None or env_b is None:
        side = lambda e: "recorded" if e is not None else "not recorded"  # noqa: E731
        return [Change("environment", side(env_a), side(env_b))]
    return _changes(_flatten(env_a.actual.model_dump()), _flatten(env_b.actual.model_dump()))


def _metrics(a: RunManifest | None, b: RunManifest | None) -> list[MetricPair]:
    final_a = a.history[-1] if a and a.history else {}
    final_b = b.history[-1] if b and b.history else {}
    names = [*final_a, *(k for k in final_b if k not in final_a)]
    return [MetricPair(k, final_a.get(k, ABSENT), final_b.get(k, ABSENT)) for k in names]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
        out = {}
        for i, v in enumerate(value):
            out.update(_flatten(v, f"{prefix}.{i}"))
        return out
    return {prefix: value}


def _changes(a: dict[str, Any], b: dict[str, Any]) -> list[Change]:
    paths = sorted(a.keys() | b.keys(), key=_natural)
    return [
        Change(p, a.get(p, ABSENT), b.get(p, ABSENT))
        for p in paths
        if a.get(p, ABSENT) != b.get(p, ABSENT)
    ]


def _natural(path: str) -> list[tuple[int, Any]]:
    # augmentation.10 sorts after augmentation.2
    return [(0, int(p)) if p.isdigit() else (1, p) for p in path.split(".")]


def _sides(out: dict[str, Any], a: Any, b: Any) -> dict[str, Any]:
    if a is not ABSENT:
        out["a"] = a
    if b is not ABSENT:
        out["b"] = b
    return out


def _json_float(v: Any) -> Any:
    return None if isinstance(v, float) and math.isnan(v) else v

