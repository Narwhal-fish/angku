"""``diff_runs``: why two runs differ, read from their bundles.

Sections: the fingerprint (what makes the ExperimentID), the RunID-only inputs (resources,
attempt), labels outside every id (meta, tracking, image tag, dataset version), the actual
environment each run recorded, and the final metrics side by side.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.application.bundle import MANIFEST
from mlops.application.diff import ABSENT, Change, diff_runs
from mlops.config import load_experiment_dict, load_experiment_yaml
from mlops.ports.artifacts import ArtifactError
from mlops.registry import discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


@pytest.fixture
def spec():
    return load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")


@pytest.fixture
def store():
    return InMemoryArtifactStore()


def _data() -> SyntheticBlobs:
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)


def _run(store, spec, **kw) -> str:
    return run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, **kw).run_id


def _with(spec, **sections):
    d = spec.model_dump()
    for dotted, value in sections.items():
        *path, key = dotted.split("__")
        target = d
        for p in path:
            target = target[p]
        target[key] = value
    return load_experiment_dict(d)


# --- the experiment fingerprint ---


def test_same_spec_different_attempt_is_the_same_experiment(spec, store):
    a, b = _run(store, spec), _run(store, spec, attempt=2)
    d = diff_runs(store, a, store, b)
    assert d.same_experiment and not d.same_run
    assert d.experiment == []
    assert d.run == [Change("attempt", 1, 2)]
    assert d.labels == []


def test_changed_seed_is_named(spec, store):
    a, b = _run(store, spec), _run(store, _with(spec, training__seed=1))
    d = diff_runs(store, a, store, b)
    assert not d.same_experiment
    assert d.experiment == [Change("training.seed", 0, 1)]
    assert d.experiment_id_a != d.experiment_id_b


def test_list_of_sections_is_compared_by_index(spec, store):
    aug = [{"name": "rand_flip", "params": {"prob": 0.5, "spatial_axis": [0]}}]
    a = _run(store, _with(spec, augmentation=aug))
    b = _run(store, _with(spec, augmentation=[{**aug[0], "params": {"prob": 0.3, "spatial_axis": [0, 1]}}]))
    d = diff_runs(store, a, store, b)
    assert d.experiment == [
        Change("augmentation.0.params.prob", 0.5, 0.3),
        # a list of scalars is one value, not one change per element
        Change("augmentation.0.params.spatial_axis", [0], [0, 1]),
    ]


def test_added_section_shows_as_absent_on_one_side(spec, store):
    a = _run(store, spec)
    b = _run(store, _with(spec, metrics=[{"name": "dice", "params": {}}]))
    d = diff_runs(store, a, store, b)
    assert d.experiment == [Change("metrics.0.name", ABSENT, "dice"), Change("metrics.0.params", ABSENT, {})]


def test_labels_change_neither_id(spec, store):
    b_spec = _with(spec, meta__name="renamed", tracking__experiment_name="other",
                   environment__image="ghcr.io/x/y:9.9", dataset__version="v2")
    other = InMemoryArtifactStore()  # same run_id: in one store the second bundle would replace the first
    d = diff_runs(store, _run(store, spec), other, _run(other, b_spec))
    assert d.same_experiment and d.same_run
    assert [c.path for c in d.labels] == [
        "dataset.version", "environment.image", "meta.name", "tracking.experiment_name",
    ]
    assert d.experiment == [] and d.run == []


def test_resources_change_the_run_only(spec, store):
    d = diff_runs(store, _run(store, spec), store, _run(store, _with(spec, resources__cpu=4)))
    assert d.same_experiment and not d.same_run
    assert d.run == [Change("resources.cpu", 2, 4)]


# --- outcome ---


def test_final_metrics_side_by_side(spec, store):
    a, b = _run(store, spec), _run(store, _with(spec, training__seed=1))
    d = diff_runs(store, a, store, b)
    names = [m.name for m in d.metrics]
    assert names == ["train/loss", "val/loss"]
    for m in d.metrics:
        assert math.isfinite(m.a) and math.isfinite(m.b)
    assert d.status_a == d.status_b == "FINISHED"


def test_metric_present_on_one_side_only(spec, store):
    a = _run(store, spec)
    b = _run(store, _with(spec, metrics=[{"name": "dice", "params": {"include_background": False}}]))
    m = {x.name: x for x in diff_runs(store, a, store, b).metrics}
    assert m["val/dice/1"].a is ABSENT and math.isfinite(m["val/dice/1"].b)


def test_environment_differences_are_listed(spec, store):
    from tests.test_environment import FixedProbe

    a = _run(store, spec, probe=FixedProbe(commit_ok=True))
    b = _run(store, spec, attempt=2, probe=FixedProbe(commit_ok=False))
    env = {c.path: c for c in diff_runs(store, a, store, b).environment}
    assert env["code.changed"] == Change("code.changed", [], ["mlops/x.py"])
    assert "python" not in env  # equal values are not listed


def test_environment_not_recorded_on_one_side(spec, store):
    from tests.test_environment import FixedProbe

    a, b = _run(store, spec), _run(store, spec, attempt=2, probe=FixedProbe())
    d = diff_runs(store, a, store, b)
    assert d.environment == [Change("environment", "not recorded", "recorded")]


# --- bundles that are not FINISHED, and bad input ---


def test_failed_and_killed_runs_can_be_diffed(spec, store):
    a = _run(store, spec)
    b = _run(store, spec, attempt=2)
    store._runs[b].pop(MANIFEST)  # killed before any manifest landed
    d = diff_runs(store, a, store, b)
    assert d.status_b == "no manifest"
    assert d.run == []  # attempt unknown without a manifest: not claimed as a difference
    assert all(m.b is ABSENT for m in d.metrics)


def test_two_artifact_roots(spec):
    sa, sb = InMemoryArtifactStore(), InMemoryArtifactStore()
    d = diff_runs(sa, _run(sa, spec), sb, _run(sb, spec))
    assert d.same_run


def test_unknown_run_raises(store):
    with pytest.raises(ArtifactError):
        diff_runs(store, "a" * 64, store, "b" * 64)


def test_to_dict_is_json_ready(spec, store):
    import json

    d = diff_runs(store, _run(store, spec), store, _run(store, _with(spec, metrics=[{"name": "dice", "params": {}}])))
    out = json.loads(json.dumps(d.to_dict()))
    assert out["same_experiment"] is False
    first = out["experiment"][0]
    assert first == {"path": "metrics.0.name", "b": "dice"}  # absent side is omitted
