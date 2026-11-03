"""`reproduce_run`: prove, not assume, that a bundle reproduces its run.

The bundle promises "re-run from spec.json + manifest.attempt gives the same
run_id, history and weights". ``reproduce_run`` executes that promise and
reports which part held. The original bundle is evidence and is never
modified: the replay goes into a scratch store the caller supplies.

Proposed API (angku/application/reproduce.py):

    report = reproduce_run(
        store, run_id, data=..., tracker=..., scratch=ArtifactStore(), device=None, atol=0.0
    )
    report.ok                 all checks passed
    report.run_id             the run that was verified (as the bundle names it)
    report.experiment_id
    report.replay_run_id      what the replay actually computed
    report.checks             {"run_id": bool, "history": bool, "weights": bool}
    report.mismatches         [Mismatch(check=..., detail=...)] — human-readable, empty when ok
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch

from angku.adapters.inmemory_artifacts import InMemoryArtifactStore
from angku.adapters.inmemory_tracker import InMemoryTracker
from angku.adapters.local_artifacts import LocalArtifactStore
from angku.adapters.synthetic_data import SyntheticBlobs
from angku.application import run_experiment
from angku.application.bundle import MANIFEST, MODEL, save_bundle
from angku.application.launch import LocalLauncher
from angku.application.reproduce import (
    Mismatch,
    ReproduceError,
    ReproduceReport,
    replay_launcher,
    reproduce_lines,
    reproduce_run,
)
from angku.config import load_experiment_dict, load_experiment_yaml
from angku.domain import RunManifest, compute_experiment_id, compute_run_id
from angku.registry import discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


@pytest.fixture
def spec():
    return load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")


@pytest.fixture(params=["memory", "local"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryArtifactStore()
    return LocalArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def run_id(spec, store):
    """A FINISHED bundle in ``store``; returns its run_id."""
    return run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store).run_id


def _data() -> SyntheticBlobs:
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)


def _reproduce(store, run_id, **kw) -> ReproduceReport:
    return reproduce_run(store, run_id, data=_data(), tracker=InMemoryTracker(), scratch=InMemoryArtifactStore(), **kw)


# --- tampering helpers: corrupt one thing in the bundle, leave the rest alone ---


def _edit_manifest(store, run_id, **changes) -> None:
    m = json.loads(store.get(run_id, MANIFEST))
    m.update(changes)
    store.put(run_id, MANIFEST, json.dumps(m).encode())


def _edit_weights(store, run_id, delta: float) -> str:
    """Add ``delta`` to the first parameter tensor; returns its key."""
    state = torch.load(io.BytesIO(store.get(run_id, MODEL)), map_location="cpu", weights_only=True)
    key = next(iter(state))
    state[key] = state[key] + delta
    buf = io.BytesIO()
    torch.save(state, buf)
    store.put(run_id, MODEL, buf.getvalue())
    return key


def _snapshot(store, run_id) -> dict[str, bytes]:
    return {name: store.get(run_id, name) for name in store.list(run_id)}


# --- the happy path ---


def test_intact_bundle_is_reproduced(spec, store, run_id):
    report = _reproduce(store, run_id)

    assert report.ok
    assert report.run_id == run_id
    assert report.replay_run_id == run_id
    assert report.experiment_id == compute_experiment_id(spec)
    assert report.checks == {"run_id": True, "history": True, "weights": True}
    assert report.mismatches == []


def test_replay_goes_to_scratch_and_the_original_bundle_is_untouched(store, run_id):
    before = _snapshot(store, run_id)
    scratch = InMemoryArtifactStore()
    reproduce_run(store, run_id, data=_data(), tracker=InMemoryTracker(), scratch=scratch)

    assert _snapshot(store, run_id) == before
    assert set(scratch.list(run_id)) >= {MANIFEST, MODEL}


def test_replay_is_recorded_in_the_tracker(store, run_id):
    tracker = InMemoryTracker()
    reproduce_run(store, run_id, data=_data(), tracker=tracker, scratch=InMemoryArtifactStore())

    (rec,) = tracker.runs
    assert rec.run_id == run_id and rec.status == "FINISHED"


# --- each check catches its own kind of drift ---


def test_altered_history_fails_only_the_history_check(store, run_id):
    m = json.loads(store.get(run_id, MANIFEST))
    m["history"][0]["val/loss"] += 1.0
    store.put(run_id, MANIFEST, json.dumps(m).encode())

    report = _reproduce(store, run_id)

    assert not report.ok
    assert report.checks == {"run_id": True, "history": False, "weights": True}
    (mm,) = report.mismatches
    assert isinstance(mm, Mismatch) and mm.check == "history"
    assert "epoch 0" in mm.detail and "val/loss" in mm.detail


def test_history_of_different_length_is_a_mismatch(store, run_id):
    m = json.loads(store.get(run_id, MANIFEST))
    m["history"].append(dict(m["history"][-1]))
    store.put(run_id, MANIFEST, json.dumps(m).encode())

    report = _reproduce(store, run_id)

    assert report.checks["history"] is False
    (mm,) = report.mismatches
    assert "2" in mm.detail and "1" in mm.detail  # bundle says 2 epochs, replay produced 1


def test_altered_weights_fail_only_the_weights_check(store, run_id):
    key = _edit_weights(store, run_id, delta=1.0)

    report = _reproduce(store, run_id)

    assert not report.ok
    assert report.checks == {"run_id": True, "history": True, "weights": False}
    (mm,) = report.mismatches
    assert mm.check == "weights" and key in mm.detail


def test_missing_weight_key_is_a_mismatch(store, run_id):
    state = torch.load(io.BytesIO(store.get(run_id, MODEL)), map_location="cpu", weights_only=True)
    key = next(iter(state))
    del state[key]
    buf = io.BytesIO()
    torch.save(state, buf)
    store.put(run_id, MODEL, buf.getvalue())

    report = _reproduce(store, run_id)

    assert report.checks["weights"] is False
    (mm,) = report.mismatches
    assert key in mm.detail


def test_manifest_attempt_that_cannot_yield_its_run_id_fails_the_run_id_check(spec, store, run_id):
    # The manifest claims attempt 2 but its run_id was computed with attempt 1:
    # replaying "as the bundle says" lands on a different run_id.
    _edit_manifest(store, run_id, attempt=2)

    report = _reproduce(store, run_id)

    assert not report.ok
    assert report.checks["run_id"] is False
    assert report.run_id == run_id
    assert report.replay_run_id == compute_run_id(compute_experiment_id(spec), spec.resources, 2)
    assert any(mm.check == "run_id" for mm in report.mismatches)


# --- tolerance: bit-exact is the default, allclose is opt-in ---


def test_weights_are_compared_bit_exact_by_default(store, run_id):
    _edit_weights(store, run_id, delta=1e-7)
    assert _reproduce(store, run_id).checks["weights"] is False


def test_atol_relaxes_the_weight_check(store, run_id):
    _edit_weights(store, run_id, delta=1e-7)
    assert _reproduce(store, run_id, atol=1e-5).checks["weights"] is True


def test_atol_does_not_hide_real_drift(store, run_id):
    _edit_weights(store, run_id, delta=1.0)
    assert _reproduce(store, run_id, atol=1e-5).checks["weights"] is False


# --- what cannot be reproduced ---


def test_failed_bundle_is_refused(spec, store):
    exp_id = compute_experiment_id(spec)
    rid = compute_run_id(exp_id, spec.resources, 1)
    manifest = RunManifest(experiment_id=exp_id, run_id=rid, attempt=1, status="FAILED", device="cpu")
    save_bundle(store, spec=spec, manifest=manifest, model_state=None)

    with pytest.raises(ReproduceError, match="FAILED"):
        _reproduce(store, rid)


def test_replay_failure_propagates_and_is_not_reported_as_a_mismatch(spec, store, run_id):
    class ExplodingLoader(SyntheticBlobs):
        def load(self, dataset, preprocessing):
            raise RuntimeError("disk on fire")

    with pytest.raises(RuntimeError, match="disk on fire"):
        reproduce_run(store, run_id, data=ExplodingLoader(), tracker=InMemoryTracker(), scratch=InMemoryArtifactStore())


def test_undefined_metric_on_both_sides_is_not_a_mismatch(spec, store):
    from angku.config import load_experiment_dict
    from angku.registry import METRICS, Params

    class _NoParams(Params):
        pass

    if "_test_reproduce_nan" not in METRICS:

        @METRICS.register("_test_reproduce_nan", params=_NoParams)
        def _nan(p: _NoParams):
            return lambda logits, target: {"never": torch.full((logits.shape[0],), float("nan"))}

    spec = load_experiment_dict({**spec.model_dump(), "metrics": [{"name": "_test_reproduce_nan", "params": {}}]})
    run_id = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store).run_id
    report = _reproduce(store, run_id)
    assert report.ok, report.mismatches


# --- environment diff: explains a NOT REPRODUCED (and is shown even when it reproduces) ---


def test_replay_records_its_environment_and_diffs_it_against_the_original(spec, store):
    from tests.test_environment import FixedProbe

    run_id = run_experiment(
        spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, probe=FixedProbe(commit_ok=True)
    ).run_id
    report = _reproduce(store, run_id, probe=FixedProbe(commit_ok=False))
    assert report.ok  # same weights: the env difference did not matter this time
    changes = {c.path: c for c in report.environment}
    assert changes["code.changed"].a == [] and changes["code.changed"].b == ["angku/x.py"]


def test_same_environment_gives_no_environment_changes(spec, store):
    from tests.test_environment import FixedProbe

    run_id = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, probe=FixedProbe()).run_id
    assert _reproduce(store, run_id, probe=FixedProbe()).environment == []


def test_environment_not_recorded_on_either_side_is_said_once(spec, store, run_id):
    from tests.test_environment import FixedProbe

    report = _reproduce(store, run_id, probe=FixedProbe())
    assert [(c.path, c.a, c.b) for c in report.environment] == [("environment", "not recorded", "recorded")]


def test_environment_changes_are_in_the_json(spec, store):
    from tests.test_environment import FixedProbe

    run_id = run_experiment(
        spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, probe=FixedProbe(commit_ok=True)
    ).run_id
    out = json.loads(json.dumps(_reproduce(store, run_id, probe=FixedProbe(commit_ok=False)).to_dict()))
    assert {"path": "code.changed", "a": [], "b": ["angku/x.py"]} in out["environment"]


# --- the replay launcher: the run is replayed by the launcher it is given ---


class _RecordingLauncher:
    """Runs locally and records what it was asked to launch."""

    def __init__(self):
        self.calls = []

    def launch(self, spec, **kw):
        self.calls.append((spec, kw))
        return LocalLauncher().launch(spec, **kw)


def test_replay_runs_through_the_given_launcher(spec, store, run_id):
    launcher = _RecordingLauncher()
    scratch = InMemoryArtifactStore()
    report = reproduce_run(store, run_id, data=_data(), tracker=InMemoryTracker(), scratch=scratch, launcher=launcher)

    assert report.ok
    ((replayed, kw),) = launcher.calls
    assert replayed == spec
    assert kw["attempt"] == 1 and kw["artifacts"] is scratch and kw["keep_best"] is None


def test_report_names_the_launcher_of_the_replay(store, run_id):
    report = _reproduce(store, run_id)

    assert report.launcher == "local"
    assert report.to_dict()["launcher"] == "local"
    assert "  launcher       local" in reproduce_lines("tiny", report)


def _multi_worker_bundle(spec, store) -> str:
    spec = _with_workers(spec, 2)
    exp_id = compute_experiment_id(spec)
    rid = compute_run_id(exp_id, spec.resources, 1)
    manifest = RunManifest(experiment_id=exp_id, run_id=rid, attempt=1, status="FINISHED", device="cpu", launcher="ray")
    save_bundle(store, spec=spec, manifest=manifest, model_state={})
    return rid


def test_multi_worker_bundle_is_refused_by_the_local_launcher_before_anything_runs(spec, store):
    rid = _multi_worker_bundle(spec, store)
    tracker = InMemoryTracker()

    with pytest.raises(ReproduceError, match="2 workers"):
        reproduce_run(store, rid, data=_data(), tracker=tracker, scratch=InMemoryArtifactStore())
    assert tracker.runs == []


# --- which launcher replays a bundle (pure: the CLI maps the name to an adapter) ---


def _manifest(launcher):
    return RunManifest(experiment_id="e", run_id="r", attempt=1, status="FINISHED", device="cpu", launcher=launcher)


def _with_workers(spec, workers):
    return load_experiment_dict(
        {
            **spec.model_dump(),
            "training": {**spec.training.model_dump(), "batch_size": 4},
            "resources": {**spec.resources.model_dump(), "workers": workers},
        }
    )


@pytest.mark.parametrize("recorded", ["local", "ray"])
def test_the_recorded_launcher_replays_the_run(spec, recorded):
    assert replay_launcher(_manifest(recorded), spec) == recorded


@pytest.mark.parametrize(("workers", "expected"), [(1, "local"), (2, "ray")])
def test_a_bundle_without_a_recorded_launcher_goes_by_its_workers(spec, workers, expected):
    assert replay_launcher(_manifest(None), _with_workers(spec, workers)) == expected


def test_an_explicit_launcher_wins_over_the_recorded_one(spec):
    assert replay_launcher(_manifest("ray"), spec, override="local") == "local"
    assert replay_launcher(_manifest(None), _with_workers(spec, 2), override="local") == "local"
