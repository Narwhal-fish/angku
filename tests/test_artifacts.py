"""Artifact store port + adapters, and the run bundle ("the legacy") built on it.

The bundle is the contract for reproducibility: given only what a store holds
for a run_id, a stranger must be able to re-run the experiment and get the
same numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.application.bundle import MANIFEST, MODEL, SPEC, load_bundle
from mlops.config import load_experiment_yaml
from mlops.domain import RunManifest
from mlops.ports.artifacts import ArtifactError
from mlops.registry import MODELS, discover

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


def _data() -> SyntheticBlobs:
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)


# --- port contract, same for every adapter ---


def test_put_get_roundtrip(store):
    store.put("run-a", "notes.txt", b"hello")
    assert store.get("run-a", "notes.txt") == b"hello"


def test_list_is_sorted_and_per_run(store):
    store.put("run-a", "b.bin", b"1")
    store.put("run-a", "checkpoints/final.pt", b"2")
    store.put("run-a", "a.bin", b"3")
    store.put("run-b", "z.bin", b"4")
    assert store.list("run-a") == ["a.bin", "b.bin", "checkpoints/final.pt"]
    assert store.list("run-b") == ["z.bin"]


def test_put_overwrites(store):
    store.put("run-a", "x", b"old")
    store.put("run-a", "x", b"new")
    assert store.get("run-a", "x") == b"new"


def test_missing_artifact_and_unknown_run_raise(store):
    store.put("run-a", "x", b"")
    with pytest.raises(ArtifactError, match="nope"):
        store.get("run-a", "nope")
    with pytest.raises(ArtifactError, match="run-zzz"):
        store.get("run-zzz", "x")
    with pytest.raises(ArtifactError, match="run-zzz"):
        store.list("run-zzz")


@pytest.mark.parametrize("bad", ["../escape", "/abs/path", "", "a/../../b", "run\\x"])
def test_unsafe_names_are_rejected(store, bad):
    with pytest.raises(ArtifactError):
        store.put("run-a", bad, b"")


def test_local_store_layout_is_run_id_then_name(tmp_path):
    root = tmp_path / "artifacts"
    LocalArtifactStore(root).put("run-a", "checkpoints/final.pt", b"x")
    assert (root / "run-a" / "checkpoints" / "final.pt").read_bytes() == b"x"


# --- RunManifest (domain) ---


def test_manifest_rejects_typos():
    with pytest.raises(ValidationError):
        RunManifest(
            experiment_id="e",
            run_id="r",
            attempt=1,
            status="FINISHED",
            device="cpu",
            history=[],
            statuss="x",
        )


# --- bundle written by the runner ---


def test_finished_run_writes_spec_manifest_and_checkpoint(spec, store):
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store)

    assert store.list(result.run_id) == sorted([MANIFEST, MODEL, SPEC])

    manifest = RunManifest.model_validate_json(store.get(result.run_id, MANIFEST))
    assert manifest.experiment_id == result.experiment_id
    assert manifest.run_id == result.run_id
    assert manifest.attempt == 1
    assert manifest.status == "FINISHED"
    assert manifest.device == "cpu"
    assert manifest.history == result.history

    # spec.json is the full spec, not just the fingerprint: meta/tracking are needed to re-run.
    assert json.loads(store.get(result.run_id, SPEC)) == spec.model_dump(mode="json")


def test_checkpoint_loads_into_a_fresh_model(spec, store):
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store)
    bundle = load_bundle(store, result.run_id)

    assert bundle.spec == spec
    assert bundle.manifest.run_id == result.run_id
    assert bundle.model_state is not None

    model = MODELS.build(spec.model.name, spec.model.params)
    model.load_state_dict(bundle.model_state, strict=True)


def test_run_can_be_reproduced_from_its_bundle_alone(spec, store):
    first = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store)
    bundle = load_bundle(store, first.run_id)

    # Only the bundle crosses this line.
    again = InMemoryArtifactStore()
    second = run_experiment(
        bundle.spec,
        data=_data(),
        tracker=InMemoryTracker(),
        artifacts=again,
        attempt=bundle.manifest.attempt,
    )

    assert second.experiment_id == first.experiment_id
    assert second.run_id == first.run_id
    assert second.history == bundle.manifest.history
    replay = load_bundle(again, second.run_id)
    for k, v in bundle.model_state.items():
        assert torch.equal(v, replay.model_state[k]), k


def test_failed_run_still_writes_spec_and_manifest(spec, store):
    class ExplodingLoader(SyntheticBlobs):
        def load(self, dataset, preprocessing):
            splits = super().load(dataset, preprocessing)
            bad = torch.utils.data.TensorDataset(torch.zeros(2, 3, 8, 8, 8), torch.zeros(2, 8, 8, 8, dtype=torch.long))
            return type(splits)(train=bad, val=splits.val)

    tracker = InMemoryTracker()
    with pytest.raises(RuntimeError):
        run_experiment(spec, data=ExplodingLoader(n_train=2, n_val=0), tracker=tracker, artifacts=store)
    run_id = tracker.runs[0].run_id
    assert store.list(run_id) == sorted([MANIFEST, SPEC])
    bundle = load_bundle(store, run_id)
    assert bundle.manifest.status == "FAILED"
    assert bundle.manifest.history == []
    assert bundle.model_state is None


def test_no_store_means_nothing_is_written(spec):
    # artifacts is optional; the runner must not require one.
    run_experiment(spec, data=_data(), tracker=InMemoryTracker())


def test_load_bundle_unknown_run(store):
    with pytest.raises(ArtifactError):
        load_bundle(store, "nope")


def test_undefined_metric_survives_the_bundle_roundtrip(store):
    # NaN = "metric undefined" (e.g. Dice on an absent class); plain JSON would turn it into null.
    import math

    from mlops.application.bundle import save_bundle

    spec = load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")
    manifest = RunManifest(
        experiment_id="e",
        run_id="r",
        attempt=1,
        status="FAILED",
        device="cpu",
        history=[{"val/dice/1": float("nan"), "val/loss": 0.5}],
    )
    save_bundle(store, spec=spec, manifest=manifest, model_state=None)
    loaded = load_bundle(store, "r").manifest.history[0]
    assert math.isnan(loaded["val/dice/1"]) and loaded["val/loss"] == 0.5


def test_manifest_without_resumes_still_loads():
    # Bundles written before checkpoint/resume existed have no "resumes" key.
    m = RunManifest.model_validate_json(
        '{"experiment_id": "e", "run_id": "r", "attempt": 1, "status": "FINISHED", "device": "cpu"}'
    )
    assert m.resumes == []


def test_runs_lists_every_run_sorted(store):
    assert store.runs() == []
    store.put("r2", "a.txt", b"x")
    store.put("r1", "sub/b.txt", b"y")
    assert store.runs() == ["r1", "r2"]


def test_local_runs_ignores_stray_files_and_missing_root(tmp_path):
    assert LocalArtifactStore(tmp_path / "nope").runs() == []
    store = LocalArtifactStore(tmp_path)
    store.put("r1", "a.txt", b"x")
    (tmp_path / "notes.txt").write_text("not a run")
    assert store.runs() == ["r1"]


def test_delete_removes_one_artifact_and_is_idempotent(store):
    store.put("r1", "a.bin", b"1")
    store.put("r1", "b/c.bin", b"2")
    store.delete("r1", "a.bin")
    assert store.list("r1") == ["b/c.bin"]
    store.delete("r1", "a.bin")  # already gone: nothing to do
    store.delete("never-written", "a.bin")
    with pytest.raises(ArtifactError):
        store.delete("r1", "../escape")


def test_local_delete_prunes_only_directories_it_emptied(tmp_path):
    store = LocalArtifactStore(tmp_path)
    store.put("r", "a/b/c.txt", b"1")
    store.put("r", "keep.txt", b"2")
    store.delete("r", "a/b/c.txt")
    assert not (tmp_path / "r" / "a").exists()  # emptied subdirectories go
    assert store.list("r") == ["keep.txt"]  # the run itself stays while it holds files
    store.delete("r", "keep.txt")
    assert not (tmp_path / "r").exists() and store.runs() == []
    store.delete("r", "keep.txt")  # still idempotent
