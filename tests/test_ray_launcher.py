"""Ray Train launcher: the same train_worker on N ranks, tracker/artifacts stay on the driver.

Runs a local Ray cluster on CPUs. Slow (~10-20s per fit); needs the `distributed` extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("ray.train")
import ray  # noqa: E402

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore  # noqa: E402
from mlops.adapters.inmemory_tracker import InMemoryTracker  # noqa: E402
from mlops.adapters.ray_launcher import RayLauncher  # noqa: E402
from mlops.adapters.synthetic_data import SyntheticBlobs  # noqa: E402
from mlops.application import RunnerError, run_experiment  # noqa: E402
from mlops.application.bundle import load_bundle  # noqa: E402
from mlops.config import load_experiment_dict, load_experiment_yaml  # noqa: E402
from mlops.domain import compute_run_id  # noqa: E402
from mlops.registry import discover  # noqa: E402

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _cluster():
    discover("experiments")
    ray.init(num_cpus=8, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


@pytest.fixture
def launcher(tmp_path):
    return RayLauncher(storage_path=tmp_path / "ray")


def _spec(workers: int, **overrides):
    base = load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")
    d = base.model_dump()
    d["resources"] = {**d["resources"], "cpu": 1, "workers": workers}
    for k, v in overrides.items():
        section, key = k.split(".")
        d[section] = {**d[section], key: v}
    return load_experiment_dict(d)


def _data(**kw) -> SyntheticBlobs:
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2, **kw)


def test_one_ray_worker_reproduces_the_local_run(launcher):
    spec = _spec(workers=1)
    local_store, ray_store = InMemoryArtifactStore(), InMemoryArtifactStore()
    local = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=local_store)

    tracker = InMemoryTracker()
    result = launcher.launch(spec, data=_data(), tracker=tracker, artifacts=ray_store)

    assert result.run_id == local.run_id
    assert result.history == local.history
    run = tracker.runs[0]
    assert run.status == "FINISHED" and run.run_id == result.run_id
    assert run.metrics == [(1, result.history[0])]

    a, b = load_bundle(local_store, local.run_id), load_bundle(ray_store, result.run_id)
    assert b.manifest.status == "FINISHED" and b.manifest.history == local.history
    # Bit-exactness is only promised within one launcher/environment: Ray workers run with
    # OMP_NUM_THREADS=1, so CPU kernels sum in a different order than the local run.
    for k, v in a.model_state.items():
        assert torch.allclose(v, b.model_state[k], atol=1e-6, rtol=0), k


def test_two_cpu_workers_split_the_global_batch(launcher):
    spec = _spec(workers=2, **{"training.epochs": 2})
    assert spec.per_worker_batch_size() == 1
    tracker, store = InMemoryTracker(), InMemoryArtifactStore()

    result = launcher.launch(spec, data=_data(), tracker=tracker, artifacts=store)

    assert result.run_id == compute_run_id(result.experiment_id, spec.resources, attempt=1)
    assert len(result.history) == 2
    assert all(torch.isfinite(torch.tensor(v)) for m in result.history for v in m.values())
    assert tracker.runs[0].status == "FINISHED"
    assert [step for step, _ in tracker.runs[0].metrics] == [1, 2]
    bundle = load_bundle(store, result.run_id)
    assert bundle.manifest.status == "FINISHED"
    assert bundle.spec == spec
    from mlops.registry import MODELS

    MODELS.build(spec.model.name, spec.model.params).load_state_dict(bundle.model_state, strict=True)


def test_gpu_workers_without_gpus_fail_before_starting(launcher):
    if ray.cluster_resources().get("GPU", 0) > 0:
        pytest.skip("cluster has GPUs; nothing to refuse")
    spec = _spec(workers=2, **{"resources.gpu": 1})
    tracker = InMemoryTracker()
    with pytest.raises(RunnerError, match="GPU"):
        launcher.launch(spec, data=_data(), tracker=tracker)
    assert tracker.runs == []


def test_worker_failure_marks_run_failed_and_keeps_failed_bundle(launcher):
    spec = _spec(workers=2, **{"model.params": {"in_channels": 3, "out_channels": 2, "hidden": 4, "spatial_dims": 3}})
    tracker, store = InMemoryTracker(), InMemoryArtifactStore()
    with pytest.raises(Exception):  # Ray wraps the worker's RuntimeError
        launcher.launch(spec, data=_data(), tracker=tracker, artifacts=store)
    run = tracker.runs[0]
    assert run.status == "FAILED"
    bundle = load_bundle(store, run.run_id)
    assert bundle.manifest.status == "FAILED" and bundle.model_state is None


def test_environment_is_captured_on_the_training_worker(launcher):
    import os

    from tests.test_environment import FixedProbe

    store = InMemoryArtifactStore()
    result = launcher.launch(_spec(workers=1), data=_data(), tracker=InMemoryTracker(), artifacts=store,
                             probe=FixedProbe())
    env = load_bundle(store, result.run_id).manifest.environment
    assert env is not None and env.checks["code.commit"].status == "ok"
    assert env.actual.platform != f"pid={os.getpid()}"  # the rank-0 worker, not the driver
