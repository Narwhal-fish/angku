"""Ray Train launcher: the same train_worker on N ranks, tracker/artifacts stay on the driver.

Runs a local Ray cluster on CPUs. Slow (~10-20s per fit); needs the `distributed` extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("ray.train")
import ray  # noqa: E402
from ray.train import TrainingFailedError  # noqa: E402

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore  # noqa: E402
from mlops.adapters.inmemory_tracker import InMemoryTracker  # noqa: E402
from mlops.adapters.ray_launcher import RayLauncher  # noqa: E402
from mlops.adapters.synthetic_data import SyntheticBlobs  # noqa: E402
from mlops.application import RunnerError, run_experiment  # noqa: E402
from mlops.application.bundle import load_bundle  # noqa: E402
from mlops.config import load_experiment_dict, load_experiment_yaml  # noqa: E402
from mlops.domain import compute_experiment_id, compute_run_id  # noqa: E402
from mlops.domain.best import BestCriterion, best_record  # noqa: E402
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
    return SyntheticBlobs(**{"n_train": 4, "n_val": 2, "shape": (8, 8, 8), "in_channels": 1, "num_classes": 2, **kw})


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
    assert (a.manifest.launcher, b.manifest.launcher) == ("local", "ray")
    # Bit-exactness is only promised within one launcher/environment: Ray workers run with
    # OMP_NUM_THREADS=1, so CPU kernels sum in a different order than the local run.
    for k, v in a.model_state.items():
        assert torch.allclose(v, b.model_state[k], atol=1e-6, rtol=0), k


def test_ray_keep_best_matches_the_local_run(launcher):
    spec = _spec(workers=1, **{"training.epochs": 3})
    crit = BestCriterion(metric="val/loss", mode="max")
    local_store, ray_store = InMemoryArtifactStore(), InMemoryArtifactStore()
    local = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=local_store, keep_best=crit)
    result = launcher.launch(spec, data=_data(), tracker=InMemoryTracker(), artifacts=ray_store, keep_best=crit)
    a, b = load_bundle(local_store, local.run_id), load_bundle(ray_store, result.run_id)
    assert b.manifest.best == best_record(result.history, crit) == a.manifest.best
    for k, v in a.best_state.items():
        assert torch.allclose(v, b.best_state[k], atol=1e-6, rtol=0), k


def test_ray_run_applies_and_records_the_spec_split(launcher):
    d = _spec(workers=1).model_dump()
    d["split"] = {"kind": "holdout", "val": 0.34, "test": 0.17}
    spec = load_experiment_dict(d)
    local_store, ray_store = InMemoryArtifactStore(), InMemoryArtifactStore()
    local = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=local_store)
    result = launcher.launch(spec, data=_data(), tracker=InMemoryTracker(), artifacts=ray_store)
    assert result.split == local.split and result.split.counts() == {"train": 3, "val": 2, "test": 1}
    assert result.history == local.history
    bundle = load_bundle(ray_store, result.run_id)
    assert bundle.split == local.split and bundle.manifest.split == local.split.counts()


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
    with pytest.raises(TrainingFailedError):  # Ray wraps the worker's RuntimeError
        launcher.launch(spec, data=_data(), tracker=tracker, artifacts=store)
    run = tracker.runs[0]
    assert run.status == "FAILED"
    bundle = load_bundle(store, run.run_id)
    assert bundle.manifest.status == "FAILED" and bundle.model_state is None


def test_environment_is_captured_on_the_training_worker(launcher):
    import os

    from tests.test_environment import FixedProbe

    store = InMemoryArtifactStore()
    result = launcher.launch(
        _spec(workers=1), data=_data(), tracker=InMemoryTracker(), artifacts=store, probe=FixedProbe()
    )
    env = load_bundle(store, result.run_id).manifest.environment
    assert env is not None and env.checks["code.commit"].status == "ok"
    assert env.actual.platform != f"pid={os.getpid()}"  # the rank-0 worker, not the driver
    assert load_bundle(store, result.run_id).manifest.data_snapshot.status == "unknown"  # synthetic data


def test_wrong_data_never_starts_a_ray_run(launcher):
    from mlops.ports.data import DataError
    from tests.test_data_snapshot import A, B, HashedBlobs

    spec = _spec(workers=1, **{"dataset.snapshot": A})
    tracker = InMemoryTracker()
    with pytest.raises(DataError, match="dataset.snapshot"):
        launcher.launch(spec, data=HashedBlobs(B), tracker=tracker)
    assert tracker.runs == []


@pytest.mark.parametrize("n_val", [2, 3])
def test_metrics_are_aggregated_over_every_rank(launcher, n_val):
    # n_val=3 on 2 ranks: DistributedSampler would pad by repeating a case, counting it twice.
    # Every val case must count exactly once, as in a single-process run.
    from mlops.registry import LOSSES, METRICS, MODELS

    spec = load_experiment_dict({**_spec(workers=2).model_dump(), "metrics": [{"name": "dice", "params": {}}]})
    store = InMemoryArtifactStore()
    result = launcher.launch(spec, data=_data(n_val=n_val), tracker=InMemoryTracker(), artifacts=store)

    # Recompute on the driver over the whole val split with the final weights: each rank only
    # saw half of it, so this matches only if the per-rank sums were all-reduced.
    model = MODELS.build(spec.model.name, spec.model.params)
    model.load_state_dict(load_bundle(store, result.run_id).model_state)
    model.eval()
    x, y = _data(n_val=n_val).load(spec.dataset, spec.preprocessing).val.tensors
    with torch.no_grad():
        logits = model(x)
        per_sample = METRICS.build("dice", {})(logits, y)
        loss = LOSSES.build(spec.loss.name, spec.loss.params)(logits, y).item()
    assert result.final_metrics["val/loss"] == pytest.approx(loss, abs=1e-6)
    for key, values in per_sample.items():
        expected = values[~values.isnan()].double().mean().item()
        assert result.final_metrics[f"val/{key}"] == pytest.approx(expected, abs=1e-6), key


# --- live metrics: rank 0 streams each epoch to the driver's tracker while training runs ---


class _Gate(torch.utils.data.Dataset):
    """Wraps a train split. From the first item of epoch 2 on, waits for ``gate`` to exist
    (or raises, with ``fail``). Counting happens in the worker process."""

    def __init__(self, inner, gate: str, fail: bool) -> None:
        self.inner, self.gate, self.fail, self.seen = inner, gate, fail, 0

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        import time

        self.seen += 1
        if self.seen > len(self.inner):  # epoch 2
            if self.fail:
                raise RuntimeError("disk on fire in epoch 2")
            deadline = time.monotonic() + 60
            while not Path(self.gate).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("epoch 1 metrics never reached the driver's tracker")
                time.sleep(0.05)
        return self.inner[i]


class GatedBlobs(SyntheticBlobs):
    def __init__(self, gate: str, fail: bool = False) -> None:
        super().__init__(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)
        self.gate, self.fail = gate, fail

    def load(self, dataset, preprocessing):
        from mlops.ports.data import DataSplits

        splits = super().load(dataset, preprocessing)
        return DataSplits(train=_Gate(splits.train, self.gate, self.fail), val=splits.val)


class GateTracker(InMemoryTracker):
    """Opens the gate when epoch 1 arrives."""

    def __init__(self, gate: str) -> None:
        super().__init__()
        self.gate = gate

    def log_metrics(self, metrics, *, step):
        super().log_metrics(metrics, step=step)
        if step == 1:
            Path(self.gate).touch()


def test_metrics_reach_the_tracker_while_training(launcher, tmp_path):
    gate = str(tmp_path / "epoch1-logged")
    spec = _spec(workers=1, **{"training.epochs": 3})
    tracker = GateTracker(gate)
    result = launcher.launch(spec, data=GatedBlobs(gate), tracker=tracker)
    # Epoch 2 only ran because epoch 1 was already in the tracker: logging is live, not replayed.
    run = tracker.runs[0]
    assert run.metrics == [(e, m) for e, m in enumerate(result.history, start=1)]  # once each, in order
    assert run.status == "FINISHED"


def test_failed_ray_run_keeps_the_epochs_it_finished(launcher, tmp_path):
    spec = _spec(workers=1, **{"training.epochs": 3})
    tracker, store = InMemoryTracker(), InMemoryArtifactStore()
    with pytest.raises(TrainingFailedError):
        launcher.launch(spec, data=GatedBlobs(str(tmp_path / "unused"), fail=True), tracker=tracker, artifacts=store)
    run = tracker.runs[0]
    assert run.status == "FAILED"
    assert [step for step, _ in run.metrics] == [1]
    manifest = load_bundle(store, run.run_id).manifest
    assert manifest.status == "FAILED" and len(manifest.history) == 1
    assert manifest.history[0] == run.metrics[0][1]


def test_relay_logs_in_order_and_drains_on_stop():
    import queue

    from mlops.adapters.ray_launcher import MetricsRelay

    q: queue.Queue = queue.Queue()
    tracker = InMemoryTracker()
    tracker.start_run(run_id="r", experiment_id="e", spec=_spec(workers=1))
    relay = MetricsRelay(q, tracker, poll=0.01)
    relay.start()
    q.put((1, {"train/loss": 1.0}))
    q.put((2, {"train/loss": 0.5}))
    q.put((3, {"train/loss": 0.25}))  # may still be queued when stop() is called
    history = relay.stop()
    assert history == [{"train/loss": 1.0}, {"train/loss": 0.5}, {"train/loss": 0.25}]
    assert [s for s, _ in tracker.current.metrics] == [1, 2, 3]


def test_relay_survives_a_tracker_error_and_reraises_it_on_stop():
    import queue

    from mlops.adapters.ray_launcher import MetricsRelay

    class Broken(InMemoryTracker):
        def log_metrics(self, metrics, *, step):
            raise ConnectionError("mlflow down")

    q: queue.Queue = queue.Queue()
    relay = MetricsRelay(q, Broken(), poll=0.01)
    relay.start()
    q.put((1, {"train/loss": 1.0}))
    with pytest.raises(ConnectionError, match="mlflow down"):
        relay.stop()


# --- checkpoint + resume through Ray Train's own checkpoints (per-rank RNG state) ---


class _FailOnce(torch.utils.data.Dataset):
    """Raises on access number ``after + 1`` (counted per worker process) unless ``marker``
    exists; creates it when raising. So the first run dies, any later run is normal."""

    def __init__(self, inner, after: int, marker: str) -> None:
        self.inner, self.after, self.marker, self.seen = inner, after, marker, 0

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        self.seen += 1
        if self.seen > self.after and not Path(self.marker).exists():
            Path(self.marker).touch()
            raise RuntimeError("node preempted")
        return self.inner[i]


class FailOnceBlobs(SyntheticBlobs):
    def __init__(self, after: int, marker: str) -> None:
        super().__init__(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)
        self.after, self.marker = after, marker

    def load(self, dataset, preprocessing):
        from mlops.ports.data import DataSplits

        splits = super().load(dataset, preprocessing)
        return DataSplits(train=_FailOnce(splits.train, self.after, self.marker), val=splits.val)


def _resumable_spec():
    base = _spec(workers=2, **{"training.epochs": 4})
    return load_experiment_dict(
        {
            **base.model_dump(),
            "augmentation": [{"name": "rand_flip", "params": {"prob": 0.5, "spatial_axis": [0, 1, 2]}}],
        }
    )


def test_ray_run_resumes_from_its_checkpoint_bit_exactly(tmp_path):
    from tests.test_environment import FixedProbe

    spec = _resumable_spec()
    ref_store = InMemoryArtifactStore()
    crit = BestCriterion(metric="val/loss", mode="max")  # best = an epoch before the interruption
    ref = RayLauncher(storage_path=tmp_path / "ref").launch(
        spec, data=_data(), tracker=InMemoryTracker(), artifacts=ref_store, keep_best=crit
    )

    launcher, store, tracker = RayLauncher(storage_path=tmp_path / "ray"), InMemoryArtifactStore(), InMemoryTracker()
    # 2 ranks x 2 samples per epoch: access 5 is the first item of epoch 3 on each rank
    data = FailOnceBlobs(after=4, marker=str(tmp_path / "preempted"))
    with pytest.raises(TrainingFailedError):
        launcher.launch(
            spec,
            data=data,
            tracker=tracker,
            artifacts=store,
            checkpoint_every=1,
            probe=FixedProbe(commit_ok=True),
            keep_best=crit,
        )
    failed = load_bundle(store, ref.run_id).manifest
    assert failed.status == "FAILED" and len(failed.history) == 2
    assert failed.environment is not None  # recorded at the start now, so a failure keeps it

    result = launcher.launch(
        spec, data=data, tracker=tracker, artifacts=store, resume=True, probe=FixedProbe(commit_ok=False)
    )
    assert result.run_id == ref.run_id
    assert result.history == ref.history
    a, b = load_bundle(ref_store, ref.run_id), load_bundle(store, result.run_id)
    assert b.manifest.status == "FINISHED"
    for k, v in a.model_state.items():
        assert torch.equal(v, b.model_state[k]), k
    # the criterion came back from launch.json, the best weights from Ray's checkpoint
    assert b.manifest.best == a.manifest.best and b.manifest.best.epoch is not None
    for k, v in a.best_state.items():
        assert torch.equal(v, b.best_state[k]), k
    assert [r.epoch for r in b.manifest.resumes] == [2]
    assert b.manifest.environment.checks["code.commit"].status == "ok"  # first segment
    assert b.manifest.resumes[0].environment.checks["code.commit"].status == "mismatch"
    assert len(tracker.runs) == 1 and [s for s, _ in tracker.runs[0].metrics] == [1, 2, 3, 4]


def test_ray_checkpointing_records_how_to_resume(tmp_path):
    from mlops.application.bundle import LAUNCH, load_launch

    store = InMemoryArtifactStore()
    result = RayLauncher(storage_path=tmp_path / "ray").launch(
        _spec(workers=1), data=_data(), tracker=InMemoryTracker(), artifacts=store, checkpoint_every=1
    )
    assert LAUNCH in store.list(result.run_id)
    launch = load_launch(store, result.run_id)
    assert launch.launcher == "ray" and launch.attempt == 1 and launch.checkpoint_every == 1
    assert launch.storage_path == str(tmp_path / "ray")


def _ray_run_dir(storage, run_id):
    return Path(storage) / f"mlops-{run_id[:12]}-a1"


def test_finished_ray_run_drops_its_ray_storage(tmp_path):
    # The final weights are in model.pt; Ray's copies (checkpoints + final) are dead weight.
    store = InMemoryArtifactStore()
    result = RayLauncher(storage_path=tmp_path / "ray").launch(
        _spec(workers=1, **{"training.epochs": 3}),
        data=_data(),
        tracker=InMemoryTracker(),
        artifacts=store,
        checkpoint_every=1,
    )
    assert not _ray_run_dir(tmp_path / "ray", result.run_id).exists()
    assert load_bundle(store, result.run_id).manifest.status == "FINISHED"


def test_ray_keeps_only_the_latest_checkpoint(tmp_path):
    result = RayLauncher(storage_path=tmp_path / "ray").launch(
        _spec(workers=1, **{"training.epochs": 3}),
        data=_data(),
        tracker=InMemoryTracker(),
        artifacts=InMemoryArtifactStore(),
        checkpoint_every=1,
        keep_checkpoint=True,
    )
    assert len([p for p in _ray_run_dir(tmp_path / "ray", result.run_id).glob("checkpoint_*") if p.is_dir()]) == 1


def test_ray_resume_with_another_keep_best_is_refused(tmp_path):
    from mlops.application.bundle import LaunchRecord, save_launch

    store, spec = InMemoryArtifactStore(), _spec(workers=1)
    run_id = compute_run_id(compute_experiment_id(spec), spec.resources, 1)
    save_launch(
        store,
        run_id,
        LaunchRecord(
            launcher="ray", attempt=1, checkpoint_every=1, storage_path=str(tmp_path / "ray"), keep_best="val/loss:min"
        ),
    )
    with pytest.raises(RunnerError, match="val/loss:min"):
        RayLauncher(storage_path=tmp_path / "ray").launch(
            spec,
            data=_data(),
            tracker=InMemoryTracker(),
            artifacts=store,
            resume=True,
            keep_best=BestCriterion(metric="val/loss", mode="max"),
        )


def test_ray_resume_without_a_checkpoint_is_refused_before_starting(tmp_path):
    tracker = InMemoryTracker()
    with pytest.raises(RunnerError, match="no Ray checkpoint"):
        RayLauncher(storage_path=tmp_path / "ray").launch(
            _spec(workers=1), data=_data(), tracker=tracker, artifacts=InMemoryArtifactStore(), resume=True
        )
    assert tracker.runs == []


def test_ray_resume_of_a_finished_run_is_refused(tmp_path):
    launcher, store = RayLauncher(storage_path=tmp_path / "ray"), InMemoryArtifactStore()
    launcher.launch(_spec(workers=1), data=_data(), tracker=InMemoryTracker(), artifacts=store, checkpoint_every=1)
    tracker = InMemoryTracker()
    with pytest.raises(RunnerError, match="already FINISHED"):
        launcher.launch(_spec(workers=1), data=_data(), tracker=tracker, artifacts=store, resume=True)
    assert tracker.runs == []


@pytest.mark.parametrize("kw", [{"checkpoint_every": 1}, {"resume": True}])
def test_ray_checkpoint_and_resume_need_an_artifact_store(launcher, kw):
    with pytest.raises(RunnerError, match="artifact store"):
        launcher.launch(_spec(workers=1), data=_data(), tracker=InMemoryTracker(), **kw)


def test_relay_keeps_history_only_items_out_of_the_tracker():
    import queue

    from mlops.adapters.ray_launcher import MetricsRelay

    q: queue.Queue = queue.Queue()
    tracker = InMemoryTracker()
    tracker.start_run(run_id="r", experiment_id="e", spec=_spec(workers=1))
    relay = MetricsRelay(q, tracker, poll=0.01)
    relay.start()
    q.put((1, {"train/loss": 1.0}, False))  # from the checkpoint: already in the tracker
    q.put(("start", {"environment": None, "resumes": []}))
    q.put((2, {"train/loss": 0.5}))
    assert relay.stop() == [{"train/loss": 1.0}, {"train/loss": 0.5}]
    assert [s for s, _ in tracker.current.metrics] == [2]
    assert relay.start_info == {"environment": None, "resumes": []}


# --- reproduce a multi-worker run (roadmap 17): replayed by Ray with the same world size ---


def test_two_worker_ray_run_is_reproduced_bit_exactly_through_the_cli(tmp_path, capsys):
    import json

    import yaml

    from mlops.cli import main

    spec = _spec(workers=2, **{"training.epochs": 2})
    path = tmp_path / "two_workers.yaml"
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json")))
    art = str(tmp_path / "artifacts")
    run = ["run", str(path), "--launcher", "ray", "--ray-storage", str(tmp_path / "ray"), "--artifacts", art]
    assert main([*run, "--tracker", "inmemory", "--keep-best", "val/loss:min", "--json"]) == 0
    run_id = json.loads(capsys.readouterr().out)["run_id"]

    # No --launcher: the bundle's manifest says ray. atol 0 = torch.equal on model.pt and best.pt.
    assert main(["reproduce", run_id, "--artifacts", art, "--tracker", "inmemory", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["launcher"] == "ray"
    assert report["checks"] == {"run_id": True, "history": True, "weights": True}
    assert (tmp_path / "artifacts" / run_id / "best.pt").exists()  # so the weights check covered best.pt too

    # The same bundle cannot be replayed with one process.
    assert main(["reproduce", run_id, "--artifacts", art, "--tracker", "inmemory", "--launcher", "local"]) == 2
    assert "2 workers" in capsys.readouterr().err
