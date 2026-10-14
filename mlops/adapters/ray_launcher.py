"""Launcher backed by Ray Train (``TorchTrainer``).

Every rank runs ``mlops.application.runner.train_worker``; Ray supplies the
process group, the device and DDP (``prepare_model``). The Tracker and the
ArtifactStore stay on the driver. Metrics are live: rank 0 puts each epoch on a
``ray.util.queue.Queue`` and a ``MetricsRelay`` thread on the driver logs it into
the tracker while ``fit()`` runs (so a run that dies in epoch 7 still shows
epochs 1-6, in MLflow and in its FAILED manifest). Rank 0 returns its history and
ships the final ``state_dict`` back as a Ray checkpoint; the driver writes the
bundle. Per-epoch ``ray.train.report`` is still called for Ray's own dashboard.

Checkpoint + resume use Ray Train's own checkpoints: every ``checkpoint_every`` epochs each
rank reports a directory with its RNG state (rank 0 adds weights, optimizer, history), which
Ray merges and persists under ``storage_path/<run name>``. ``resume=True`` launches the same
run name on the same storage; Ray hands each rank the latest checkpoint via
``get_checkpoint()`` and the run continues bit-exactly. ``launch.json`` in the artifact store
records the storage path so ``mlops resume`` can find it. On KubeRay that storage must be
shared by all pods (a PVC or object storage).

On Kubernetes the same class runs inside a KubeRay ``RayJob`` whose entrypoint
is ``mlops run --launcher ray``; see ``mlops.adapters.kuberay``.

Requires the ``distributed`` extra; the core never imports ray.
"""

from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import ray
import ray.train
import ray.train.torch
import torch
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.util.queue import Queue

from mlops.application.bundle import (
    MANIFEST,
    LaunchRecord,
    TrainState,
    load_launch,
    save_bundle,
    save_launch,
    save_spec,
)
from mlops.application.runner import RunnerError, RunResult, Worker, train_worker, verify_data
from mlops.domain.environment import EnvCheck, EnvironmentRecord, record_environment
from mlops.domain.experiment import ExperimentSpec
from mlops.domain.identity import compute_experiment_id, compute_run_id
from mlops.domain.manifest import ResumePoint, RunManifest
from mlops.ports.artifacts import ArtifactError, ArtifactStore
from mlops.ports.data import DataProvider
from mlops.ports.environment import EnvironmentProbe
from mlops.ports.tracking import RunStatus, Tracker
from mlops.registry import discover

_STATE_FILE = "model.pt"  # final weights, rank 0
# Mid-run checkpoints (checkpoint_every): Ray merges every rank's files into one directory.
_TRAIN_STATE = "state.pt"  # rank 0: weights, optimizer, history, environment, resume points
_RNG_FILE = "rng-{rank}.pt"  # every rank: its own RNG states (ranks draw different augmentations)


def _train_loop(config: dict[str, Any]) -> dict[str, Any] | None:
    """Runs on every Ray worker. Only rank 0 returns something."""
    for p in config["sys_path"]:
        if p not in sys.path:
            sys.path.append(p)
    discover(config["experiments"])

    spec = ExperimentSpec.model_validate_json(config["spec"])
    ctx = ray.train.get_context()
    worker = Worker(rank=ctx.get_world_rank(), world_size=ctx.get_world_size())
    device = ray.train.torch.get_device()
    metrics_queue = config["metrics_queue"]

    # Captured on the training rank, not the driver: its GPU, threads and image are what count.
    # At the start, so a run that fails (or is checkpointed) still carries it.
    segment_env = None
    if worker.is_leader and config["probe"] is not None:
        segment_env = record_environment(spec, config["probe"].capture(spec))
    resume_from, environment, resumes = None, segment_env, []
    if config["resume"]:
        resume_from, environment, earlier = _restore(worker)
        resumes = [*earlier, ResumePoint(epoch=resume_from.epoch, environment=segment_env)]
    start_info = {
        "environment": environment.model_dump_json() if environment is not None else None,
        "resumes": [r.model_dump_json() for r in resumes],
    }
    if worker.is_leader:
        metrics_queue.put(("start", start_info))
        for epoch, metrics in enumerate(resume_from.history if resume_from else [], start=1):
            metrics_queue.put((epoch, metrics, False))  # driver's history only; already in the tracker

    data_check: list[EnvCheck] = []

    def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
        if worker.is_leader:  # metrics are already all-reduced; one copy reaches the driver
            metrics_queue.put((epoch, metrics))
        # report() is a barrier: every rank calls it the same number of times.
        ray.train.report({"epoch": epoch, **metrics})

    def on_checkpoint(state: TrainState) -> None:
        d = tempfile.mkdtemp(prefix="mlops-ckpt-")
        torch.save(state.rng, os.path.join(d, _RNG_FILE.format(rank=worker.rank)))
        if worker.is_leader:
            torch.save({
                "epoch": state.epoch, "history": state.history, "world_size": worker.world_size,
                "model": dict(state.model_state), "optimizer": state.optimizer_state, **start_info,
            }, os.path.join(d, _TRAIN_STATE))
        ray.train.report({"epoch": state.epoch, "checkpoint": 1.0}, checkpoint=Checkpoint.from_directory(d))

    model, history = train_worker(
        spec,
        data=config["data"],
        device=device,
        worker=worker,
        wrap_model=ray.train.torch.prepare_model,
        on_epoch=on_epoch,
        on_data=data_check.append,
        resume_from=resume_from,
        checkpoint_every=config["checkpoint_every"],
        on_checkpoint=on_checkpoint,
    )

    checkpoint = None
    if worker.is_leader:
        d = tempfile.mkdtemp(prefix="mlops-ckpt-")
        torch.save(model.state_dict(), os.path.join(d, _STATE_FILE))
        checkpoint = Checkpoint.from_directory(d)
    ray.train.report({"epoch": len(history), "done": 1.0}, checkpoint=checkpoint)
    if not worker.is_leader:
        return None
    return {
        "history": history,
        "device": str(device),
        **start_info,
        "data_snapshot": data_check[0].model_dump_json() if data_check else None,
    }


def _restore(worker: Worker) -> tuple[TrainState, EnvironmentRecord | None, list[ResumePoint]]:
    """This rank's TrainState from Ray's latest checkpoint of the run (same name + storage)."""
    ckpt = ray.train.get_checkpoint()
    if ckpt is None:
        raise RunnerError("no Ray checkpoint for this run to resume from")
    with ckpt.as_directory() as d:
        state_file, rng_file = Path(d) / _TRAIN_STATE, Path(d) / _RNG_FILE.format(rank=worker.rank)
        if not state_file.exists():
            raise RunnerError("the latest Ray checkpoint holds final weights only: the run already finished")
        st = torch.load(state_file, map_location="cpu", weights_only=True)
        if st["world_size"] != worker.world_size or not rng_file.exists():
            raise RunnerError(
                f"checkpoint was taken with {st['world_size']} workers, this run has {worker.world_size}"
            )
        rng = torch.load(rng_file, map_location="cpu", weights_only=True)
    state = TrainState(epoch=st["epoch"], history=st["history"], model_state=st["model"],
                       optimizer_state=st["optimizer"], rng=rng)
    environment = EnvironmentRecord.model_validate_json(st["environment"]) if st["environment"] else None
    return state, environment, [ResumePoint.model_validate_json(r) for r in st["resumes"]]


class MetricsRelay:
    """Driver-side thread: ``(epoch, metrics)`` items from a queue -> ``tracker.log_metrics``.

    ``(epoch, metrics, False)`` only joins ``history`` (epochs restored from a checkpoint are
    already in the tracker); ``("start", info)`` is kept as ``start_info`` (environment and
    resume points, so a run that fails still records them).

    The tracker is only touched from this thread while it runs, and from the caller after
    ``stop()``. A tracker error stops logging (training goes on) and is re-raised by ``stop()``.
    """

    def __init__(self, source: Any, tracker: Tracker, *, poll: float = 0.2) -> None:
        self.source, self.tracker, self.poll = source, tracker, poll
        self.history: list[dict[str, float]] = []
        self.start_info: dict[str, Any] | None = None
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="mlops-ray-metrics", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[dict[str, float]]:
        """Stop, log whatever is still queued, return every epoch received (in order)."""
        self._stop.set()
        self._thread.join()
        while True:
            try:
                self._handle(self.source.get_nowait())
            except queue.Empty:  # ray.util.queue.Empty subclasses it
                break
        if self._error is not None:
            raise self._error
        return self.history

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.source.get(timeout=self.poll)
            except queue.Empty:
                continue
            self._handle(item)

    def _handle(self, item: tuple[Any, ...]) -> None:
        if item[0] == "start":
            self.start_info = item[1]
            return
        epoch, metrics, *rest = item
        self.history.append(metrics)
        if self._error is None and (not rest or rest[0]):
            try:
                self.tracker.log_metrics(metrics, step=epoch)
            except BaseException as e:  # noqa: BLE001 - surfaced by stop()
                self._error = e


class RayLauncher:
    def __init__(
        self,
        *,
        address: str | None = None,
        storage_path: str | os.PathLike[str] | None = None,
        experiments: str = "experiments",
    ) -> None:
        # address=None connects to a running cluster (RAY_ADDRESS) or starts a local one.
        self.address = address
        self.storage_path = Path(storage_path) if storage_path else None
        self.experiments = experiments

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
    ) -> RunResult:
        experiment_id = compute_experiment_id(spec)
        run_id = compute_run_id(experiment_id, spec.resources, attempt)
        if checkpoint_every is not None and checkpoint_every < 1:
            raise RunnerError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
        if (checkpoint_every or resume) and artifacts is None:
            raise RunnerError("checkpoint_every / resume need an artifact store (spec.json + launch.json go there)")
        storage = str(self.storage_path or Path(tempfile.gettempdir()) / "mlops-ray")
        name = f"mlops-{run_id[:12]}-a{attempt}"
        if resume:
            assert artifacts is not None
            _refuse_if_finished(artifacts, run_id)
            if "://" not in storage and not any((Path(storage) / name).glob("checkpoint_*")):
                raise RunnerError(f"no Ray checkpoint for run {run_id} under {Path(storage) / name}: nothing to resume")
            if checkpoint_every is None:
                try:
                    checkpoint_every = load_launch(artifacts, run_id).checkpoint_every
                except ArtifactError:
                    pass
        res = spec.resources
        use_gpu = res.gpu > 0 and device != "cpu"

        if not ray.is_initialized():
            ray.init(address=self.address, ignore_reinit_error=True)
        if use_gpu:
            have = ray.cluster_resources().get("GPU", 0)
            if have < res.workers * res.gpu:
                raise RunnerError(
                    f"spec needs {res.workers} workers x {res.gpu} GPU but the Ray cluster has "
                    f"{have:g} GPU(s); pass device='cpu' to override"
                )

        # Wrong data must not even start a run (as with the local runner); ranks re-check what they read.
        verify_data(spec, data)
        if checkpoint_every:
            assert artifacts is not None
            # A killed driver never reaches save_bundle; `mlops resume` needs these to come back.
            save_spec(artifacts, run_id, spec)
            save_launch(artifacts, run_id, LaunchRecord(
                launcher="ray", attempt=attempt, checkpoint_every=checkpoint_every, storage_path=storage,
            ))
        # num_cpus=0: schedulable anywhere (a KubeRay head has num-cpus=0) and never competes
        # with the training workers for CPUs.
        metrics_queue = Queue(actor_options={"num_cpus": 0})
        trainer = TorchTrainer(
            _train_loop,
            train_loop_config={
                "metrics_queue": metrics_queue,
                "checkpoint_every": checkpoint_every,
                "resume": resume,
                "spec": spec.model_dump_json(),
                "data": data,
                "probe": probe,
                "experiments": self.experiments,
                "sys_path": [os.getcwd(), *sys.path],
            },
            scaling_config=ScalingConfig(
                num_workers=res.workers,
                use_gpu=use_gpu,
                resources_per_worker={"CPU": res.cpu, **({"GPU": res.gpu} if use_gpu else {})},
            ),
            # Same name + storage on resume: that is how Ray hands back the latest checkpoint.
            run_config=RunConfig(name=name, storage_path=storage),
        )

        history: list[dict[str, float]] = []
        dev = "cuda" if use_gpu else "cpu"
        environment: EnvironmentRecord | None = None
        resumes: list[ResumePoint] = []
        data_snapshot: EnvCheck | None = None

        def _manifest(status: RunStatus) -> RunManifest:
            return RunManifest(
                experiment_id=experiment_id, run_id=run_id, attempt=attempt,
                status=status, device=dev, history=history, environment=environment,
                data_snapshot=data_snapshot, resumes=resumes,
            )

        def _take_start_info(info: dict[str, Any] | None) -> None:
            nonlocal environment, resumes
            if info is not None:
                environment = EnvironmentRecord.model_validate_json(info["environment"]) if info["environment"] else None
                resumes = [ResumePoint.model_validate_json(r) for r in info["resumes"]]

        tracker.start_run(run_id=run_id, experiment_id=experiment_id, spec=spec, resume=resume)
        relay = MetricsRelay(metrics_queue, tracker)
        relay.start()
        try:
            result = trainer.fit()
            relay.stop()
            leader = result.return_value
            if not leader or result.checkpoint is None:
                raise RunnerError("Ray run finished without rank 0 returning a history/checkpoint")
            history.extend(leader["history"])
            dev = leader["device"]
            _take_start_info(leader)
            if leader["data_snapshot"] is not None:
                data_snapshot = EnvCheck.model_validate_json(leader["data_snapshot"])
            with result.checkpoint.as_directory() as d:
                state = torch.load(Path(d) / _STATE_FILE, map_location="cpu", weights_only=True)
        except BaseException:
            try:  # keep every epoch that did finish, in the tracker and the FAILED manifest
                history[:] = relay.stop()
            except BaseException:  # noqa: BLE001 - the training failure is the error to report
                history[:] = relay.history
            _take_start_info(relay.start_info)
            tracker.end_run("FAILED")
            if artifacts is not None:
                save_bundle(artifacts, spec=spec, manifest=_manifest("FAILED"), model_state=None)
            raise
        finally:
            metrics_queue.shutdown()
        tracker.end_run("FINISHED")
        if artifacts is not None:
            save_bundle(artifacts, spec=spec, manifest=_manifest("FINISHED"), model_state=state)
        return RunResult(
            experiment_id=experiment_id, run_id=run_id, device=dev, history=history,
            environment=environment, data_snapshot=data_snapshot, resumes=resumes,
        )


def _refuse_if_finished(store: ArtifactStore, run_id: str) -> None:
    try:
        status = RunManifest.model_validate_json(store.get(run_id, MANIFEST)).status
    except ArtifactError:
        return
    if status == "FINISHED":
        raise RunnerError(f"run {run_id} already FINISHED; nothing to resume")

