"""Launcher backed by Ray Train (``TorchTrainer``).

Every rank runs ``mlops.application.runner.train_worker``; Ray supplies the
process group, the device and DDP (``prepare_model``). The Tracker and the
ArtifactStore stay on the driver. Metrics are live: rank 0 puts each epoch on a
``ray.util.queue.Queue`` and a ``MetricsRelay`` thread on the driver logs it into
the tracker while ``fit()`` runs (so a run that dies in epoch 7 still shows
epochs 1-6, in MLflow and in its FAILED manifest). Rank 0 returns its history and
ships the final ``state_dict`` back as a Ray checkpoint; the driver writes the
bundle. Per-epoch ``ray.train.report`` is still called for Ray's own dashboard.

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

from mlops.application.bundle import save_bundle
from mlops.application.runner import RunnerError, RunResult, Worker, train_worker, verify_data
from mlops.domain.environment import EnvCheck, EnvironmentRecord, record_environment
from mlops.domain.experiment import ExperimentSpec
from mlops.domain.identity import compute_experiment_id, compute_run_id
from mlops.domain.manifest import RunManifest
from mlops.ports.artifacts import ArtifactStore
from mlops.ports.data import DataProvider
from mlops.ports.environment import EnvironmentProbe
from mlops.ports.tracking import RunStatus, Tracker
from mlops.registry import discover

_STATE_FILE = "model.pt"


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

    data_check: list[EnvCheck] = []

    metrics_queue = config["metrics_queue"]

    def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
        if worker.is_leader:  # metrics are already all-reduced; one copy reaches the driver
            metrics_queue.put((epoch, metrics))
        # report() is a barrier: every rank calls it the same number of times.
        ray.train.report({"epoch": epoch, **metrics})

    model, history = train_worker(
        spec,
        data=config["data"],
        device=device,
        worker=worker,
        wrap_model=ray.train.torch.prepare_model,
        on_epoch=on_epoch,
        on_data=data_check.append,
    )

    checkpoint = None
    environment = None
    if worker.is_leader:
        # Captured on the training rank, not the driver: its GPU, threads and image are what count.
        probe = config["probe"]
        if probe is not None:
            environment = record_environment(spec, probe.capture(spec)).model_dump_json()
        d = tempfile.mkdtemp(prefix="mlops-ckpt-")
        torch.save(model.state_dict(), os.path.join(d, _STATE_FILE))
        checkpoint = Checkpoint.from_directory(d)
    ray.train.report({"epoch": len(history), "done": 1.0}, checkpoint=checkpoint)
    if not worker.is_leader:
        return None
    return {
        "history": history,
        "device": str(device),
        "environment": environment,
        "data_snapshot": data_check[0].model_dump_json() if data_check else None,
    }


class MetricsRelay:
    """Driver-side thread: ``(epoch, metrics)`` items from a queue -> ``tracker.log_metrics``.

    The tracker is only touched from this thread while it runs, and from the caller after
    ``stop()``. A tracker error stops logging (training goes on) and is re-raised by ``stop()``.
    """

    def __init__(self, source: Any, tracker: Tracker, *, poll: float = 0.2) -> None:
        self.source, self.tracker, self.poll = source, tracker, poll
        self.history: list[dict[str, float]] = []
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

    def _handle(self, item: tuple[int, dict[str, float]]) -> None:
        epoch, metrics = item
        self.history.append(metrics)
        if self._error is None:
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
        if checkpoint_every or resume:
            # Needs per-rank RNG state and Ray's own checkpoint/restore.
            raise RunnerError("checkpoint_every / resume are not supported by the Ray launcher yet")
        experiment_id = compute_experiment_id(spec)
        run_id = compute_run_id(experiment_id, spec.resources, attempt)
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
        # num_cpus=0: schedulable anywhere (a KubeRay head has num-cpus=0) and never competes
        # with the training workers for CPUs.
        metrics_queue = Queue(actor_options={"num_cpus": 0})
        trainer = TorchTrainer(
            _train_loop,
            train_loop_config={
                "metrics_queue": metrics_queue,
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
            run_config=RunConfig(
                name=f"mlops-{run_id[:12]}-a{attempt}",
                storage_path=str(self.storage_path or Path(tempfile.gettempdir()) / "mlops-ray"),
            ),
        )

        history: list[dict[str, float]] = []
        dev = "cuda" if use_gpu else "cpu"
        environment: EnvironmentRecord | None = None
        data_snapshot: EnvCheck | None = None

        def _manifest(status: RunStatus) -> RunManifest:
            return RunManifest(
                experiment_id=experiment_id, run_id=run_id, attempt=attempt,
                status=status, device=dev, history=history, environment=environment,
                data_snapshot=data_snapshot,
            )

        tracker.start_run(run_id=run_id, experiment_id=experiment_id, spec=spec)
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
            if leader["environment"] is not None:
                environment = EnvironmentRecord.model_validate_json(leader["environment"])
            if leader["data_snapshot"] is not None:
                data_snapshot = EnvCheck.model_validate_json(leader["data_snapshot"])
            with result.checkpoint.as_directory() as d:
                state = torch.load(Path(d) / _STATE_FILE, map_location="cpu", weights_only=True)
        except BaseException:
            try:  # keep every epoch that did finish, in the tracker and the FAILED manifest
                history[:] = relay.stop()
            except BaseException:  # noqa: BLE001 - the training failure is the error to report
                history[:] = relay.history
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
            environment=environment, data_snapshot=data_snapshot,
        )

