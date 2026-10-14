"""Launcher backed by Ray Train (``TorchTrainer``).

Every rank runs ``mlops.application.runner.train_worker``; Ray supplies the
process group, the device and DDP (``prepare_model``). The Tracker and the
ArtifactStore stay on the driver: rank 0 returns its history and ships the
final ``state_dict`` back as a Ray checkpoint, and the driver replays the
history into the tracker and writes the bundle. Metrics therefore reach the
tracker after ``fit()`` returns, not live (per-epoch ``ray.train.report`` is
still called so the Ray dashboard shows progress).

On Kubernetes the same class runs inside a KubeRay ``RayJob`` whose entrypoint
is ``mlops run --launcher ray``; see ``mlops.adapters.kuberay``.

Requires the ``distributed`` extra; the core never imports ray.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import ray
import ray.train
import ray.train.torch
import torch
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

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

    def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
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

        trainer = TorchTrainer(
            _train_loop,
            train_loop_config={
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

        # Wrong data must not even start a run (as with the local runner); ranks re-check what they read.
        verify_data(spec, data)
        tracker.start_run(run_id=run_id, experiment_id=experiment_id, spec=spec)
        try:
            result = trainer.fit()
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
            tracker.end_run("FAILED")
            if artifacts is not None:
                save_bundle(artifacts, spec=spec, manifest=_manifest("FAILED"), model_state=None)
            raise

        for epoch, metrics in enumerate(history, start=1):
            tracker.log_metrics(metrics, step=epoch)
        tracker.end_run("FINISHED")
        if artifacts is not None:
            save_bundle(artifacts, spec=spec, manifest=_manifest("FINISHED"), model_state=state)
        return RunResult(
            experiment_id=experiment_id, run_id=run_id, device=dev, history=history,
            environment=environment, data_snapshot=data_snapshot,
        )

