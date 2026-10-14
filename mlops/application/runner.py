"""Use case: execute one run of an ExperimentSpec.

Flow: ids -> seed -> build (model, loss, augmentations, optimizer) via the
registries -> data via the DataProvider port -> epoch loop -> Tracker port.

The runner imports no concrete component. Callers are responsible for
``mlops.registry.discover("experiments")`` (or an equivalent) before calling
``run_experiment`` so the names in the spec resolve.

``train_worker`` is the per-process loop: it takes a ``Worker(rank, world_size)``,
shards the data by rank and uses ``batch_size // world_size`` per process
(``training.batch_size`` is the global batch). ``run_experiment`` runs it in
this process with world_size 1; a distributed launcher (Ray) runs it on every
rank and only rank 0's callbacks reach the tracker.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, DistributedSampler

from mlops.application.bundle import save_bundle
from mlops.domain.environment import EnvironmentRecord, record_environment
from mlops.domain.experiment import ExperimentSpec, Resources, Training
from mlops.domain.identity import compute_experiment_id, compute_run_id
from mlops.domain.manifest import RunManifest
from mlops.ports.artifacts import ArtifactStore
from mlops.ports.data import DataProvider, Sample
from mlops.ports.environment import EnvironmentProbe
from mlops.ports.tracking import RunStatus, Tracker
from mlops.registry import AUGMENTATIONS, LOSSES, MODELS

# An augmentation is a callable applied to one on-device batch: (inputs, targets) -> (inputs, targets).
Augment = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]

_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


class RunnerError(Exception):
    pass


@dataclass(frozen=True)
class RunResult:
    experiment_id: str
    run_id: str
    device: str
    # One entry per epoch, in order: {"train/loss": ..., "val/loss": ...}.
    history: list[dict[str, float]] = field(default_factory=list)
    environment: EnvironmentRecord | None = None

    @property
    def final_metrics(self) -> dict[str, float]:
        return self.history[-1] if self.history else {}


@dataclass(frozen=True)
class Worker:
    """Where this process sits in the (possibly distributed) run."""

    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} outside world_size {self.world_size}")

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


# Hooks a launcher can plug into the loop. ``on_epoch(epoch, metrics)`` fires on every rank
# with the all-reduced metrics; the caller decides that only the leader forwards them.
EpochHook = Callable[[int, dict[str, float]], None]


# -- building blocks -----------------------------------------------------------


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the loop can touch and, by default, pin torch to deterministic kernels.

    Seeding alone is not enough: multi-threaded scatter/index_add on CPU and several
    cuDNN/cuBLAS paths sum in a data-race-dependent order, so the same seed can give
    weights that differ in the last bits (seen on SwinUNETR's relative_position_bias_table).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed % 2**32)
    if deterministic:
        # cuBLAS needs this before its first CUDA call, or deterministic mode raises.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(resources: Resources, override: str | None = None) -> torch.device:
    """``resources.gpu > 0`` means CUDA is required, not merely preferred."""
    if override is not None:
        return torch.device(override)
    if resources.gpu > 0:
        if not torch.cuda.is_available():
            raise RunnerError(
                f"spec requests gpu={resources.gpu} but CUDA is not available; "
                "pass device='cpu' to override"
            )
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(training: Training, params: Iterable[Tensor]) -> torch.optim.Optimizer:
    try:
        cls = _OPTIMIZERS[training.optimizer.lower()]
    except KeyError:
        known = ", ".join(sorted(_OPTIMIZERS))
        raise RunnerError(f"unknown optimizer {training.optimizer!r}; known: {known}") from None
    return cls(params, lr=training.learning_rate)


def _loader(
    dataset, *, batch_size: int, shuffle: bool, seed: int, worker: Worker
) -> DataLoader[Sample]:
    if worker.world_size == 1:
        g = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=False)
    sampler = DistributedSampler(
        dataset, num_replicas=worker.world_size, rank=worker.rank, shuffle=shuffle, seed=seed
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def _set_epoch(loader: DataLoader[Sample], epoch: int) -> None:
    # DistributedSampler reshuffles per epoch only if told which epoch it is.
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)


def _mean_across_ranks(total: float, count: int, device: torch.device) -> float:
    """Sample-weighted mean over every rank when a process group exists, else local."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t)
        total, count = float(t[0]), int(t[1])
    return total / max(count, 1)


def _train_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[Sample],
    augments: list[Augment],
    device: torch.device,
) -> float:
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for aug in augments:
            x, y = aug(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * x.shape[0]
        count += x.shape[0]
    return _mean_across_ranks(total, count, device)


@torch.no_grad()
def _eval_epoch(
    model: nn.Module, loss_fn: nn.Module, loader: DataLoader[Sample], device: torch.device
) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        total += float(loss) * x.shape[0]
        count += x.shape[0]
    return _mean_across_ranks(total, count, device)


# -- per-worker loop -----------------------------------------------------------


def train_worker(
    spec: ExperimentSpec,
    *,
    data: DataProvider,
    device: torch.device,
    worker: Worker = Worker(),
    wrap_model: Callable[[nn.Module], nn.Module] | None = None,
    on_start: Callable[[], None] | None = None,
    on_epoch: EpochHook | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Run the epoch loop for one rank. Returns the (unwrapped) model and the history.

    ``wrap_model`` is where a launcher injects DDP (the returned model is the
    original so its ``state_dict`` has no ``module.`` prefix). ``on_start`` fires
    once everything is built and data is loaded, right before the first epoch.
    """
    if spec.resources.workers != worker.world_size:
        raise RunnerError(
            f"spec says resources.workers={spec.resources.workers} but the launcher "
            f"started world_size={worker.world_size}"
        )

    # Everything that touches RNG happens after this line. Every rank builds the
    # identical model; DDP would broadcast rank 0's weights anyway.
    seed_everything(spec.training.seed, deterministic=spec.training.deterministic)

    model: nn.Module = MODELS.build(spec.model.name, spec.model.params).to(device)
    loss_fn = LOSSES.build(spec.loss.name, spec.loss.params)
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(device)
    augments: list[Augment] = [AUGMENTATIONS.build(a.name, a.params) for a in spec.augmentation]
    optimizer = build_optimizer(spec.training, model.parameters())
    trained = wrap_model(model) if wrap_model is not None else model

    splits = data.load(spec.dataset, spec.preprocessing)
    bs, seed = spec.per_worker_batch_size(), spec.training.seed
    train_loader = _loader(splits.train, batch_size=bs, shuffle=True, seed=seed, worker=worker)
    val_loader = (
        _loader(splits.val, batch_size=bs, shuffle=False, seed=seed, worker=worker)
        if splits.val is not None
        else None
    )
    if worker.rank:
        # Ranks must not draw the same augmentation coin flips; rank 0 keeps the plain seed
        # so a world_size-1 distributed run reproduces a local run exactly.
        torch.manual_seed(seed + worker.rank)

    if on_start is not None:
        on_start()
    history: list[dict[str, float]] = []
    for epoch in range(1, spec.training.epochs + 1):
        _set_epoch(train_loader, epoch)
        metrics = {"train/loss": _train_epoch(trained, loss_fn, optimizer, train_loader, augments, device)}
        if val_loader is not None:
            metrics["val/loss"] = _eval_epoch(trained, loss_fn, val_loader, device)
        history.append(metrics)
        if on_epoch is not None:
            on_epoch(epoch, metrics)
    return model, history


# -- use case ------------------------------------------------------------------


def run_experiment(
    spec: ExperimentSpec,
    *,
    data: DataProvider,
    tracker: Tracker,
    artifacts: ArtifactStore | None = None,
    attempt: int = 1,
    device: str | None = None,
    probe: EnvironmentProbe | None = None,
) -> RunResult:
    """Train ``spec`` once, in this process. Raises ``UnpinnedInputError`` if the spec is not fully pinned.

    With ``artifacts``, the run bundle (spec, manifest, final checkpoint) is
    written on FINISHED; on FAILED the spec and manifest are still written so
    the failure is reproducible. With ``probe``, the actual environment and its
    checks against the spec's pins go into the manifest (recorded, not enforced).
    Requires ``resources.workers == 1``; use a distributed launcher otherwise.
    """
    experiment_id = compute_experiment_id(spec)
    run_id = compute_run_id(experiment_id, spec.resources, attempt)
    dev = resolve_device(spec.resources, device)
    environment = record_environment(spec, probe.capture(spec)) if probe is not None else None

    history: list[dict[str, float]] = []
    started = False

    def _manifest(status: RunStatus) -> RunManifest:
        return RunManifest(
            experiment_id=experiment_id,
            run_id=run_id,
            attempt=attempt,
            status=status,
            device=str(dev),
            history=history,
            environment=environment,
        )

    def on_start() -> None:
        nonlocal started
        tracker.start_run(run_id=run_id, experiment_id=experiment_id, spec=spec)
        started = True

    def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
        tracker.log_metrics(metrics, step=epoch)
        history.append(metrics)

    try:
        model, _ = train_worker(spec, data=data, device=dev, on_start=on_start, on_epoch=on_epoch)
    except BaseException:
        if started:
            tracker.end_run("FAILED")
            if artifacts is not None:
                save_bundle(artifacts, spec=spec, manifest=_manifest("FAILED"), model_state=None)
        raise
    tracker.end_run("FINISHED")
    if artifacts is not None:
        save_bundle(artifacts, spec=spec, manifest=_manifest("FINISHED"), model_state=model.state_dict())

    return RunResult(
        experiment_id=experiment_id, run_id=run_id, device=str(dev), history=history, environment=environment
    )
