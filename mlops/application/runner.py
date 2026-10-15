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

Checkpoint + resume: with ``checkpoint_every=N`` the full training state (weights,
optimizer, every RNG, history) is stored as ``checkpoint.pt`` after every N-th epoch;
``resume=True`` continues the same run_id from it, bit-exactly, and records the resume
point in the manifest.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from mlops.application.bundle import (
    CHECKPOINT,
    MANIFEST,
    Checkpoint,
    TrainState,
    load_checkpoint,
    save_bundle,
    save_checkpoint,
    save_spec,
)
from mlops.domain.best import BestCriterion, best_epoch, best_record
from mlops.domain.environment import EnvCheck, EnvironmentRecord, check_snapshot, record_environment
from mlops.domain.experiment import ExperimentSpec, Resources, Training
from mlops.domain.identity import compute_experiment_id, compute_run_id
from mlops.domain.manifest import ResumePoint, RunManifest
from mlops.ports.artifacts import ArtifactError, ArtifactStore
from mlops.ports.data import DataError, DataProvider, Sample
from mlops.ports.environment import EnvironmentProbe
from mlops.ports.tracking import RunStatus, Tracker
from mlops.registry import AUGMENTATIONS, INFERERS, LOSSES, METRICS, MODELS

# An augmentation is a callable applied to one on-device batch: (inputs, targets) -> (inputs, targets).
Augment = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
# A validation metric: (logits, targets) -> {key: per-sample values (N,)}, NaN = undefined there.
Metric = Callable[[Tensor, Tensor], dict[str, Tensor]]
# How the model is applied to one validation batch: (model, inputs) -> logits.
Inferer = Callable[[nn.Module, Tensor], Tensor]

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
    # One entry per epoch, in order: {"train/loss": ..., "val/loss": ..., "val/<metric key>": ...}.
    history: list[dict[str, float]] = field(default_factory=list)
    environment: EnvironmentRecord | None = None
    data_snapshot: EnvCheck | None = None
    resumes: list[ResumePoint] = field(default_factory=list)

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


class ShardSampler(Sampler[int]):
    """Rank ``r`` of ``world_size`` takes indices ``r, r + world_size, ...``, in order.

    For validation: unlike ``DistributedSampler`` it never pads, so every case counts exactly
    once in the all-reduced (sum, count) metrics and a 2-rank val score equals a 1-process one.
    Ranks may get different numbers of batches, so evaluation must not run collectives per
    batch (``train_worker`` evaluates the unwrapped model, never the DDP wrapper).
    """

    def __init__(self, n: int, *, rank: int, world_size: int) -> None:
        self.n, self.rank, self.world_size = n, rank, world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.n, self.world_size))

    def __len__(self) -> int:
        return len(range(self.rank, self.n, self.world_size))


def _loader(
    dataset, *, batch_size: int, shuffle: bool, seed: int, worker: Worker
) -> DataLoader[Sample]:
    if worker.world_size == 1:
        g = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=False)
    if not shuffle:  # validation: shard without padding
        shard = ShardSampler(len(dataset), rank=worker.rank, world_size=worker.world_size)
        return DataLoader(dataset, batch_size=batch_size, sampler=shard, drop_last=False)
    # DistributedSampler is a pure function of (seed, epoch): nothing to checkpoint.
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


def _nanmean_across_ranks(sums: dict[str, list[float]], device: torch.device) -> dict[str, float]:
    """``{key: [sum, count]}`` -> mean per key over every rank; NaN when no rank had a defined value.

    One all_reduce for all keys, in sorted order so every rank packs the tensor identically.
    """
    keys = sorted(sums)
    if torch.distributed.is_available() and torch.distributed.is_initialized() and keys:
        t = torch.tensor([sums[k] for k in keys], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t)
        sums = {k: [float(t[i, 0]), float(t[i, 1])] for i, k in enumerate(keys)}
    return {k: sums[k][0] / sums[k][1] if sums[k][1] else float("nan") for k in keys}


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader[Sample],
    device: torch.device,
    metrics: Sequence[Metric] = (),
    infer: Inferer | None = None,
) -> dict[str, float]:
    model.eval()
    total, count = 0.0, 0
    sums: dict[str, list[float]] = {}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = infer(model, x) if infer is not None else model(x)
        total += float(loss_fn(logits, y)) * x.shape[0]
        count += x.shape[0]
        batch: dict[str, Tensor] = {}
        for metric in metrics:
            for key, values in metric(logits, y).items():
                if key in batch or key == "loss":
                    raise RunnerError(f"two metrics both produce val/{key}; give them distinct names")
                batch[key] = values
        for key, values in batch.items():
            defined = values[~values.isnan()]
            acc = sums.setdefault(key, [0.0, 0.0])
            acc[0] += float(defined.double().sum())
            acc[1] += defined.numel()
    out = {"val/loss": _mean_across_ranks(total, count, device)}
    out.update({f"val/{k}": v for k, v in _nanmean_across_ranks(sums, device).items()})
    return out


def _capture_rng(loaders: dict[str, DataLoader[Sample] | None]) -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state(), "python": random.getstate()}
    for name, loader in loaders.items():
        if loader is not None and loader.generator is not None:
            state[f"loader/{name}"] = loader.generator.get_state()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        kind, keys, pos, has_gauss, cached = np.random.get_state()
        # As a tensor: the checkpoint is loaded with weights_only=True, which refuses ndarrays.
        state["numpy"] = (kind, torch.from_numpy(keys.astype(np.int64)), pos, has_gauss, cached)
    return state


def _restore_rng(state: dict[str, Any], loaders: dict[str, DataLoader[Sample] | None]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    for name, loader in loaders.items():
        if f"loader/{name}" in state and loader is not None and loader.generator is not None:
            loader.generator.set_state(state[f"loader/{name}"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np

        kind, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((kind, keys.numpy().astype(np.uint32), pos, has_gauss, cached))


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
    on_data: Callable[[EnvCheck], None] | None = None,
    resume_from: TrainState | None = None,
    checkpoint_every: int | None = None,
    on_checkpoint: Callable[[TrainState], None] | None = None,
    keep_best: BestCriterion | None = None,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, Any] | None]:
    """Run the epoch loop for one rank. Returns the (unwrapped) model, the full history and,
    with ``keep_best``, the weights (on CPU) of the best epoch (None without it, or when the
    metric was never defined). A ``keep_best`` metric missing from the first epoch's row raises.

    ``resume_from`` continues after ``resume_from.epoch`` with its weights, optimizer state
    and RNG states. ``on_checkpoint`` gets this rank's ``TrainState`` after every
    ``checkpoint_every``-th epoch; its tensors are live, so serialise them right away.

    ``wrap_model`` is where a launcher injects DDP (the returned model is the
    original so its ``state_dict`` has no ``module.`` prefix). ``on_data`` gets
    the data snapshot check first thing (every rank verifies the data it reads);
    ``on_start`` fires once everything is built and data is loaded, right before
    the first epoch.
    """
    if spec.resources.workers != worker.world_size:
        raise RunnerError(
            f"spec says resources.workers={spec.resources.workers} but the launcher "
            f"started world_size={worker.world_size}"
        )
    data_check = verify_data(spec, data)
    if on_data is not None:
        on_data(data_check)

    # Everything that touches RNG happens after this line. Every rank builds the
    # identical model; DDP would broadcast rank 0's weights anyway.
    seed_everything(spec.training.seed, deterministic=spec.training.deterministic)

    model: nn.Module = MODELS.build(spec.model.name, spec.model.params).to(device)
    if resume_from is not None:
        model.load_state_dict(resume_from.model_state)
    loss_fn = LOSSES.build(spec.loss.name, spec.loss.params)
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(device)
    augments: list[Augment] = [AUGMENTATIONS.build(a.name, a.params) for a in spec.augmentation]
    metrics: list[Metric] = [METRICS.build(m.name, m.params) for m in spec.metrics]
    infer: Inferer | None = (
        INFERERS.build(spec.inference.name, spec.inference.params) if spec.inference is not None else None
    )
    optimizer = build_optimizer(spec.training, model.parameters())
    if resume_from is not None:
        optimizer.load_state_dict(resume_from.optimizer_state)
    trained = wrap_model(model) if wrap_model is not None else model

    splits = data.load(spec.dataset, spec.preprocessing)
    if metrics and splits.val is None:
        raise RunnerError(
            f"metrics need a validation split, but {spec.dataset.name!r} served none; "
            "drop `metrics` from the spec or serve a val split"
        )
    bs, seed = spec.per_worker_batch_size(), spec.training.seed
    train_loader = _loader(splits.train, batch_size=bs, shuffle=True, seed=seed, worker=worker)
    # With an inferer, validation goes one volume at a time: full volumes differ in shape.
    val_bs = 1 if infer is not None else bs
    val_loader = (
        _loader(splits.val, batch_size=val_bs, shuffle=False, seed=seed, worker=worker)
        if splits.val is not None
        else None
    )
    if worker.rank:
        # Ranks must not draw the same augmentation coin flips; rank 0 keeps the plain seed
        # so a world_size-1 distributed run reproduces a local run exactly.
        torch.manual_seed(seed + worker.rank)
    loaders = {"train": train_loader, "val": val_loader}
    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    if resume_from is not None:
        # Last, so it overrides every seeding above: the run continues mid-stream.
        _restore_rng(resume_from.rng, loaders)
        history = list(resume_from.history)
        best_state = resume_from.best_model_state
    first = len(history) + 1

    if on_start is not None:
        on_start()
    for epoch in range(first, spec.training.epochs + 1):
        _set_epoch(train_loader, epoch)
        row = {"train/loss": _train_epoch(trained, loss_fn, optimizer, train_loader, augments, device)}
        if val_loader is not None:
            # The unwrapped model: same weights, but no DDP collectives, which would deadlock
            # when ranks hold different numbers of val batches (ShardSampler).
            row.update(_eval_epoch(model, loss_fn, val_loader, device, metrics, infer))
        history.append(row)
        if on_epoch is not None:
            on_epoch(epoch, row)
        if keep_best is not None:
            if keep_best.metric not in row:
                raise RunnerError(
                    f"keep-best metric {keep_best.metric!r} is not logged; this run logs: {', '.join(sorted(row))}"
                )
            found = best_epoch(history, keep_best)
            if found is not None and found[0] == epoch:  # a copy: the live weights keep training
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if checkpoint_every and on_checkpoint is not None and epoch % checkpoint_every == 0:
            on_checkpoint(TrainState(
                epoch=epoch, history=list(history), model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(), rng=_capture_rng(loaders),
                best_model_state=best_state,
            ))
    return model, history, best_state


def verify_data(spec: ExperimentSpec, data: DataProvider) -> EnvCheck:
    """Compare the provider's content hash with ``dataset.snapshot``; raise ``DataError`` on mismatch."""
    check = check_snapshot(spec.dataset.snapshot, data.snapshot(spec.dataset))
    if check.status == "mismatch":
        raise DataError(check.detail)
    return check


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
    checkpoint_every: int | None = None,
    resume: bool = False,
    keep_checkpoint: bool = False,
    keep_best: BestCriterion | None = None,
) -> RunResult:
    """Train ``spec`` once, in this process. Raises ``UnpinnedInputError`` if the spec is not fully pinned.

    With ``artifacts``, the run bundle (spec, manifest, final checkpoint) is
    written on FINISHED; on FAILED the spec and manifest are still written so
    the failure is reproducible. With ``probe``, the actual environment and its
    checks against the spec's pins go into the manifest (recorded, not enforced).
    With ``checkpoint_every=N`` (needs ``artifacts``) the training state is stored after
    every N-th epoch, together with ``spec.json``. ``resume=True`` continues this run_id from
    that checkpoint (a FAILED or killed run; not a FINISHED one): same tracker run, and the
    manifest's ``resumes`` records the epoch and this segment's environment. Once the run is
    FINISHED its ``checkpoint.pt`` is deleted (nothing left to resume; ``reproduce`` replays from
    scratch) unless ``keep_checkpoint``; a FAILED run keeps it. ``keep_best`` also writes the
    best epoch's weights as ``best.pt`` and ``manifest.best`` (FINISHED only); a resume continues
    with the criterion stored in the checkpoint.
    Requires ``resources.workers == 1``; use a distributed launcher otherwise.
    """
    experiment_id = compute_experiment_id(spec)
    run_id = compute_run_id(experiment_id, spec.resources, attempt)
    if checkpoint_every is not None and checkpoint_every < 1:
        raise RunnerError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
    if checkpoint_every and artifacts is None:
        raise RunnerError("checkpoint_every needs an artifact store to write checkpoint.pt into")
    if resume and artifacts is None:
        raise RunnerError("resume needs an artifact store holding the run's checkpoint.pt")
    ckpt = _resume_point(artifacts, run_id, experiment_id) if resume else None
    dev = resolve_device(spec.resources, device)
    segment_env = record_environment(spec, probe.capture(spec)) if probe is not None else None

    if ckpt is None:
        environment, resumes, history = segment_env, [], []
    else:
        environment = ckpt.environment
        resumes = [*ckpt.resumes, ResumePoint(epoch=ckpt.epoch, environment=segment_env)]
        history = list(ckpt.history)
        checkpoint_every = checkpoint_every or ckpt.checkpoint_every
        keep_best = resumed_criterion(keep_best, ckpt.keep_best)
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
            data_snapshot=data_snapshot,
            resumes=resumes,
            best=best_record(history, keep_best) if keep_best is not None and status == "FINISHED" else None,
        )

    data_snapshot: EnvCheck | None = None
    spec_saved = False

    def on_data(check: EnvCheck) -> None:
        nonlocal data_snapshot
        data_snapshot = check

    def on_start() -> None:
        nonlocal started
        tracker.start_run(run_id=run_id, experiment_id=experiment_id, spec=spec, resume=ckpt is not None)
        started = True

    def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
        tracker.log_metrics(metrics, step=epoch)
        history.append(metrics)

    def on_checkpoint(state: TrainState) -> None:
        nonlocal spec_saved
        assert artifacts is not None and checkpoint_every is not None
        if not spec_saved:  # a killed run never reaches save_bundle; resume needs the spec
            save_spec(artifacts, run_id, spec)
            spec_saved = True
        save_checkpoint(artifacts, Checkpoint(
            experiment_id=experiment_id, run_id=run_id, attempt=attempt, checkpoint_every=checkpoint_every,
            state=state, environment=environment, resumes=resumes, keep_best=keep_best,
        ))

    try:
        model, _, best_state = train_worker(
            spec, data=data, device=dev, on_start=on_start, on_epoch=on_epoch, on_data=on_data,
            resume_from=ckpt.state if ckpt is not None else None,
            checkpoint_every=checkpoint_every, on_checkpoint=on_checkpoint, keep_best=keep_best,
        )
    except BaseException:
        if started:
            tracker.end_run("FAILED")
            if artifacts is not None:
                save_bundle(artifacts, spec=spec, manifest=_manifest("FAILED"), model_state=None)
        raise
    tracker.end_run("FINISHED")
    if artifacts is not None:
        save_bundle(artifacts, spec=spec, manifest=_manifest("FINISHED"), model_state=model.state_dict(),
                    best_state=best_state)
        if not keep_checkpoint:  # after the FINISHED manifest landed: a crash in between only leaves it
            artifacts.delete(run_id, CHECKPOINT)

    return RunResult(
        experiment_id=experiment_id, run_id=run_id, device=str(dev), history=history,
        environment=environment, data_snapshot=data_snapshot, resumes=resumes,
    )


def resumed_criterion(requested: BestCriterion | None, stored: BestCriterion | None) -> BestCriterion | None:
    """A resume continues with the run's own keep-best criterion; asking for another is refused
    (the best so far was chosen by the stored one, or not tracked at all)."""
    if requested is not None and requested != stored:
        started = f"--keep-best {stored}" if stored is not None else "no --keep-best"
        raise RunnerError(f"this run was started with {started}; a resume cannot change it to {requested}")
    return stored


def _resume_point(store: ArtifactStore, run_id: str, experiment_id: str) -> Checkpoint:
    # FINISHED first: a finished run has usually dropped its checkpoint, and that is the reason.
    try:
        status = RunManifest.model_validate_json(store.get(run_id, MANIFEST)).status
    except ArtifactError:  # killed before any manifest was written
        status = None
    if status == "FINISHED":
        raise RunnerError(f"run {run_id} already FINISHED; nothing to resume")
    try:
        ckpt = load_checkpoint(store, run_id)
    except ArtifactError:
        raise RunnerError(
            f"no checkpoint for run {run_id}: nothing to resume "
            "(only runs started with checkpoint_every can be resumed)"
        ) from None
    if (ckpt.run_id, ckpt.experiment_id) != (run_id, experiment_id):
        raise RunnerError(f"checkpoint stored under run {run_id} belongs to run {ckpt.run_id}")
    return ckpt
