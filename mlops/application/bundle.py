"""The run bundle: everything needed to reproduce a run, laid out in an ArtifactStore.

    spec.json       the full ExperimentSpec (meta and tracking included, so it can be re-run as-is)
    manifest.json   RunManifest: ids, attempt, status, per-epoch history
    model.pt        final model ``state_dict`` (absent when the run FAILED)
    best.pt         ``state_dict`` of the best epoch (``manifest.best``), only with keep-best, FINISHED only
    checkpoint.pt   latest mid-run training state, only with ``checkpoint_every`` on the local
                    launcher: what ``resume`` continues from. Not evidence; a replay never reads it.
    launch.json     how a checkpointing Ray run was launched (attempt, Ray storage path): the
                    checkpoints themselves live in Ray's storage, this says where to resume from.

JSON rather than YAML on purpose: ``mlops.config`` is the only YAML-aware
layer, and a bundle must be readable without it.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
from typing import Any

import torch

from mlops.domain.best import BestCriterion
from mlops.domain.environment import EnvironmentRecord
from mlops.domain.experiment import ExperimentSpec
from mlops.domain.manifest import ResumePoint, RunManifest
from mlops.ports.artifacts import ArtifactStore

SPEC = "spec.json"
MANIFEST = "manifest.json"
MODEL = "model.pt"
BEST = "best.pt"
CHECKPOINT = "checkpoint.pt"
LAUNCH = "launch.json"
_CHECKPOINT_FORMAT = 1


@dataclass(frozen=True)
class Bundle:
    spec: ExperimentSpec
    manifest: RunManifest
    model_state: dict[str, Any] | None
    best_state: dict[str, Any] | None = None


def save_bundle(
    store: ArtifactStore,
    *,
    spec: ExperimentSpec,
    manifest: RunManifest,
    model_state: Mapping[str, Any] | None,
    best_state: Mapping[str, Any] | None = None,
) -> None:
    run_id = manifest.run_id
    save_spec(store, run_id, spec)
    # Weights before the manifest: a FINISHED manifest promises its files are there.
    for name, state in ((MODEL, model_state), (BEST, best_state)):
        if state is not None:
            buf = io.BytesIO()
            torch.save(dict(state), buf)
            store.put(run_id, name, buf.getvalue())
    store.put(run_id, MANIFEST, manifest.model_dump_json(indent=2).encode())


def load_bundle(store: ArtifactStore, run_id: str) -> Bundle:
    spec = ExperimentSpec.model_validate_json(store.get(run_id, SPEC))
    manifest = RunManifest.model_validate_json(store.get(run_id, MANIFEST))
    names = store.list(run_id)

    def weights(name: str) -> dict[str, Any] | None:
        if name not in names:
            return None
        return torch.load(io.BytesIO(store.get(run_id, name)), map_location="cpu", weights_only=True)

    return Bundle(spec=spec, manifest=manifest, model_state=weights(MODEL), best_state=weights(BEST))


def save_spec(store: ArtifactStore, run_id: str, spec: ExperimentSpec) -> None:
    store.put(run_id, SPEC, spec.model_dump_json(indent=2).encode())


# -- mid-run checkpoint --------------------------------------------------------


@dataclass(frozen=True)
class TrainState:
    """Everything the epoch loop needs to continue exactly where it stopped."""

    epoch: int  # epochs completed
    history: list[dict[str, float]]
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    # RNG states (torch / cuda / python / numpy / the train loader's generator): without them
    # a resumed run draws different shuffles and augmentations and is no longer the same run.
    rng: dict[str, Any]
    # keep-best: weights of the best epoch so far (which epoch is derived from the history).
    best_model_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class Checkpoint:
    experiment_id: str
    run_id: str
    attempt: int
    checkpoint_every: int
    state: TrainState
    # The first segment's environment and earlier resume points: a killed run never wrote its
    # manifest, so these ride along until the final one is written.
    environment: EnvironmentRecord | None = None
    resumes: list[ResumePoint] = field(default_factory=list)
    keep_best: BestCriterion | None = None  # the run's criterion: a resume continues with it

    @property
    def epoch(self) -> int:
        return self.state.epoch

    @property
    def history(self) -> list[dict[str, float]]:
        return self.state.history


def save_checkpoint(store: ArtifactStore, ckpt: Checkpoint) -> None:
    """One file, so a store with atomic per-file writes never holds a half-written checkpoint."""
    st = ckpt.state
    payload = {
        "format": _CHECKPOINT_FORMAT,
        "experiment_id": ckpt.experiment_id,
        "run_id": ckpt.run_id,
        "attempt": ckpt.attempt,
        "checkpoint_every": ckpt.checkpoint_every,
        "epoch": st.epoch,
        "history": st.history,
        "model": dict(st.model_state),
        "optimizer": st.optimizer_state,
        "rng": st.rng,
        "environment": ckpt.environment.model_dump_json() if ckpt.environment is not None else None,
        "resumes": [r.model_dump_json() for r in ckpt.resumes],
        "keep_best": str(ckpt.keep_best) if ckpt.keep_best is not None else None,
        "best_model": dict(st.best_model_state) if st.best_model_state is not None else None,
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    store.put(ckpt.run_id, CHECKPOINT, buf.getvalue())


def load_checkpoint(store: ArtifactStore, run_id: str) -> Checkpoint:
    """Raise ``ArtifactError`` when ``run_id`` has no checkpoint."""
    p = torch.load(io.BytesIO(store.get(run_id, CHECKPOINT)), map_location="cpu", weights_only=True)
    if p.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint format {p.get('format')!r} not understood (want {_CHECKPOINT_FORMAT})")
    return Checkpoint(
        experiment_id=p["experiment_id"],
        run_id=p["run_id"],
        attempt=p["attempt"],
        checkpoint_every=p["checkpoint_every"],
        state=TrainState(
            epoch=p["epoch"], history=p["history"], model_state=p["model"],
            optimizer_state=p["optimizer"], rng=p["rng"], best_model_state=p.get("best_model"),
        ),
        environment=EnvironmentRecord.model_validate_json(p["environment"]) if p["environment"] else None,
        resumes=[ResumePoint.model_validate_json(r) for r in p["resumes"]],
        # .get: checkpoints written before keep-best existed have neither key
        keep_best=BestCriterion.parse(p["keep_best"]) if p.get("keep_best") else None,
    )


# -- launch record (Ray runs keep their checkpoints in Ray's storage) ---------------------------


@dataclass(frozen=True)
class LaunchRecord:
    launcher: str
    attempt: int
    checkpoint_every: int
    storage_path: str | None = None
    keep_best: str | None = None  # BestCriterion as text; a resume continues with it


def save_launch(store: ArtifactStore, run_id: str, launch: LaunchRecord) -> None:
    store.put(run_id, LAUNCH, json.dumps(asdict(launch), indent=2).encode())


def load_launch(store: ArtifactStore, run_id: str) -> LaunchRecord:
    """Raise ``ArtifactError`` when the run has no launch record."""
    return LaunchRecord(**json.loads(store.get(run_id, LAUNCH)))
