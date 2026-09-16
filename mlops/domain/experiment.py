"""Experiment specification: the single core abstraction of the platform.

Pure Python + Pydantic. No infrastructure imports allowed here.

Every section is frozen (immutable) and forbids unknown fields so that a typo
such as ``learing_rate`` fails at load time instead of silently being ignored.
Sections whose *contents* are researcher-defined (model.params, loss.params,
preprocessing, augmentation params) are free-form dicts; their *structure*
(that there is a ``name`` and a ``params``) is still enforced.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UNPINNED_WORDS = {"", "latest", "main", "master", "head"}


class _Strict(BaseModel):
    """Base for every section: immutable, unknown fields rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class Meta(_Strict):
    """Human-facing labels. Never part of any id."""

    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class Code(_Strict):
    repo: str
    # Full 40-hex git commit SHA. None means "not pinned yet" (draft).
    commit: str | None = None

    @field_validator("commit")
    @classmethod
    def _must_be_full_sha(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not GIT_SHA_RE.match(v):
            raise ValueError(
                f"code.commit must be a full 40-hex git commit SHA (not a branch/tag), got {v!r}"
            )
        return v


class Environment(_Strict):
    # Human-readable image reference (may contain a tag). Not part of ExperimentID.
    image: str
    # Content-addressed digest. This is what pins the environment.
    image_digest: str | None = None

    @field_validator("image_digest")
    @classmethod
    def _must_be_digest(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not IMAGE_DIGEST_RE.match(v):
            raise ValueError(
                f"environment.image_digest must look like 'sha256:<64 hex>', got {v!r}"
            )
        return v


class Dataset(_Strict):
    name: str
    # Human-readable version label (e.g. "v2"). Not part of ExperimentID.
    version: str = ""
    # Immutable snapshot id (content hash / DVC rev / lakeFS commit). This pins the data.
    snapshot: str | None = None

    @field_validator("snapshot")
    @classmethod
    def _not_a_moving_target(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v.strip().lower() in _UNPINNED_WORDS:
            raise ValueError(
                f"dataset.snapshot must be an immutable snapshot id, got {v!r}"
            )
        return v


class Model(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Training(_Strict):
    epochs: int = Field(gt=0)
    # Global effective batch size. The runner divides this by world_size;
    # this is what lets `resources` stay out of the ExperimentID.
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    optimizer: str
    seed: int
    # Force torch onto deterministic kernels (slower on GPU, but two runs with the same
    # seed give bit-identical weights). Part of the fingerprint because it changes results.
    deterministic: bool = True


class Augmentation(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Loss(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Resources(_Strict):
    """Hardware allocation. Part of RunID only, never ExperimentID.

    ``workers`` is the world size: the number of training processes. ``gpu``,
    ``cpu`` and ``memory`` are *per worker*. ``gpu > 0`` means every worker
    needs CUDA.
    """

    gpu: int = Field(ge=0)
    cpu: int = Field(gt=0)
    memory: str
    workers: int = Field(default=1, ge=1)


class Tracking(_Strict):
    """Where metadata goes. Operational; never part of any id."""

    backend: str = "mlflow"
    experiment_name: str = ""
    uri: str = ""


class ExperimentSpec(_Strict):
    meta: Meta
    code: Code
    environment: Environment
    dataset: Dataset
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    model: Model
    training: Training
    augmentation: list[Augmentation] = Field(default_factory=list)
    loss: Loss
    resources: Resources
    tracking: Tracking = Field(default_factory=Tracking)

    @model_validator(mode="after")
    def _batch_splits_across_workers(self) -> ExperimentSpec:
        if self.training.batch_size % self.resources.workers:
            raise ValueError(
                f"training.batch_size={self.training.batch_size} must be divisible by "
                f"resources.workers={self.resources.workers} (batch_size is the global batch)"
            )
        return self

    def per_worker_batch_size(self) -> int:
        return self.training.batch_size // self.resources.workers

    def unpinned_inputs(self) -> list[str]:
        """Dotted paths of inputs that are still not pinned to an immutable id."""
        missing: list[str] = []
        if self.code.commit is None:
            missing.append("code.commit")
        if self.environment.image_digest is None:
            missing.append("environment.image_digest")
        if self.dataset.snapshot is None:
            missing.append("dataset.snapshot")
        return missing

    def is_pinned(self) -> bool:
        return not self.unpinned_inputs()

    def fingerprint(self) -> dict[str, Any]:
        """The exact subset of the spec that defines the scientific experiment.

        Includes: code commit, image digest, dataset name+snapshot, preprocessing,
        model, training (incl. seed), augmentation, loss.
        Excludes: meta, image tag, dataset version label, resources, tracking.
        """
        return {
            "code": {"commit": self.code.commit},
            "environment": {"image_digest": self.environment.image_digest},
            "dataset": {"name": self.dataset.name, "snapshot": self.dataset.snapshot},
            "preprocessing": self.preprocessing,
            "model": self.model.model_dump(),
            "training": self.training.model_dump(),
            "augmentation": [a.model_dump() for a in self.augmentation],
            "loss": self.loss.model_dump(),
        }
