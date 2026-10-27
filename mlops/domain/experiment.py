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
from typing import Any, Literal

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
            raise ValueError(f"code.commit must be a full 40-hex git commit SHA (not a branch/tag), got {v!r}")
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
            raise ValueError(f"environment.image_digest must look like 'sha256:<64 hex>', got {v!r}")
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
            raise ValueError(f"dataset.snapshot must be an immutable snapshot id, got {v!r}")
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


class Metric(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Inference(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Split(_Strict):
    """How the dataset's cases are split (``mlops.domain.split.assign_split``). Optional: a spec
    without it keeps the provider's own split, so ids from before the field existed hold.

    ``holdout``: ``val`` (+ ``test``) as fractions of all cases. ``kfold``: ``test`` is taken out
    first, the rest is cut into ``k`` folds and fold ``fold`` is val; each fold is its own
    experiment. Defaults are filled in here, so ``spec.json`` and the fingerprint always carry
    the values actually used. ``seed`` orders the cases for ``shuffled``; it is not
    ``training.seed`` (another training seed must not move cases into the test set).
    """

    kind: Literal["holdout", "kfold"] = "holdout"
    val: float | None = None
    test: float = 0.0
    k: int | None = None
    fold: int | None = None
    order: Literal["shuffled", "sorted"] = "shuffled"
    seed: int = 0

    @model_validator(mode="after")
    def _fields_fit_the_kind(self) -> Split:
        if not 0.0 <= self.test < 1.0:
            raise ValueError(f"split.test must be in [0, 1), got {self.test}")
        if self.kind == "holdout":
            if self.k is not None or self.fold is not None:
                raise ValueError("split.k and split.fold are for kind: kfold")
            val = 0.2 if self.val is None else self.val
            if not 0.0 < val < 1.0:
                raise ValueError(f"split.val must be in (0, 1), got {val}")
            if val + self.test >= 1.0:
                raise ValueError(f"split.val + split.test must leave training cases, got {val} + {self.test}")
            object.__setattr__(self, "val", val)
        else:
            if self.val is not None:
                raise ValueError("split.val is for kind: holdout (kfold uses k and fold)")
            k = 5 if self.k is None else self.k
            if k < 2:
                raise ValueError(f"split.k must be >= 2, got {k}")
            if self.fold is None:
                raise ValueError(
                    f"split.fold is required for kind: kfold (0..{k - 1}); each fold is its own experiment"
                )
            if not 0 <= self.fold < k:
                raise ValueError(f"split.fold must be in 0..{k - 1}, got {self.fold}")
            object.__setattr__(self, "k", k)
        return self


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
    # Validation metrics beyond val/loss. They never touch the weights, but they shape the
    # history a bundle promises to reproduce, so they are in the fingerprint (when present).
    metrics: list[Metric] = Field(default_factory=list)
    # How the model is applied to a validation volume (e.g. sliding window). None = model(x).
    # Shapes val/loss and the metrics, so it is in the fingerprint (when present).
    inference: Inference | None = None
    # Which cases are train / val / test. None = the provider's own split. In the fingerprint
    # (when present): it decides what the model sees.
    split: Split | None = None
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
        model, training (incl. seed), augmentation, loss, metrics, inference and split (each only
        when set, so ids from before those fields existed are unchanged).
        Excludes: meta, image tag, dataset version label, resources, tracking.
        """
        fp: dict[str, Any] = {
            "code": {"commit": self.code.commit},
            "environment": {"image_digest": self.environment.image_digest},
            "dataset": {"name": self.dataset.name, "snapshot": self.dataset.snapshot},
            "preprocessing": self.preprocessing,
            "model": self.model.model_dump(),
            "training": self.training.model_dump(),
            "augmentation": [a.model_dump() for a in self.augmentation],
            "loss": self.loss.model_dump(),
        }
        if self.metrics:
            fp["metrics"] = [m.model_dump() for m in self.metrics]
        if self.inference is not None:
            fp["inference"] = self.inference.model_dump()
        if self.split is not None:
            fp["split"] = self.split.model_dump()
        return fp
