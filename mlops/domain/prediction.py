"""What a prediction over earlier runs produced, as pure data (``result.json``).

A result is derived from run bundles, never part of them: the member runs are read, not
written. ``result_id`` hashes everything that decides the numbers (which runs, which weights
file of each, which part of the data, the method), so the same inputs give the same id and,
predictions being deterministic, the same numbers.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mlops.domain.environment import EnvCheck, EnvironmentRecord
from mlops.domain.identity import canonical_json
from mlops.domain.manifest import check_format

RESULT_FORMAT_VERSION = 1

Weights = Literal["model.pt", "best.pt"]
Part = Literal["val", "test"]
Kind = Literal["predict", "ensemble"]
Vote = Literal["soft", "hard"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")


class Member(_Frozen):
    """One run whose model takes part, and the weights file used."""

    run_id: str
    experiment_id: str
    weights: Weights
    # The epoch those weights are from (best.pt: the best epoch; model.pt: the last one).
    epoch: int | None = None


class CaseResult(_Frozen):
    case: str
    # Per case: "loss" and every key the spec's metrics produce; NaN = undefined for this case.
    metrics: dict[str, float]


class PredictionRecord(_Frozen):
    result_id: str
    format_version: int = Field(default=0, ge=0, strict=True)
    kind: Kind = "predict"
    name: str  # meta.name of the (first) member's spec, for humans only
    members: list[Member]
    part: Part
    dataset: str
    data_snapshot: EnvCheck
    method: dict[str, Any] = Field(default_factory=dict)
    device: str
    cases: list[CaseResult]
    # Mean over the cases where each metric is defined (NaN if none).
    metrics: dict[str, float]
    # ensemble only: each member's own means on the same cases, by run_id (did voting help?)
    member_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    environment: EnvironmentRecord | None = None
    # Caveats a reader must see next to the numbers (e.g. val was used for model selection).
    notes: list[str] = Field(default_factory=list)


def compute_result_id(
    *, kind: str, members: list[Member], part: str, dataset_snapshot: str | None, method: dict[str, Any]
) -> str:
    payload = {
        "kind": kind,
        "members": [{"run_id": m.run_id, "weights": m.weights} for m in members],
        "part": part,
        "dataset_snapshot": dataset_snapshot,
        "method": method,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def parse_result(raw: str | bytes) -> PredictionRecord:
    check_format(raw, "format_version", RESULT_FORMAT_VERSION, "result.json")
    return PredictionRecord.model_validate_json(raw)
