"""What a run produced, as pure data. Written into every run bundle."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlops.domain.environment import EnvCheck, EnvironmentRecord


class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    run_id: str
    attempt: int = Field(ge=1)
    status: Literal["FINISHED", "FAILED"]
    device: str
    # One entry per completed epoch, in order.
    history: list[dict[str, float]] = Field(default_factory=list)
    # What the run actually ran on, verified against the spec's pins. None = not captured
    # (no probe given, or a bundle written before environments were recorded).
    environment: EnvironmentRecord | None = None
    # The served data vs dataset.snapshot. None only in bundles predating the check.
    data_snapshot: EnvCheck | None = None
