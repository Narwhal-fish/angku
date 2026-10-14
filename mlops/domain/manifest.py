"""What a run produced, as pure data. Written into every run bundle."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlops.domain.environment import EnvCheck, EnvironmentRecord


class RunManifest(BaseModel):
    # NaN is a real value here (a metric undefined for the whole split); plain JSON would
    # write it as null and the bundle would no longer load.
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

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
