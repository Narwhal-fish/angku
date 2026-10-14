"""What a run produced, as pure data. Written into every run bundle."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlops.domain.environment import EnvCheck, EnvironmentRecord


class ResumePoint(BaseModel):
    """The run was interrupted and continued from the checkpoint taken after ``epoch``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch: int = Field(ge=0)
    # Where the resumed segment ran (it may be another machine). None = no probe given.
    environment: EnvironmentRecord | None = None


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
    # Every time the run was resumed from a checkpoint, in order. Empty = ran in one go.
    # ``environment`` above describes the first segment.
    resumes: list[ResumePoint] = Field(default_factory=list)
