"""What a run produced, as pure data. Written into every run bundle."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlops.domain.best import BestRecord
from mlops.domain.environment import EnvCheck, EnvironmentRecord
from mlops.domain.errors import BundleFormatError

FORMAT_VERSION = 1


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

    format_version: int = Field(default=0, ge=0, strict=True)
    experiment_id: str
    run_id: str
    attempt: int = Field(ge=1)
    status: Literal["FINISHED", "FAILED"]
    device: str
    # Which launcher ran it ("local", "ray"): a replay needs the same one for bit-exact weights.
    # None = a bundle written before launchers were recorded.
    launcher: str | None = None
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
    # Run started with keep-best: which epoch best.pt holds (FINISHED only). None = not asked.
    best: BestRecord | None = None
    # Cases per part ({"train": n, "val": n, "test": n}) when the spec sets `split`; which cases
    # they are is in split.json. None = the provider's own split.
    split: dict[str, int] | None = None


def check_format(raw: str | bytes, key: str, known: int, what: str) -> None:
    try:
        data = json.loads(raw)
    except ValueError:
        return
    if isinstance(data, dict):
        version = data.get(key, 0)
        if isinstance(version, int) and not isinstance(version, bool) and version > known:
            raise BundleFormatError(what, version, known)


def parse_manifest(raw: str | bytes) -> RunManifest:
    check_format(raw, "format_version", FORMAT_VERSION, "manifest.json")
    return RunManifest.model_validate_json(raw)
