from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

HEARTBEAT = 15.0  # seconds between heartbeats
STALE_AFTER = 60.0  # a heartbeat older than this = the process is gone


class Progress(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    run_id: str
    state: Literal["running", "ended"]
    epoch: int  # last epoch logged (absolute; 0 = none yet in this segment)
    epochs: int  # spec.training.epochs
    started: float  # unix seconds
    updated: float  # last write: epoch or heartbeat
    history: list[dict[str, float]]  # epochs logged by this process (a resume starts empty)


def is_live(p: Progress, *, now: float) -> bool:
    return p.state == "running" and now - p.updated <= STALE_AFTER
