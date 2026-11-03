"""Live progress of a run that is still training: ``progress.json`` next to the bundle.

Not part of the bundle: derived, in no id, never read by ``reproduce``. Written by the
``LiveProgress`` tracker adapter (start, every epoch, and a heartbeat), removed by the CLI once
the launch returned. A run is live while its progress says ``running`` and the heartbeat is
fresher than ``STALE_AFTER`` seconds; a killed process stops beating and goes stale. A manifest
always wins: a leftover progress file next to one is ignored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from angku.ports.artifacts import ArtifactError, ArtifactStore

PROGRESS = "progress.json"
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


def save_progress(store: ArtifactStore, p: Progress) -> None:
    store.put(p.run_id, PROGRESS, p.model_dump_json().encode())


def load_progress(store: ArtifactStore, run_id: str) -> Progress:
    return Progress.model_validate_json(store.get(run_id, PROGRESS))


def read_progress(store: ArtifactStore, run_id: str) -> Progress | None:
    """``load_progress`` that treats a missing or unreadable file as no progress."""
    try:
        return load_progress(store, run_id)
    except (ArtifactError, ValidationError, ValueError):
        return None


def is_live(p: Progress, *, now: float) -> bool:
    return p.state == "running" and now - p.updated <= STALE_AFTER
