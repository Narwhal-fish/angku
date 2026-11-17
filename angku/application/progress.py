"""Live progress of a run that is still training: ``progress.json`` next to the bundle.

Not part of the bundle: derived, in no id, never read by ``reproduce``. Written by the
``LiveProgress`` tracker adapter (start, every epoch, and a heartbeat), removed by the CLI once
the launch returned. A run is live while its progress says ``running`` and the heartbeat is
fresher than ``STALE_AFTER`` seconds; a killed process stops beating and goes stale. A manifest
always wins: a leftover progress file next to one is ignored.
"""

from __future__ import annotations

from pydantic import ValidationError

from angku.domain.progress import HEARTBEAT, STALE_AFTER, Progress, is_live
from angku.ports.artifacts import ArtifactError, ArtifactStore

__all__ = [
    "HEARTBEAT",
    "PROGRESS",
    "STALE_AFTER",
    "Progress",
    "is_live",
    "load_progress",
    "read_progress",
    "save_progress",
]

PROGRESS = "progress.json"


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
