"""Port: where a run's files go (local disk, S3, MLflow artifacts, a test double).

The store is a dumb bucket of bytes keyed by ``(run_id, name)``. It does not
know what a checkpoint or a manifest is; ``mlops.application.bundle`` defines
the layout. Names are relative POSIX paths (``checkpoints/final.pt``); a name
that could escape the run's directory must be refused.
"""

from __future__ import annotations

from typing import Protocol


class ArtifactError(Exception):
    """Unknown run, missing artifact, or an unsafe artifact name."""


def validate_name(name: str) -> str:
    """Return ``name`` if it is a safe relative path, else raise ``ArtifactError``."""
    if not name or name.startswith("/") or "\\" in name:
        raise ArtifactError(f"unsafe artifact name {name!r}")
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ArtifactError(f"unsafe artifact name {name!r}")
    return name


class ArtifactStore(Protocol):
    def put(self, run_id: str, name: str, data: bytes) -> None:
        """Store ``data`` under ``name`` for ``run_id``, replacing any previous value."""
        ...

    def get(self, run_id: str, name: str) -> bytes:
        """Raise ``ArtifactError`` if ``run_id`` or ``name`` is unknown."""
        ...

    def list(self, run_id: str) -> list[str]:
        """Sorted artifact names for ``run_id``. Raise ``ArtifactError`` for an unknown run."""
        ...

    def runs(self) -> list[str]:
        """Sorted ids of every run holding at least one artifact (empty store -> [])."""
        ...
