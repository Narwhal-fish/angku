"""Port: where Ray Train keeps a run's checkpoints (``storage_path/<ray run name>``).

Outside the artifact root: ``launch.json`` records the storage path, Ray writes the directory.
Only ``angku gc`` needs this port, to remove what a deleted run left there.
"""

from __future__ import annotations

from typing import Protocol


class RayStorage(Protocol):
    def exists(self, storage_path: str, name: str) -> bool | None:
        """Whether ``storage_path/name`` exists; None = cannot tell from here (remote storage)."""
        ...

    def delete(self, storage_path: str, name: str) -> None:
        """Remove ``storage_path/name`` and everything under it; missing is not an error."""
        ...
