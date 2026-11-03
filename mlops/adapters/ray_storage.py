"""Ray storage on a local or mounted filesystem, for ``angku gc``. Plain pathlib: no ray import.

``mount`` replaces the recorded ``storage_path`` when the same storage is mounted somewhere else
on this machine (a KubeRay pod writes ``/ray-storage``; on the kind host it is
``.kind/ray-storage``). Without ``mount``, a remote path (``s3://...``) cannot be checked:
``exists`` says None and ``delete`` refuses.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

_RAY_RUN_NAME = re.compile(r"mlops-[0-9a-f]{12}-a[0-9]+")


class LocalRayStorage:
    def __init__(self, mount: str | os.PathLike[str] | None = None) -> None:
        self.mount = Path(mount) if mount is not None else None

    def path(self, storage_path: str, name: str) -> Path | None:
        """The directory on this machine; None for remote storage that is not mounted."""
        if not _RAY_RUN_NAME.fullmatch(name):
            raise ValueError(f"not a Ray run name of ours: {name!r}")
        if self.mount is not None:
            return self.mount / name
        if "://" in storage_path:
            return None
        return Path(storage_path) / name

    def exists(self, storage_path: str, name: str) -> bool | None:
        p = self.path(storage_path, name)
        return None if p is None else p.is_dir()

    def delete(self, storage_path: str, name: str) -> None:
        p = self.path(storage_path, name)
        if p is None:
            raise ValueError(f"cannot delete remote Ray storage {storage_path}/{name} from here")
        if p.is_dir():
            shutil.rmtree(p)

    def size(self, storage_path: str, name: str) -> int | None:
        p = self.path(storage_path, name)
        if p is None or not p.is_dir():
            return None
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
