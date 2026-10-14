"""Artifact store on a local filesystem: ``<root>/<run_id>/<name>``.

Works for a workstation and for any mounted volume (NFS, a PVC in a Ray
worker). Writes are atomic per artifact: bytes go to a temp file first and
are renamed into place, so a crash never leaves a half-written checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

from mlops.ports.artifacts import ArtifactError, validate_name


class LocalArtifactStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def put(self, run_id: str, name: str, data: bytes) -> None:
        path = self._path(run_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get(self, run_id: str, name: str) -> bytes:
        path = self._path(run_id, name)
        if not path.is_file():
            if not (self.root / run_id).is_dir():
                raise ArtifactError(f"unknown run {run_id}")
            raise ArtifactError(f"no artifact {name!r} for run {run_id}")
        return path.read_bytes()

    def list(self, run_id: str) -> list[str]:
        run_dir = self.root / run_id
        if not run_dir.is_dir():
            raise ArtifactError(f"unknown run {run_id}")
        return sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file())

    def runs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and any(p.iterdir()))

    def _path(self, run_id: str, name: str) -> Path:
        if not run_id or "/" in run_id or run_id in (".", ".."):
            raise ArtifactError(f"unsafe run id {run_id!r}")
        return self.root / run_id / validate_name(name)
