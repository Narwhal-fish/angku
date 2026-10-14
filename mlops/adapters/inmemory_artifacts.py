"""Artifact store that keeps everything in memory. For tests and dry runs."""

from __future__ import annotations

from mlops.ports.artifacts import ArtifactError, validate_name


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self._runs: dict[str, dict[str, bytes]] = {}

    def put(self, run_id: str, name: str, data: bytes) -> None:
        self._runs.setdefault(run_id, {})[validate_name(name)] = bytes(data)

    def get(self, run_id: str, name: str) -> bytes:
        validate_name(name)
        try:
            return self._run(run_id)[name]
        except KeyError:
            raise ArtifactError(f"no artifact {name!r} for run {run_id}") from None

    def list(self, run_id: str) -> list[str]:
        return sorted(self._run(run_id))

    def runs(self) -> list[str]:
        return sorted(r for r, files in self._runs.items() if files)

    def _run(self, run_id: str) -> dict[str, bytes]:
        try:
            return self._runs[run_id]
        except KeyError:
            raise ArtifactError(f"unknown run {run_id}") from None
