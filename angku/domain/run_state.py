from __future__ import annotations

from enum import StrEnum

from angku.domain.manifest import RunManifest
from angku.domain.progress import Progress, is_live


class RunState(StrEnum):
    ABSENT = "ABSENT"
    RUNNING = "RUNNING"
    RESUMING = "RESUMING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    INCOMPLETE = "INCOMPLETE"
    UNREADABLE = "UNREADABLE"


LIVE = frozenset({RunState.RUNNING, RunState.RESUMING})


def derive_state(
    *,
    has_spec: bool,
    has_checkpoint: bool,
    manifest: RunManifest | None,
    progress: Progress | None,
    now: float,
    unreadable: bool = False,
) -> RunState:
    if unreadable:
        return RunState.UNREADABLE
    if not has_spec:
        return RunState.ABSENT if manifest is None else RunState.UNREADABLE
    live = progress is not None and is_live(progress, now=now)
    if manifest is None:
        if live:
            return RunState.RUNNING
        return RunState.INTERRUPTED if has_checkpoint else RunState.INCOMPLETE
    if manifest.status == "FAILED":
        return RunState.RESUMING if live else RunState.FAILED
    return RunState.FINISHED


def is_resumable(state: RunState, *, has_checkpoint: bool) -> bool:
    return has_checkpoint and state in (RunState.FAILED, RunState.INTERRUPTED)
