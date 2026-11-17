from __future__ import annotations

import time
from enum import StrEnum
from typing import NamedTuple

from angku.application.bundle import MANIFEST, SPEC
from angku.application.errors import RunnerError
from angku.application.listing import RunSummary, summarise_run
from angku.domain.run_state import LIVE, RunState, derive_state
from angku.ports.artifacts import ArtifactError, ArtifactStore


class Event(StrEnum):
    RUN = "run"
    RESUME = "resume"
    FINISH = "finish"
    FAIL = "fail"
    KILL = "kill"
    GC = "gc"


class Transition(NamedTuple):
    source: RunState
    event: Event
    target: RunState


ALLOWED_TRANSITIONS: frozenset[Transition] = frozenset(
    Transition(RunState[s], Event[e], RunState[t])
    for s, e, t in (
        ("ABSENT", "RUN", "RUNNING"),
        ("RUNNING", "FINISH", "FINISHED"),
        ("RUNNING", "FAIL", "FAILED"),
        ("RUNNING", "KILL", "INTERRUPTED"),
        ("RUNNING", "KILL", "INCOMPLETE"),
        ("FAILED", "RESUME", "RESUMING"),
        ("INTERRUPTED", "RESUME", "RUNNING"),
        ("RESUMING", "FINISH", "FINISHED"),
        ("RESUMING", "FAIL", "FAILED"),
        ("RESUMING", "KILL", "FAILED"),
        ("RESUMING", "KILL", "INTERRUPTED"),
        ("FAILED", "GC", "ABSENT"),
        ("INTERRUPTED", "GC", "ABSENT"),
        ("INCOMPLETE", "GC", "ABSENT"),
        ("FINISHED", "GC", "ABSENT"),
    )
)


class TransitionError(RunnerError):
    pass


def allows(state: RunState, event: Event) -> bool:
    return any(t.source is state and t.event is event for t in ALLOWED_TRANSITIONS)


def current_state(store: ArtifactStore, run_id: str, *, now: float | None = None) -> tuple[RunState, RunSummary | None]:
    now = time.time() if now is None else now
    try:
        files = store.list(run_id)
    except ArtifactError:
        files = []
    if SPEC not in files:
        state = derive_state(
            has_spec=False, has_checkpoint=False, manifest=None, progress=None, now=now, unreadable=MANIFEST in files
        )
        return state, None
    summary = summarise_run(store, run_id, now=now)
    assert summary is not None
    return RunState(summary.status), summary


def require(store: ArtifactStore, run_id: str, event: Event, *, now: float | None = None) -> RunState:
    now = time.time() if now is None else now
    state, summary = current_state(store, run_id, now=now)
    resumable = summary is not None and summary.resumable
    if allows(state, event) and (event is not Event.RESUME or resumable):
        return state
    raise TransitionError(_refusal(run_id, event, state, summary, now))


def _refusal(run_id: str, event: Event, state: RunState, summary: RunSummary | None, now: float) -> str:
    what = f"{state} ({summary.error})" if summary is not None and summary.error else str(state)
    if state in LIVE:
        age = now - summary.updated if summary is not None and summary.updated is not None else 0.0
        live = f"run {run_id} is {state} (heartbeat {age:.0f}s ago)"
        if event is Event.RUN:
            return f"{live}: pick a new --attempt"
        return f"{live}: resuming it now would race the live process"
    if event is Event.RUN:
        if state is RunState.ABSENT:
            return f"run {run_id}: starting it is not allowed"
        hint = "pick a new --attempt"
        if summary is not None and summary.resumable and allows(state, Event.RESUME):
            hint += f", or continue it with `angku resume {run_id}`"
        return f"run {run_id} already exists ({what}): {hint}"
    if state is RunState.FINISHED:
        return f"run {run_id} already FINISHED; nothing to resume"
    if state is RunState.UNREADABLE:
        return f"run {run_id} is {what}; nothing to resume"
    if summary is not None and summary.resumable:
        return f"run {run_id} is {state}: resuming it is not allowed"
    return (
        f"no checkpoint for run {run_id}: nothing to resume (only runs started with --checkpoint-every can be resumed)"
    )
