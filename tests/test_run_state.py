from __future__ import annotations

import pytest

from angku.domain.manifest import RunManifest
from angku.domain.progress import STALE_AFTER, Progress
from angku.domain.run_state import RunState, derive_state, is_resumable

NOW = 1000.0


def _manifest(status: str) -> RunManifest:
    return RunManifest(experiment_id="e" * 64, run_id="r" * 64, attempt=1, status=status, device="cpu")


def _progress(state: str = "running", updated: float = NOW) -> Progress:
    return Progress(run_id="r" * 64, state=state, epoch=1, epochs=3, started=900.0, updated=updated, history=[])


LIVE = _progress()
STALE = _progress(updated=NOW - STALE_AFTER - 1)
ENDED = _progress(state="ended")


@pytest.mark.parametrize(
    ("has_spec", "has_checkpoint", "manifest", "progress", "expected"),
    [
        (False, False, None, None, RunState.ABSENT),
        (True, False, None, None, RunState.INCOMPLETE),
        (True, True, None, None, RunState.INTERRUPTED),
        (True, False, None, LIVE, RunState.RUNNING),
        (True, True, None, LIVE, RunState.RUNNING),
        (True, False, None, STALE, RunState.INCOMPLETE),
        (True, True, None, STALE, RunState.INTERRUPTED),
        (True, True, None, ENDED, RunState.INTERRUPTED),
        (True, False, _manifest("FINISHED"), None, RunState.FINISHED),
        (True, True, _manifest("FINISHED"), LIVE, RunState.FINISHED),
        (True, False, _manifest("FAILED"), None, RunState.FAILED),
        (True, True, _manifest("FAILED"), STALE, RunState.FAILED),
        (True, True, _manifest("FAILED"), LIVE, RunState.RESUMING),
        (False, False, _manifest("FINISHED"), None, RunState.UNREADABLE),
    ],
)
def test_derive_state(has_spec, has_checkpoint, manifest, progress, expected):
    state = derive_state(
        has_spec=has_spec, has_checkpoint=has_checkpoint, manifest=manifest, progress=progress, now=NOW
    )
    assert state is expected


def test_unreadable_wins():
    state = derive_state(has_spec=True, has_checkpoint=True, manifest=None, progress=LIVE, now=NOW, unreadable=True)
    assert state is RunState.UNREADABLE


def test_heartbeat_exactly_stale_after_old_is_still_live():
    p = _progress(updated=NOW - STALE_AFTER)
    assert derive_state(has_spec=True, has_checkpoint=False, manifest=None, progress=p, now=NOW) is RunState.RUNNING


@pytest.mark.parametrize("state", list(RunState))
@pytest.mark.parametrize("has_checkpoint", [False, True])
def test_only_failed_or_interrupted_runs_with_a_checkpoint_are_resumable(state, has_checkpoint):
    expected = has_checkpoint and state in (RunState.FAILED, RunState.INTERRUPTED)
    assert is_resumable(state, has_checkpoint=has_checkpoint) is expected


def test_state_values_are_what_listing_prints():
    assert [s.value for s in RunState] == [
        "ABSENT",
        "RUNNING",
        "RESUMING",
        "FINISHED",
        "FAILED",
        "INTERRUPTED",
        "INCOMPLETE",
        "UNREADABLE",
    ]
