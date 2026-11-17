from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from angku.adapters.inmemory_artifacts import InMemoryArtifactStore
from angku.adapters.inmemory_tracker import InMemoryTracker
from angku.adapters.synthetic_data import SyntheticBlobs
from angku.application import RunnerError, lifecycle, run_experiment
from angku.application.bundle import CHECKPOINT, MANIFEST, MODEL, SPEC, save_spec
from angku.application.lifecycle import ALLOWED_TRANSITIONS, Event, TransitionError, current_state, require
from angku.application.progress import Progress, save_progress
from angku.cli import main
from angku.config import load_experiment_dict, load_experiment_yaml
from angku.domain import RunManifest, compute_experiment_id, compute_run_id
from angku.domain.run_state import LIVE, RunState
from angku.registry import discover

ROOT = Path(__file__).parent.parent
SMOKE = ROOT / "experiments/configs/tiny_cnn_smoke.yaml"


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


@pytest.fixture
def spec():
    base = load_experiment_yaml(SMOKE)
    return load_experiment_dict({**base.model_dump(), "training": {**base.training.model_dump(), "epochs": 2}})


def _data() -> SyntheticBlobs:
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)


def _run_id(spec, attempt=1) -> str:
    return compute_run_id(compute_experiment_id(spec), spec.resources, attempt)


def _targets(source: RunState, event: Event) -> set[RunState]:
    return {t.target for t in ALLOWED_TRANSITIONS if (t.source, t.event) == (source, event)}


def test_a_new_run_starts_only_from_nothing():
    assert {t.source for t in ALLOWED_TRANSITIONS if t.event is Event.RUN} == {RunState.ABSENT}
    assert _targets(RunState.ABSENT, Event.RUN) == {RunState.RUNNING}


def test_only_failed_or_interrupted_runs_resume():
    assert {t.source for t in ALLOWED_TRANSITIONS if t.event is Event.RESUME} == {
        RunState.FAILED,
        RunState.INTERRUPTED,
    }
    assert _targets(RunState.FAILED, Event.RESUME) == {RunState.RESUMING}
    assert _targets(RunState.INTERRUPTED, Event.RESUME) == {RunState.RUNNING}


def test_live_and_unreadable_runs_take_no_command():
    commands = {Event.RUN, Event.RESUME, Event.GC}
    for t in ALLOWED_TRANSITIONS:
        if t.event in commands:
            assert t.source not in LIVE | {RunState.UNREADABLE}, t


def test_a_finished_run_only_leaves_through_gc():
    assert {(t.event, t.target) for t in ALLOWED_TRANSITIONS if t.source is RunState.FINISHED} == {
        (Event.GC, RunState.ABSENT)
    }


def _store_in(state: RunState, spec, now: float) -> tuple[InMemoryArtifactStore, str]:
    store, run_id = InMemoryArtifactStore(), _run_id(spec)
    if state is RunState.ABSENT:
        return store, run_id
    if state is RunState.UNREADABLE:
        store.put(run_id, SPEC, b"{not json")
        return store, run_id
    save_spec(store, run_id, spec)
    status = {RunState.FINISHED: "FINISHED", RunState.FAILED: "FAILED", RunState.RESUMING: "FAILED"}.get(state)
    if status is not None:
        m = RunManifest(
            experiment_id=compute_experiment_id(spec), run_id=run_id, attempt=1, status=status, device="cpu"
        )
        store.put(run_id, MANIFEST, m.model_dump_json().encode())
    if state in (RunState.FAILED, RunState.INTERRUPTED, RunState.RESUMING):
        store.put(run_id, CHECKPOINT, b"x")
    if state in LIVE:
        p = Progress(run_id=run_id, state="running", epoch=1, epochs=2, started=now, updated=now - 4, history=[])
        save_progress(store, p)
    return store, run_id


@pytest.mark.parametrize("state", list(RunState))
def test_current_state_is_the_derived_state(spec, state):
    store, run_id = _store_in(state, spec, now=1000.0)
    assert current_state(store, run_id, now=1000.0)[0] is state


@pytest.mark.parametrize("state", list(RunState))
@pytest.mark.parametrize("event", [Event.RUN, Event.RESUME])
def test_require_follows_the_table(spec, state, event):
    store, run_id = _store_in(state, spec, now=1000.0)
    if _targets(state, event):
        require(store, run_id, event, now=1000.0)
    else:
        with pytest.raises(TransitionError):
            require(store, run_id, event, now=1000.0)


def test_failed_run_without_checkpoint_is_not_resumable(spec):
    store, run_id = _store_in(RunState.FAILED, spec, now=1000.0)
    store.delete(run_id, CHECKPOINT)
    with pytest.raises(TransitionError, match="no checkpoint"):
        require(store, run_id, Event.RESUME, now=1000.0)


@pytest.mark.parametrize(
    ("state", "event", "words"),
    [
        (RunState.FINISHED, Event.RUN, ["already exists (FINISHED)", "--attempt"]),
        (RunState.FAILED, Event.RUN, ["--attempt", "angku resume"]),
        (RunState.INCOMPLETE, Event.RUN, ["--attempt"]),
        (RunState.RUNNING, Event.RUN, ["is RUNNING (heartbeat 4s ago)", "--attempt"]),
        (RunState.FINISHED, Event.RESUME, ["already FINISHED; nothing to resume"]),
        (RunState.RESUMING, Event.RESUME, ["is RESUMING (heartbeat 4s ago)", "race the live process"]),
        (RunState.ABSENT, Event.RESUME, ["no checkpoint"]),
        (RunState.UNREADABLE, Event.RESUME, ["UNREADABLE (", "nothing to resume"]),
        (RunState.UNREADABLE, Event.RUN, ["already exists (UNREADABLE (", "--attempt"]),
    ],
)
def test_refusals_say_why(spec, state, event, words):
    store, run_id = _store_in(state, spec, now=1000.0)
    with pytest.raises(TransitionError) as e:
        require(store, run_id, event, now=1000.0)
    assert run_id in str(e.value) and all(w in str(e.value) for w in words)
    if "angku resume" not in words:
        assert "angku resume" not in str(e.value)


def test_transition_error_is_a_runner_error():
    assert issubclass(TransitionError, RunnerError)


def test_run_experiment_refuses_a_second_run_of_a_finished_run(spec):
    store = InMemoryArtifactStore()
    run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store)
    run_id = _run_id(spec)
    before = {n: store.get(run_id, n) for n in store.list(run_id)}
    tracker = InMemoryTracker()
    with pytest.raises(TransitionError, match="already exists"):
        run_experiment(spec, data=_data(), tracker=tracker, artifacts=store)
    assert tracker.runs == []
    assert {n: store.get(run_id, n) for n in store.list(run_id)} == before


def test_run_experiment_refuses_to_resume_a_run_being_resumed(spec):
    store, run_id = _store_in(RunState.RESUMING, spec, now=time.time())
    tracker = InMemoryTracker()
    with pytest.raises(TransitionError, match="RESUMING"):
        run_experiment(spec, data=_data(), tracker=tracker, artifacts=store, resume=True)
    assert tracker.runs == []


def _finished_cli_run(tmp_path, capsys) -> tuple[Path, str]:
    art = tmp_path / "artifacts"
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(art), "--json"]) == 0
    return art, json.loads(capsys.readouterr().out)["run_id"]


def test_widening_the_table_lets_a_finished_run_run_again(tmp_path, capsys, monkeypatch):
    art, run_id = _finished_cli_run(tmp_path, capsys)
    extra = lifecycle.Transition(RunState.FINISHED, Event.RUN, RunState.RUNNING)
    monkeypatch.setattr(lifecycle, "ALLOWED_TRANSITIONS", ALLOWED_TRANSITIONS | {extra})
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(art)]) == 0


def test_an_empty_table_refuses_every_entry_point(tmp_path, capsys, monkeypatch, spec):
    from tests.test_cli import _killed_run

    killed_art, killed = _killed_run(tmp_path, capsys, monkeypatch)
    finished = InMemoryArtifactStore()
    run_experiment(
        spec, data=_data(), tracker=InMemoryTracker(), artifacts=finished, checkpoint_every=1, keep_checkpoint=True
    )
    finished.put(_run_id(spec), MANIFEST, finished.get(_run_id(spec), MANIFEST).replace(b'"FINISHED"', b'"FAILED"'))
    monkeypatch.setattr(lifecycle, "ALLOWED_TRANSITIONS", frozenset())
    with pytest.raises(TransitionError):
        run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=InMemoryArtifactStore())
    with pytest.raises(TransitionError):
        run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=finished, resume=True)
    art = tmp_path / "fresh"
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(art)]) == 2
    assert not art.exists()
    capsys.readouterr()
    assert main(["resume", killed, "--artifacts", str(killed_art), "--tracker", "inmemory"]) == 2
    assert "resuming it is not allowed" in capsys.readouterr().err


def _target_diagram() -> str:
    doc = (ROOT / "docs/run-lifecycle.md").read_text()
    section = doc.split("\n## Target lifecycle\n", 1)[1].split("\n## ", 1)[0]
    return section.split("```mermaid\n", 1)[1].split("```", 1)[0]


def _diagram_edges(diagram: str) -> set[tuple[str, str, str]]:
    edges = set()
    for line in diagram.splitlines():
        m = re.fullmatch(r"\s*(\w+)\s*-->\s*(\w+)\s*:\s*(\w+)\s*", line)
        if m:
            edges.add((m[1], m[3], m[2]))
    return edges


def test_target_diagram_is_the_allowed_transitions():
    expected = {(t.source.value, t.event.value, t.target.value) for t in ALLOWED_TRANSITIONS}
    assert _diagram_edges(_target_diagram()) == expected


def test_target_diagram_shows_every_state():
    diagram = _target_diagram()
    assert all(re.search(rf"^\s*{s.value}\b", diagram, re.M) for s in RunState)


def test_edges_are_parsed_from_mermaid_lines():
    assert _diagram_edges("    A --> B : run\n    [*] --> A\n    C --> D\n") == {("A", "run", "B")}


class _Killed(BaseException):
    pass


class _StoreFailingOn(InMemoryArtifactStore):
    def __init__(self, inner: InMemoryArtifactStore, name: str, error: BaseException) -> None:
        super().__init__()
        self._runs, self.name, self.error = inner._runs, name, error

    def put(self, run_id, name, data):
        if name == self.name:
            raise self.error
        super().put(run_id, name, data)


def _failed_at_last_epoch(spec, store) -> str:
    from tests.test_checkpoint import FlakyTracker

    with pytest.raises(Exception, match="killed at epoch 2"):
        run_experiment(spec, data=_data(), tracker=FlakyTracker(2), artifacts=store, checkpoint_every=1)
    return _run_id(spec)


def test_a_failed_resume_drops_weights_a_killed_segment_left(spec):
    from tests.test_checkpoint import FlakyTracker

    spec = spec.model_copy(update={"training": spec.training.model_copy(update={"epochs": 3})})
    store = InMemoryArtifactStore()
    with pytest.raises(_Killed):
        run_experiment(
            spec,
            data=_data(),
            tracker=InMemoryTracker(),
            artifacts=_StoreFailingOn(store, MANIFEST, _Killed()),
            checkpoint_every=2,
        )
    run_id = _run_id(spec)
    assert current_state(store, run_id)[0] is RunState.INTERRUPTED and MODEL in store.list(run_id)
    with pytest.raises(Exception, match="killed at epoch 3"):
        run_experiment(spec, data=_data(), tracker=FlakyTracker(3), artifacts=store, resume=True)
    assert current_state(store, run_id)[0] is RunState.FAILED
    assert MODEL not in store.list(run_id)


def test_a_resume_whose_bundle_write_fails_stops_claiming_failed(spec):
    store = InMemoryArtifactStore()
    run_id = _failed_at_last_epoch(spec, store)
    tracker = InMemoryTracker()
    with pytest.raises(OSError):
        run_experiment(
            spec,
            data=_data(),
            tracker=tracker,
            artifacts=_StoreFailingOn(store, MANIFEST, OSError("disk full")),
            resume=True,
        )
    assert MANIFEST not in store.list(run_id) and MODEL in store.list(run_id)
    assert current_state(store, run_id)[0] is RunState.INTERRUPTED
    assert tracker.runs[-1].status == "FAILED"
    run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, resume=True)
    assert current_state(store, run_id)[0] is RunState.FINISHED


def test_a_killed_bundle_write_of_a_resume_is_interrupted():
    assert _targets(RunState.RESUMING, Event.KILL) == {RunState.FAILED, RunState.INTERRUPTED}
