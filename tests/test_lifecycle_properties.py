from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, event, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule  # noqa: E402

from angku.adapters.inmemory_artifacts import InMemoryArtifactStore  # noqa: E402
from angku.adapters.inmemory_tracker import InMemoryTracker, RecordedRun  # noqa: E402
from angku.adapters.live_progress import LiveProgress  # noqa: E402
from angku.adapters.synthetic_data import SyntheticBlobs  # noqa: E402
from angku.application import run_experiment  # noqa: E402
from angku.application.bundle import BEST, CHECKPOINT, MANIFEST, MODEL, SPEC  # noqa: E402
from angku.application.gc import GcPolicy, apply_gc, plan_gc  # noqa: E402
from angku.application.lifecycle import (  # noqa: E402
    ALLOWED_TRANSITIONS,
    Event,
    TransitionError,
    allows,
    current_state,
)
from angku.application.listing import summarise_run  # noqa: E402
from angku.application.progress import PROGRESS  # noqa: E402
from angku.config import load_experiment_dict, load_experiment_yaml  # noqa: E402
from angku.domain import compute_experiment_id, compute_run_id, parse_manifest  # noqa: E402
from angku.domain.best import BestCriterion  # noqa: E402
from angku.domain.run_state import RunState  # noqa: E402
from angku.registry import discover  # noqa: E402

SMOKE = Path(__file__).parent.parent / "experiments/configs/tiny_cnn_smoke.yaml"
EPOCHS = 3


class Killed(BaseException):
    pass


class TrainingBlewUp(RuntimeError):
    pass


class Process:
    def __init__(self, fault: tuple | None) -> None:
        self.fault = fault
        self.dead = False

    def check(self) -> None:
        if self.dead:
            raise Killed

    def write(self, name: str) -> None:
        self.check()
        kind, arg = self.fault or (None, None)
        if kind == "kill" and name == arg:
            self.dead = True
            raise Killed
        if kind == "disk_full" and name == arg:
            raise OSError("No space left on device")


class ProcessStore:
    def __init__(self, disk: InMemoryArtifactStore, process: Process) -> None:
        self.disk, self.process = disk, process

    def put(self, run_id, name, data):
        self.process.write(name)
        self.disk.put(run_id, name, data)

    def delete(self, run_id, name):
        self.process.check()
        self.disk.delete(run_id, name)

    def get(self, run_id, name):
        return self.disk.get(run_id, name)

    def list(self, run_id):
        return self.disk.list(run_id)

    def runs(self):
        return self.disk.runs()


class ProcessTracker(InMemoryTracker):
    def __init__(self, server: list[RecordedRun], process: Process) -> None:
        super().__init__()
        self.runs, self.process = server, process

    def _alive(self) -> None:
        if self.process.dead:
            raise Killed

    def start_run(self, **kw):
        self._alive()
        super().start_run(**kw)

    def log_metrics(self, metrics, *, step):
        self._alive()
        if self.process.fault == ("error", step):
            raise TrainingBlewUp(f"epoch {step}")
        super().log_metrics(metrics, step=step)

    def end_run(self, status):
        self._alive()
        super().end_run(status)


class NoHeartbeat(LiveProgress):
    def _beat(self) -> None:
        return None


FILES = [SPEC, PROGRESS, CHECKPOINT, MODEL, BEST, MANIFEST]
faults = st.one_of(
    st.none(),
    st.tuples(st.just("error"), st.integers(1, EPOCHS)),
    st.tuples(st.sampled_from(["disk_full", "kill"]), st.sampled_from(FILES)),
)


def _spec():
    base = load_experiment_yaml(SMOKE)
    return load_experiment_dict({**base.model_dump(), "training": {**base.training.model_dump(), "epochs": EPOCHS}})


def _end_states(source: RunState, event: Event) -> set[RunState]:
    middle = {t.target for t in ALLOWED_TRANSITIONS if (t.source, t.event) == (source, event)}
    ends = {Event.FINISH, Event.FAIL, Event.KILL}
    return {t.target for t in ALLOWED_TRANSITIONS if t.source in middle and t.event in ends}


class Lifecycle(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.disk = InMemoryArtifactStore()
        self.server: list[RecordedRun] = []
        self.spec = _spec()
        self.finished: dict[str, dict[str, bytes]] = {}

    def _run_id(self, attempt: int) -> str:
        return compute_run_id(compute_experiment_id(self.spec), self.spec.resources, attempt)

    def _snapshot(self, run_id: str) -> dict[str, bytes]:
        if run_id not in self.disk.runs():
            return {}
        return {n: self.disk.get(run_id, n) for n in self.disk.list(run_id)}

    def _launch(self, attempt: int, fault, kind: Event, **kw) -> None:
        run_id = self._run_id(attempt)
        before, files = current_state(self.disk, run_id)[0], self._snapshot(run_id)
        process = Process(fault)
        store = ProcessStore(self.disk, process)
        tracker = NoHeartbeat(ProcessTracker(self.server, process), store, clock=lambda: 0.0)
        try:
            run_experiment(
                self.spec,
                data=SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2),
                tracker=tracker,
                artifacts=store,
                attempt=attempt,
                resume=kind is Event.RESUME,
                **kw,
            )
        except TransitionError:
            event(f"{kind} refused from {before}")
            assert not allows(before, kind) or kind is Event.RESUME
            assert self._snapshot(run_id) == files
            return
        except (Killed, OSError, TrainingBlewUp):
            pass
        if not process.dead and PROGRESS in self._snapshot(run_id):
            self.disk.delete(run_id, PROGRESS)
        after = current_state(self.disk, run_id)[0]
        event(f"{kind} {before} -> {after}")
        assert allows(before, kind), (before, kind)
        assert after in _end_states(before, kind) | {before}, (before, kind, after)

    @initialize(
        start=st.sampled_from([None, ("kill", MANIFEST), ("error", EPOCHS)]),
        attempt=st.integers(1, 2),
    )
    def start(self, start, attempt):
        if start is not None:
            self._launch(attempt, start, Event.RUN, checkpoint_every=2)

    @rule(
        attempt=st.integers(1, 2),
        fault=faults,
        checkpoint_every=st.sampled_from([None, 1, 2]),
        keep_best=st.booleans(),
        keep_checkpoint=st.booleans(),
    )
    def run(self, attempt, fault, checkpoint_every, keep_best, keep_checkpoint):
        self._launch(
            attempt,
            fault,
            Event.RUN,
            checkpoint_every=checkpoint_every,
            keep_best=BestCriterion(metric="val/loss", mode="min") if keep_best else None,
            keep_checkpoint=keep_checkpoint,
        )

    @rule(attempt=st.integers(1, 2), fault=faults)
    def resume(self, attempt, fault):
        self._launch(attempt, fault, Event.RESUME)

    def _resumable(self) -> list[int]:
        return [a for a in (1, 2) if (s := summarise_run(self.disk, self._run_id(a))) is not None and s.resumable]

    @precondition(lambda self: self._resumable())
    @rule(data=st.data(), fault=faults)
    def resume_a_resumable_run(self, data, fault):
        self._launch(data.draw(st.sampled_from(self._resumable())), fault, Event.RESUME)

    @rule(resumable=st.booleans())
    def gc(self, resumable):
        apply_gc(self.disk, plan_gc(self.disk, GcPolicy(resumable=resumable)))

    def _bundles(self):
        for run_id in self.disk.runs():
            files = self.disk.list(run_id)
            manifest = parse_manifest(self.disk.get(run_id, MANIFEST)) if MANIFEST in files else None
            tracked = [r for r in self.server if r.run_id == run_id]
            yield files, manifest, tracked[-1].status if tracked else None

    @invariant()
    def a_failed_bundle_holds_no_weights(self):
        for files, manifest, _ in self._bundles():
            if manifest is not None and manifest.status == "FAILED":
                assert MODEL not in files and BEST not in files

    @invariant()
    def a_finished_bundle_has_its_manifest_and_weights(self):
        for files, manifest, tracked in self._bundles():
            if tracked == "FINISHED":
                assert manifest is not None and manifest.status == "FINISHED"
            if manifest is not None and manifest.status == "FINISHED":
                assert MODEL in files

    @invariant()
    def a_finished_bundle_never_changes(self):
        for run_id in self.disk.runs():
            files = self.disk.list(run_id)
            if MANIFEST not in files or parse_manifest(self.disk.get(run_id, MANIFEST)).status != "FINISHED":
                assert run_id not in self.finished
                continue
            bundle = {n: self.disk.get(run_id, n) for n in files if n not in (CHECKPOINT, PROGRESS)}
            assert self.finished.setdefault(run_id, bundle) == bundle

    @invariant()
    def tracker_and_manifest_agree(self):
        for _, manifest, tracked in self._bundles():
            if manifest is not None and tracked in ("FINISHED", "FAILED"):
                assert manifest.status == tracked


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


TestLifecycle = Lifecycle.TestCase
TestLifecycle.settings = settings(
    max_examples=1000,
    stateful_step_count=8,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
