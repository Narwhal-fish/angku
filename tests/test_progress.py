"""Live progress of a run that is still training (roadmap 13, phase C).

``progress.json`` sits next to the bundle but is not part of it: derived, in no id, never read by
``reproduce``. ``LiveProgress`` (adapter) wraps the tracker the CLI hands a launcher, so local and
Ray runs both get it without touching the runner:

    start_run      writes spec.json (the run is listed at once) + progress {state running, epoch 0}
    log_metrics    epoch = step, history += metrics
    heartbeat      a thread refreshes ``updated`` every ``heartbeat`` seconds
    end_run        state ended; the CLI deletes the file once the launch returned

A run is RUNNING when it has no manifest and its progress is ``running`` and fresher than
``STALE_AFTER``; a killed run goes stale and lists as INTERRUPTED / INCOMPLETE as before. A RUNNING
run is never resumable: ``mlops resume`` refuses it, so two processes never write one run.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import yaml

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.live_progress import LiveProgress
from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.application.bundle import CHECKPOINT, MANIFEST, SPEC, save_spec
from mlops.application.listing import list_runs
from mlops.application.progress import PROGRESS, STALE_AFTER, Progress, is_live, load_progress, save_progress
from mlops.cli import main
from mlops.config import load_experiment_yaml
from mlops.domain import compute_experiment_id, compute_run_id

ROOT = Path(__file__).parent.parent
SMOKE = ROOT / "experiments/configs/tiny_cnn_smoke.yaml"


@pytest.fixture
def spec():
    return load_experiment_yaml(SMOKE)


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _progress(**kw) -> Progress:
    base = dict(
        run_id="r" * 64,
        state="running",
        epoch=2,
        epochs=5,
        started=900.0,
        updated=1000.0,
        history=[{"val/loss": 0.5}, {"val/loss": 0.4}],
    )
    return Progress(**{**base, **kw})


# --- model ---


def test_is_live():
    p = _progress()
    assert is_live(p, now=1000.0 + STALE_AFTER)
    assert not is_live(p, now=1000.0 + STALE_AFTER + 1)
    assert not is_live(_progress(state="ended"), now=1000.0)


def test_progress_rejects_typos():
    with pytest.raises(ValueError):
        Progress.model_validate({**_progress().model_dump(), "epochz": 1})


# --- LiveProgress ---


def test_live_progress_forwards_and_records(spec):
    store, inner, clock = InMemoryArtifactStore(), InMemoryTracker(), Clock()
    tracker = LiveProgress(inner, store, clock=clock, heartbeat=3600)
    tracker.start_run(run_id="r" * 64, experiment_id="e" * 64, spec=spec)
    assert store.get("r" * 64, SPEC)  # listed from the first moment
    p = load_progress(store, "r" * 64)
    assert (p.state, p.epoch, p.epochs, p.history, p.started) == ("running", 0, spec.training.epochs, [], 1000.0)

    clock.t = 1010.0
    tracker.log_metrics({"val/loss": 0.5}, step=1)
    p = load_progress(store, "r" * 64)
    assert (p.epoch, p.history, p.updated) == (1, [{"val/loss": 0.5}], 1010.0)

    tracker.end_run("FINISHED")
    assert load_progress(store, "r" * 64).state == "ended"
    assert inner.runs[0].metrics == [(1, {"val/loss": 0.5})] and inner.runs[0].status == "FINISHED"


def test_heartbeat_refreshes_updated(spec):
    store, clock = InMemoryArtifactStore(), Clock()
    beat = threading.Event()

    class Store(InMemoryArtifactStore):
        def put(self, run_id, name, data):
            super().put(run_id, name, data)
            if name == PROGRESS and json.loads(data)["updated"] == 2000.0:
                beat.set()

    store = Store()
    tracker = LiveProgress(InMemoryTracker(), store, clock=clock, heartbeat=0.01)
    tracker.start_run(run_id="r" * 64, experiment_id="e" * 64, spec=spec)
    clock.t = 2000.0
    assert beat.wait(5)
    tracker.end_run("FAILED")
    assert load_progress(store, "r" * 64).state == "ended"


# --- listing ---


def _stored(spec, *, progress: Progress | None, checkpoint: bool = False) -> tuple[InMemoryArtifactStore, str]:
    store = InMemoryArtifactStore()
    run_id = compute_run_id(compute_experiment_id(spec), spec.resources, 1)
    save_spec(store, run_id, spec)
    if progress is not None:
        save_progress(store, progress.model_copy(update={"run_id": run_id}))
    if checkpoint:
        store.put(run_id, CHECKPOINT, b"x")
    return store, run_id


def test_a_live_run_lists_as_running(spec):
    store, _ = _stored(spec, progress=_progress(), checkpoint=True)
    (s,) = list_runs(store, now=1010.0)
    assert s.status == "RUNNING"
    assert (s.epochs_done, s.final_metrics, s.resumable) == (2, {"val/loss": 0.4}, False)


def test_a_stale_run_falls_back(spec):
    store, _ = _stored(spec, progress=_progress(), checkpoint=True)
    assert list_runs(store, now=1000.0 + STALE_AFTER + 1)[0].status == "INTERRUPTED"
    store, _ = _stored(spec, progress=_progress(state="ended"))
    assert list_runs(store, now=1000.0)[0].status == "INCOMPLETE"


def test_manifest_wins_over_progress(tmp_path):
    art = tmp_path / "artifacts"
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(art)]) == 0
    (run_dir,) = art.iterdir()
    assert not (run_dir / PROGRESS).exists()  # the CLI removed it after the launch
    store = LocalArtifactStore(art)
    save_progress(store, _progress(run_id=run_dir.name))  # a leftover from a crash after the manifest
    assert list_runs(store, now=1000.0)[0].status == "FINISHED"


# --- the CLI wires it in ---


def test_cli_run_writes_progress_while_training(tmp_path, monkeypatch):
    """During training the run dir holds spec.json + a running progress.json."""
    seen = []
    from mlops.adapters import inmemory_tracker

    real = inmemory_tracker.InMemoryTracker.log_metrics

    def spy(self, metrics, *, step):
        real(self, metrics, step=step)
        seen.append({p.name for p in (tmp_path / "artifacts").glob("*/*")})

    monkeypatch.setattr(inmemory_tracker.InMemoryTracker, "log_metrics", spy)
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(tmp_path / "artifacts")]) == 0
    assert seen and {SPEC, PROGRESS} <= seen[0] and MANIFEST not in seen[0]


def test_cli_without_artifacts_writes_no_progress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(SMOKE), "--tracker", "inmemory"]) == 0
    assert list(tmp_path.iterdir()) == []


def test_resume_refuses_a_running_run(tmp_path, capsys, spec):
    art = tmp_path / "artifacts"
    data = yaml.safe_load(SMOKE.read_text())
    data["training"]["epochs"] = 2
    (tmp_path / "two.yaml").write_text(yaml.safe_dump(data))
    assert (
        main(
            [
                "run",
                str(tmp_path / "two.yaml"),
                "--tracker",
                "inmemory",
                "--artifacts",
                str(art),
                "--checkpoint-every",
                "1",
                "--keep-checkpoint",
            ]
        )
        == 0
    )
    (run_dir,) = art.iterdir()
    (run_dir / MANIFEST).unlink()
    import time

    save_progress(LocalArtifactStore(art), _progress(run_id=run_dir.name, updated=time.time()))
    capsys.readouterr()
    assert main(["resume", run_dir.name, "--artifacts", str(art), "--tracker", "inmemory"]) == 2
    assert "RUNNING" in capsys.readouterr().err
