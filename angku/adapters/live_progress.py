"""Tracker wrapper that keeps ``progress.json`` current while a run trains.

Wraps whatever tracker the CLI hands a launcher, so local and Ray runs (the Ray driver's
MetricsRelay is the tracker's only caller) both report progress without the runner knowing.
Every call is forwarded unchanged first. ``start_run`` also writes ``spec.json``, so a run is
listed from its first moment; a heartbeat thread refreshes ``updated`` between epochs, which can
be long, so "stale" means the process is gone rather than the epoch is slow.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping

from angku.application.bundle import save_spec
from angku.application.progress import HEARTBEAT, Progress, save_progress
from angku.domain.experiment import ExperimentSpec
from angku.ports.artifacts import ArtifactStore
from angku.ports.tracking import RunStatus, Tracker


class LiveProgress:
    def __init__(
        self,
        inner: Tracker,
        store: ArtifactStore,
        *,
        clock: Callable[[], float] = time.time,
        heartbeat: float = HEARTBEAT,
    ) -> None:
        self._inner = inner
        self._store = store
        self._clock = clock
        self._heartbeat = heartbeat
        self._lock = threading.Lock()  # one writer at a time: the run's thread or the heartbeat
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._p: Progress | None = None

    def start_run(self, *, run_id: str, experiment_id: str, spec: ExperimentSpec, resume: bool = False) -> None:
        self._inner.start_run(run_id=run_id, experiment_id=experiment_id, spec=spec, resume=resume)
        save_spec(self._store, run_id, spec)
        now = self._clock()
        self._write(
            Progress(
                run_id=run_id,
                state="running",
                epoch=0,
                epochs=spec.training.epochs,
                started=now,
                updated=now,
                history=[],
            )
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, name="angku-progress", daemon=True)
        self._thread.start()

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        self._inner.log_metrics(metrics, step=step)
        self._update(epoch=step, history=[*self._current().history, dict(metrics)])

    def end_run(self, status: RunStatus) -> None:
        self._inner.end_run(status)
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._update(state="ended")

    def _current(self) -> Progress:
        assert self._p is not None, "log_metrics/end_run before start_run"
        return self._p

    def _beat(self) -> None:
        while not self._stop.wait(self._heartbeat):
            self._update()

    def _update(self, **changes: object) -> None:
        with self._lock:
            self._p = self._current().model_copy(update={**changes, "updated": self._clock()})
            save_progress(self._store, self._p)

    def _write(self, p: Progress) -> None:
        with self._lock:
            self._p = p
            save_progress(self._store, p)
