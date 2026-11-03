"""Tracker backed by MLflow (``tracking.backend: mlflow`` in the yaml).

Talks to ``MlflowClient`` directly instead of the fluent ``mlflow.*`` API so
no global state leaks between runs or tests. Our own ids are the source of
truth: they are stored as tags (``angku.run_id``, ``angku.experiment_id``)
and the spec fingerprint is flattened into params so runs can be compared
in the UI. MLflow's own run id is an implementation detail kept private.

Requires the ``tracking`` extra; importing this module imports mlflow, so
the core never imports it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag

from angku.domain.experiment import ExperimentSpec
from angku.ports.tracking import RunStatus, TrackerError

BACKEND = "mlflow"


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """``{"a": {"b": 1}, "c": [x, y]}`` -> ``{"a.b": "1", "c.0": ..., "c.1": ...}``."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten(v, f"{key}."))
        elif isinstance(v, list):
            out.update(flatten({str(i): x for i, x in enumerate(v)}, f"{key}."))
        else:
            out[key] = str(v)
    return out


class MlflowTracker:
    def __init__(self, uri: str | None = None) -> None:
        # Explicit uri wins over spec.tracking.uri; one of them must be set.
        self._uri = uri
        self._client: MlflowClient | None = None
        self._mlflow_run_id: str | None = None

    def start_run(self, *, run_id: str, experiment_id: str, spec: ExperimentSpec, resume: bool = False) -> None:
        if self._mlflow_run_id is not None:
            raise TrackerError("a run is still active; call end_run first")
        t = spec.tracking
        if t.backend != BACKEND:
            raise TrackerError(f"spec says tracking.backend={t.backend!r}, this tracker is {BACKEND!r}")
        uri = self._uri or t.uri
        if not uri:
            raise TrackerError("no MLflow uri: pass MlflowTracker(uri=...) or set tracking.uri in the spec")
        if not t.experiment_name:
            raise TrackerError("tracking.experiment_name must be set for MLflow")

        client = MlflowClient(tracking_uri=uri)
        exp = client.get_experiment_by_name(t.experiment_name)
        mlflow_exp_id = exp.experiment_id if exp is not None else client.create_experiment(t.experiment_name)

        if resume:
            earlier = client.search_runs(
                [mlflow_exp_id],
                filter_string=f"tags.`angku.run_id` = '{run_id}'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if earlier:
                # Params are already there (same spec); just reopen it for the remaining epochs.
                client.update_run(earlier[0].info.run_id, status="RUNNING")
                self._client, self._mlflow_run_id = client, earlier[0].info.run_id
                return

        run = client.create_run(
            mlflow_exp_id,
            run_name=run_id[:12],
            tags={
                "angku.run_id": run_id,
                "angku.experiment_id": experiment_id,
                "angku.name": spec.meta.name,
            },
        )
        params = flatten(spec.fingerprint())
        client.log_batch(
            run.info.run_id,
            params=[Param(k, v[:6000]) for k, v in params.items()],
            tags=[RunTag(f"angku.tag.{tag}", "true") for tag in spec.meta.tags],
        )
        self._client, self._mlflow_run_id = client, run.info.run_id

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        client, run_id = self._active()
        ts = int(time.time() * 1000)
        client.log_batch(run_id, metrics=[Metric(k, float(v), ts, step) for k, v in metrics.items()])

    def end_run(self, status: RunStatus) -> None:
        client, run_id = self._active()
        client.set_terminated(run_id, status=status)
        self._client, self._mlflow_run_id = None, None

    def _active(self) -> tuple[MlflowClient, str]:
        if self._client is None or self._mlflow_run_id is None:
            raise TrackerError("no active run; call start_run first")
        return self._client, self._mlflow_run_id
