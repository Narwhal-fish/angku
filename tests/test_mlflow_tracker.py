"""MLflow adapter for the Tracker port.

Uses MlflowClient against a sqlite store in tmp_path (MLflow 3 deprecates the
file store), so nothing touches the global fluent state or ./mlruns.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
from mlflow import MlflowClient  # noqa: E402

from mlops.adapters.mlflow_tracker import MlflowTracker  # noqa: E402
from mlops.adapters.synthetic_data import SyntheticBlobs  # noqa: E402
from mlops.application import run_experiment  # noqa: E402
from mlops.config import load_experiment_dict, load_experiment_yaml  # noqa: E402
from mlops.ports.tracking import TrackerError  # noqa: E402
from mlops.registry import discover  # noqa: E402

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


@pytest.fixture
def spec():
    return load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")


def _sqlite(path: Path) -> str:
    return f"sqlite:///{path}"


@pytest.fixture(scope="module")
def _template_db(tmp_path_factory):
    # Schema migration costs ~3s; run it once and copy the file per test.
    path = tmp_path_factory.mktemp("mlflow") / "template.db"
    MlflowClient(_sqlite(path)).search_experiments()
    return path


@pytest.fixture
def fresh_db(_template_db, tmp_path):
    def make(name: str = "mlflow.db") -> str:
        dst = tmp_path / name
        shutil.copy(_template_db, dst)
        return _sqlite(dst)

    return make


@pytest.fixture
def uri(fresh_db):
    return fresh_db()


def _find(client: MlflowClient, experiment_name: str, run_id: str):
    exp = client.get_experiment_by_name(experiment_name)
    assert exp is not None, f"experiment {experiment_name!r} not created"
    runs = client.search_runs([exp.experiment_id], filter_string=f"tags.mlops.run_id = '{run_id}'")
    assert len(runs) == 1
    return runs[0]


def test_run_lifecycle_lands_in_mlflow(spec, uri):
    tracker = MlflowTracker(uri=uri)
    tracker.start_run(run_id="r" * 64, experiment_id="e" * 64, spec=spec)
    tracker.log_metrics({"train/loss": 1.5, "val/loss": 2.0}, step=1)
    tracker.log_metrics({"train/loss": 1.0}, step=2)
    tracker.end_run("FINISHED")

    run = _find(MlflowClient(uri), spec.tracking.experiment_name, "r" * 64)
    assert run.info.status == "FINISHED"
    assert run.data.tags["mlops.experiment_id"] == "e" * 64
    assert run.data.tags["mlops.run_id"] == "r" * 64
    assert run.data.tags["mlops.name"] == spec.meta.name
    # fingerprint flattened into params so the UI can filter/compare on them
    assert run.data.params["model.name"] == "tiny_cnn"
    assert run.data.params["model.params.hidden"] == "4"
    assert run.data.params["training.seed"] == "0"
    assert run.data.params["dataset.snapshot"] == spec.dataset.snapshot
    assert run.data.params["code.commit"] == spec.code.commit
    assert run.data.metrics == {"train/loss": 1.0, "val/loss": 2.0}
    history = MlflowClient(uri).get_metric_history(run.info.run_id, "train/loss")
    assert [(m.step, m.value) for m in history] == [(1, 1.5), (2, 1.0)]


def test_failed_status(spec, uri):
    tracker = MlflowTracker(uri=uri)
    tracker.start_run(run_id="r1", experiment_id="e1", spec=spec)
    tracker.end_run("FAILED")
    assert _find(MlflowClient(uri), "smoke", "r1").info.status == "FAILED"


def test_uri_comes_from_spec_when_not_given(spec, uri):
    spec = spec.model_copy(update={"tracking": spec.tracking.model_copy(update={"uri": uri})})
    tracker = MlflowTracker()
    tracker.start_run(run_id="r1", experiment_id="e1", spec=spec)
    tracker.end_run("FINISHED")
    _find(MlflowClient(uri), "smoke", "r1")


def test_constructor_uri_wins_over_spec(spec, uri, fresh_db):
    other = fresh_db("other.db")
    spec = spec.model_copy(update={"tracking": spec.tracking.model_copy(update={"uri": other})})
    tracker = MlflowTracker(uri=uri)
    tracker.start_run(run_id="r1", experiment_id="e1", spec=spec)
    tracker.end_run("FINISHED")
    _find(MlflowClient(uri), "smoke", "r1")
    assert MlflowClient(other).get_experiment_by_name("smoke") is None


def test_no_uri_anywhere_is_refused(spec):
    # never silently write ./mlruns in the cwd
    assert spec.tracking.uri == ""
    with pytest.raises(TrackerError, match="tracking.uri"):
        MlflowTracker().start_run(run_id="r1", experiment_id="e1", spec=spec)


def test_empty_experiment_name_is_refused(spec, uri):
    spec = spec.model_copy(update={"tracking": spec.tracking.model_copy(update={"experiment_name": ""})})
    with pytest.raises(TrackerError, match="experiment_name"):
        MlflowTracker(uri=uri).start_run(run_id="r1", experiment_id="e1", spec=spec)


def test_wrong_backend_is_refused(spec, uri):
    spec = spec.model_copy(update={"tracking": spec.tracking.model_copy(update={"backend": "wandb"})})
    with pytest.raises(TrackerError, match="wandb"):
        MlflowTracker(uri=uri).start_run(run_id="r1", experiment_id="e1", spec=spec)


def test_calls_out_of_order(spec, uri):
    tracker = MlflowTracker(uri=uri)
    with pytest.raises(TrackerError, match="no active run"):
        tracker.log_metrics({"x": 1.0}, step=1)
    with pytest.raises(TrackerError, match="no active run"):
        tracker.end_run("FINISHED")
    tracker.start_run(run_id="r1", experiment_id="e1", spec=spec)
    with pytest.raises(TrackerError, match="still active"):
        tracker.start_run(run_id="r2", experiment_id="e1", spec=spec)
    tracker.end_run("FINISHED")


def test_end_to_end_through_the_runner(spec, uri):
    data = SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)
    result = run_experiment(spec, data=data, tracker=MlflowTracker(uri=uri))
    run = _find(MlflowClient(uri), "smoke", result.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics == pytest.approx(result.final_metrics)


def test_importing_the_core_does_not_import_mlflow():
    code = (
        "import sys, mlops.application, mlops.adapters, mlops.adapters.inmemory_tracker;"
        "assert 'mlflow' not in sys.modules, 'mlflow leaked into the core import graph'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)


def test_metric_keys_and_undefined_values_reach_mlflow(spec, uri):
    import math

    spec = load_experiment_dict({**spec.model_dump(), "metrics": [{"name": "dice", "params": {}}]})
    tracker = MlflowTracker(uri=uri)
    tracker.start_run(run_id="r" * 64, experiment_id="e" * 64, spec=spec)
    tracker.log_metrics({"val/dice/spleen": 0.75, "val/dice/liver": float("nan")}, step=1)
    tracker.end_run("FINISHED")

    run = _find(MlflowClient(uri), spec.tracking.experiment_name, "r" * 64)
    assert run.data.metrics["val/dice/spleen"] == 0.75
    assert math.isnan(run.data.metrics["val/dice/liver"])
    assert run.data.params["metrics.0.name"] == "dice"


def test_resume_reopens_the_same_mlflow_run(spec, uri):
    tracker = MlflowTracker(uri=uri)
    tracker.start_run(run_id="r" * 64, experiment_id="e" * 64, spec=spec)
    tracker.log_metrics({"train/loss": 1.5}, step=1)
    tracker.end_run("FAILED")

    tracker = MlflowTracker(uri=uri)
    tracker.start_run(run_id="r" * 64, experiment_id="e" * 64, spec=spec, resume=True)
    tracker.log_metrics({"train/loss": 1.0}, step=2)
    tracker.end_run("FINISHED")

    run = _find(MlflowClient(uri), spec.tracking.experiment_name, "r" * 64)  # still exactly one run
    assert run.info.status == "FINISHED"
    history = MlflowClient(uri).get_metric_history(run.info.run_id, "train/loss")
    assert [(m.step, m.value) for m in history] == [(1, 1.5), (2, 1.0)]


def test_resume_without_an_earlier_mlflow_run_starts_one(spec, uri):
    tracker = MlflowTracker(uri=uri)
    tracker.start_run(run_id="n" * 64, experiment_id="e" * 64, spec=spec, resume=True)
    tracker.end_run("FINISHED")
    assert _find(MlflowClient(uri), spec.tracking.experiment_name, "n" * 64).info.status == "FINISHED"
