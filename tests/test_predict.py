"""`predict_run` / `mlops predict`: a finished run's model on one part of its data, as a
reproducible result next to the runs (roadmap 16, phase 2 of the split/predict/ensemble plan)."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.application.bundle import MANIFEST, save_bundle, load_bundle
from mlops.application.listing import list_runs
from mlops.application.predict import (
    PREDICTIONS,
    REPORT,
    RESULT,
    VAL_NOTE,
    PredictError,
    list_results,
    load_result,
    mean_metrics,
    predict_run,
)
from mlops.config import load_experiment_dict
from mlops.domain.best import BestCriterion
from mlops.domain.prediction import CaseResult
from mlops.ports.data import DataError
from mlops.registry import discover

SMOKE_YAML = Path(__file__).parent.parent / "experiments/configs/tiny_cnn_smoke.yaml"


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


def _spec(*, split=None, epochs=2, metrics=True):
    raw = yaml.safe_load(SMOKE_YAML.read_text())
    raw["training"]["epochs"] = epochs
    if split is not None:
        raw["split"] = split
    if metrics:
        raw["metrics"] = [{"name": "dice", "params": {"include_background": False}}]
    return load_experiment_dict(raw)


def _data():
    return SyntheticBlobs(n_train=6, n_val=4)


WITH_TEST = {"kind": "holdout", "val": 0.2, "test": 0.3}  # 10 cases: test 3, val 2, train 5


@pytest.fixture
def trained():
    store = InMemoryArtifactStore()
    r = run_experiment(_spec(split=WITH_TEST), data=_data(), tracker=InMemoryTracker(), artifacts=store,
                       keep_best=BestCriterion.parse("val/loss:min"))
    return store, r


def _snapshot(store, run_id):
    return {name: store.get(run_id, name) for name in store.list(run_id)}


def test_predicts_the_test_cases_and_writes_a_result(trained):
    store, run = trained
    before = _snapshot(store, run.run_id)
    rec = predict_run(store, run.run_id, data=_data())
    assert rec.part == "test" and [c.case for c in rec.cases] == run.split.test
    assert set(rec.metrics) == {"loss", "dice/1"} and rec.notes == []
    m = rec.members[0]
    assert (m.run_id, m.weights) == (run.run_id, "best.pt")  # keep-best weights by default
    assert m.epoch == load_bundle(store, run.run_id).manifest.best.epoch
    assert load_result(store, rec.result_id) == rec
    assert store.get(rec.result_id, REPORT).decode().startswith("# tiny-cnn-smoke — predict on test")
    assert _snapshot(store, run.run_id) == before  # the run is evidence: untouched


def test_same_inputs_same_result_bit_for_bit(trained):
    store, run = trained
    a = predict_run(store, run.run_id, data=_data())
    b = predict_run(store, run.run_id, data=_data(), out=InMemoryArtifactStore())
    assert a == b


def test_weights_and_part_change_the_result_id(trained):
    store, run = trained
    ids = {
        predict_run(store, run.run_id, data=_data()).result_id,
        predict_run(store, run.run_id, data=_data(), weights="model.pt").result_id,
        predict_run(store, run.run_id, data=_data(), part="val").result_id,
    }
    assert len(ids) == 3


def test_val_numbers_match_the_training_val_metrics_and_carry_a_warning():
    store = InMemoryArtifactStore()
    run = run_experiment(_spec(split=WITH_TEST), data=_data(), tracker=InMemoryTracker(), artifacts=store)
    rec = predict_run(store, run.run_id, data=_data(), part="val", weights="model.pt")
    last = run.history[-1]
    assert rec.metrics["loss"] == pytest.approx(last["val/loss"], rel=1e-6)
    assert rec.metrics["dice/1"] == pytest.approx(last["val/dice/1"], rel=1e-6)
    assert rec.notes == [VAL_NOTE]
    assert rec.members[0].epoch == 2


def test_run_without_spec_split_predicts_val_with_numbered_cases():
    store = InMemoryArtifactStore()
    run = run_experiment(_spec(), data=_data(), tracker=InMemoryTracker(), artifacts=store)
    with pytest.raises(PredictError, match="no test split"):
        predict_run(store, run.run_id, data=_data())
    rec = predict_run(store, run.run_id, data=_data(), part="val")
    assert [c.case for c in rec.cases] == ["case-000", "case-001", "case-002", "case-003"]


def test_save_predictions_writes_one_label_map_per_case(trained):
    store, run = trained
    rec = predict_run(store, run.run_id, data=_data(), save_predictions=True)
    for c in rec.cases:
        labels = torch.load(io.BytesIO(store.get(rec.result_id, f"{PREDICTIONS}/{c.case}.pt")))
        assert labels.dtype == torch.uint8 and labels.shape == (8, 8, 8)


def test_refuses_a_failed_run_and_missing_best_weights(trained):
    store, run = trained
    plain = InMemoryArtifactStore()
    r = run_experiment(_spec(split=WITH_TEST), data=_data(), tracker=InMemoryTracker(), artifacts=plain)
    with pytest.raises(PredictError, match="no best.pt"):
        predict_run(plain, r.run_id, data=_data(), weights="best.pt")
    b = load_bundle(plain, r.run_id)
    save_bundle(plain, spec=b.spec, manifest=b.manifest.model_copy(update={"status": "FAILED"}), model_state=None)
    with pytest.raises(PredictError, match="FAILED"):
        predict_run(plain, r.run_id, data=_data())


def test_refuses_when_the_data_now_splits_differently(trained):
    store, run = trained
    with pytest.raises(PredictError, match="splits differently"):
        predict_run(store, run.run_id, data=SyntheticBlobs(n_train=7, n_val=4))  # one more case


def test_refuses_wrong_data():
    store = InMemoryArtifactStore()
    raw = yaml.safe_load(SMOKE_YAML.read_text())
    raw["dataset"]["snapshot"] = "sha256:" + "1" * 64
    raw["split"] = WITH_TEST
    spec = load_experiment_dict(raw)

    class Hashed(SyntheticBlobs):
        def snapshot(self, dataset):
            return "sha256:" + "1" * 64

    run = run_experiment(spec, data=Hashed(n_train=6, n_val=4), tracker=InMemoryTracker(), artifacts=store)

    class Other(SyntheticBlobs):
        def snapshot(self, dataset):
            return "sha256:" + "2" * 64

    with pytest.raises(DataError):
        predict_run(store, run.run_id, data=Other(n_train=6, n_val=4))


def test_mean_skips_undefined_cases():
    rows = [CaseResult(case="a", metrics={"d": 1.0}), CaseResult(case="b", metrics={"d": math.nan}),
            CaseResult(case="c", metrics={"d": 0.0})]
    assert mean_metrics(rows) == {"d": 0.5}
    assert math.isnan(mean_metrics([rows[1]])["d"])


def test_results_live_beside_runs_without_being_listed_as_runs(tmp_path):
    store = LocalArtifactStore(tmp_path)
    run = run_experiment(_spec(split=WITH_TEST), data=_data(), tracker=InMemoryTracker(), artifacts=store)
    rec = predict_run(store, run.run_id, data=_data())
    assert [s.run_id for s in list_runs(store)] == [run.run_id]
    assert [r.result_id for r in list_results(store)] == [rec.result_id]
    assert json.loads((tmp_path / rec.result_id / RESULT).read_text())["result_id"] == rec.result_id


# -- CLI ---------------------------------------------------------------------------

from mlops.cli import main  # noqa: E402


def _cli_run(tmp_path, capsys, split=WITH_TEST):
    raw = yaml.safe_load(SMOKE_YAML.read_text())
    raw["split"] = split
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert main(["run", str(path), "--tracker", "inmemory", "--artifacts", str(tmp_path / "a"), "--json"]) == 0
    return json.loads(capsys.readouterr().out)["run_id"]


def test_cli_predict_and_results(tmp_path, capsys):
    run_id = _cli_run(tmp_path, capsys)
    assert main(["predict", run_id, "--artifacts", str(tmp_path / "a")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("tiny-cnn-smoke: predicted test (4 cases)") and "loss=" in out
    assert main(["results", "--artifacts", str(tmp_path / "a")]) == 0
    assert "predict test (4)" in capsys.readouterr().out


def test_cli_predict_json_and_val_note(tmp_path, capsys):
    run_id = _cli_run(tmp_path, capsys)
    assert main(["predict", run_id, "--artifacts", str(tmp_path / "a"), "--part", "val", "--json"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["part"] == "val" and rec["notes"] == [VAL_NOTE]


def test_cli_predict_errors_are_exit_2(tmp_path, capsys):
    run_id = _cli_run(tmp_path, capsys, split={"kind": "holdout"})  # no test
    assert main(["predict", run_id, "--artifacts", str(tmp_path / "a")]) == 2
    assert "no test split" in capsys.readouterr().err
    assert main(["predict", "f" * 64, "--artifacts", str(tmp_path / "a")]) == 2
