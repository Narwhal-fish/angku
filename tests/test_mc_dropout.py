"""`predict_run(mc_samples=)` / `angku predict --mc-samples`: MC dropout uncertainty (roadmap 15,
phase 4 of the split/predict/ensemble plan)."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.application.predict import (
    UNCERTAINTY,
    PredictError,
    enable_dropout,
    load_result,
    predict_run,
    uncertainty,
)
from mlops.config import load_experiment_dict
from mlops.registry import discover

SMOKE_YAML = Path(__file__).parent.parent / "experiments/configs/tiny_cnn_smoke.yaml"
WITH_TEST = {"kind": "holdout", "val": 0.2, "test": 0.3}
KEYS = {"loss", "dice/0", "dice/1", "dice/mean", "uncertainty/entropy", "uncertainty/variance"}


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


def _raw(dropout):
    raw = yaml.safe_load(SMOKE_YAML.read_text())
    raw["training"]["epochs"] = 2
    raw["split"] = WITH_TEST
    raw["metrics"] = [{"name": "dice", "params": {}}]
    if dropout:
        raw["model"]["params"]["dropout"] = dropout
    return raw


def _data():
    return SyntheticBlobs(n_train=6, n_val=4)


def _train(store, dropout=0.3):
    spec = load_experiment_dict(_raw(dropout))
    return run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store).run_id


@pytest.fixture(scope="module")
def trained():
    store = InMemoryArtifactStore()
    return store, _train(store)


# -- pure helpers ---------------------------------------------------------------------


def test_enable_dropout_leaves_everything_else_in_eval():
    model = nn.Sequential(nn.Conv1d(1, 2, 1), nn.BatchNorm1d(2), nn.Dropout(0.5), nn.Dropout(0.0))
    assert enable_dropout(model.eval()) == 1  # p=0 does not count
    assert [m.training for m in model] == [False, False, True, True]


def test_uncertainty_of_agreeing_and_disagreeing_samples():
    p = torch.tensor([[[0.5], [0.5]]])  # (1, C=2, V=1)
    entropy, variance = uncertainty(torch.stack([p, p]))
    assert entropy.shape == (1, 1)
    assert entropy.item() == pytest.approx(math.log(2)) and variance.item() == 0
    a, b = torch.tensor([[[1.0], [0.0]]]), torch.tensor([[[0.0], [1.0]]])
    entropy, variance = uncertainty(torch.stack([a, b]))
    assert entropy.item() == pytest.approx(math.log(2)) and variance.item() == pytest.approx(0.25)


# -- predict_run ----------------------------------------------------------------------


def test_mc_dropout_adds_uncertainty_and_records_the_method(trained):
    store, run_id = trained
    before = {n: store.get(run_id, n) for n in store.list(run_id)}
    rec = predict_run(store, run_id, data=_data(), mc_samples=8)
    assert rec.method == {"mc_dropout": {"samples": 8, "seed": 0}}
    assert set(rec.metrics) == KEYS
    for c in rec.cases:
        assert c.metrics["uncertainty/entropy"] > 0 and c.metrics["uncertainty/variance"] > 0
    assert load_result(store, rec.result_id) == rec
    md = store.get(rec.result_id, "report.md").decode()
    assert md.startswith("# tiny-cnn-smoke — predict with MC dropout (8 samples) on test")
    assert {n: store.get(run_id, n) for n in store.list(run_id)} == before


def test_mc_results_are_reproducible_and_the_settings_are_in_the_id(trained):
    store, run_id = trained
    a = predict_run(store, run_id, data=_data(), mc_samples=8)
    assert a == predict_run(store, run_id, data=_data(), mc_samples=8, out=InMemoryArtifactStore())
    other_seed = predict_run(store, run_id, data=_data(), mc_samples=8, mc_seed=1)
    fewer = predict_run(store, run_id, data=_data(), mc_samples=4)
    plain = predict_run(store, run_id, data=_data())
    assert len({a.result_id, other_seed.result_id, fewer.result_id, plain.result_id}) == 4
    assert a.metrics["uncertainty/entropy"] != other_seed.metrics["uncertainty/entropy"]


def test_plain_predict_of_a_dropout_model_is_deterministic_and_has_no_uncertainty(trained):
    store, run_id = trained
    a = predict_run(store, run_id, data=_data())
    assert a == predict_run(store, run_id, data=_data(), out=InMemoryArtifactStore())
    assert not any(k.startswith("uncertainty/") for k in a.metrics) and a.method == {}


def test_save_predictions_also_stores_the_uncertainty_maps(trained):
    store, run_id = trained
    rec = predict_run(store, run_id, data=_data(), mc_samples=4, save_predictions=True)
    for c in rec.cases:
        maps = torch.load(io.BytesIO(store.get(rec.result_id, f"{UNCERTAINTY}/{c.case}.pt")))
        assert set(maps) == {"entropy", "variance"}
        assert maps["entropy"].shape == (8, 8, 8) and maps["entropy"].dtype == torch.float32
        assert float(maps["entropy"].mean()) == pytest.approx(c.metrics["uncertainty/entropy"], rel=1e-5)


def test_refuses_a_model_without_dropout_and_too_few_samples(trained):
    store, run_id = trained
    with pytest.raises(PredictError, match="at least 2"):
        predict_run(store, run_id, data=_data(), mc_samples=1)
    plain_store = InMemoryArtifactStore()
    plain = _train(plain_store, dropout=0)
    with pytest.raises(PredictError, match="no dropout"):
        predict_run(plain_store, plain, data=_data(), mc_samples=4)


# -- CLI ---------------------------------------------------------------------------

from mlops.cli import main  # noqa: E402


def test_cli_predict_mc_samples(tmp_path, capsys):
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(_raw(0.3)))
    art = str(tmp_path / "a")
    assert main(["run", str(path), "--tracker", "inmemory", "--artifacts", art, "--json"]) == 0
    run_id = json.loads(capsys.readouterr().out)["run_id"]
    assert main(["predict", run_id, "--artifacts", art, "--mc-samples", "4", "--mc-seed", "3"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("tiny-cnn-smoke: predicted test (4 cases) with MC dropout (4 samples, seed 3)")
    assert "uncertainty/entropy=" in out
    assert main(["predict", run_id, "--artifacts", art, "--mc-samples", "1"]) == 2
    assert "at least 2" in capsys.readouterr().err
