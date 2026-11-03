"""`ensemble_runs` / `angku ensemble`: soft and hard voting over several runs' predictions on the
same cases (roadmap 15, phase 3 of the split/predict/ensemble plan)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.application.predict import PredictError, combine, ensemble_runs, load_result, predict_run
from mlops.config import load_experiment_dict
from mlops.registry import discover

SMOKE_YAML = Path(__file__).parent.parent / "experiments/configs/tiny_cnn_smoke.yaml"
WITH_TEST = {"kind": "holdout", "val": 0.2, "test": 0.3}


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


def _spec(*, seed=0, split=WITH_TEST, metrics=None, hidden=4):
    raw = yaml.safe_load(SMOKE_YAML.read_text())
    raw["training"]["epochs"] = 2
    raw["training"]["seed"] = seed
    raw["model"]["params"]["hidden"] = hidden
    raw["split"] = split
    raw["metrics"] = metrics if metrics is not None else [{"name": "dice", "params": {}}]
    return load_experiment_dict(raw)


def _data():
    return SyntheticBlobs(n_train=6, n_val=4)


def _train(store, spec, attempt=1):
    return run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, attempt=attempt).run_id


@pytest.fixture(scope="module")
def three():
    """Three different runs of comparable experiments (one has another architecture)."""
    store = InMemoryArtifactStore()
    ids = [_train(store, _spec(seed=0)), _train(store, _spec(seed=1)), _train(store, _spec(seed=2, hidden=8))]
    return store, ids


# -- combine ------------------------------------------------------------------------


def _logits(*per_voxel):
    """(1, C, V) logits whose softmax is the given per-voxel probabilities."""
    return torch.tensor(per_voxel, dtype=torch.float64).T.unsqueeze(0).log()


def test_soft_averages_probabilities():
    a = _logits([0.9, 0.1], [0.4, 0.6])
    b = _logits([0.3, 0.7], [0.4, 0.6])
    out, with_loss = combine([a, b], "soft")
    assert with_loss
    assert torch.allclose(out.softmax(dim=1), torch.tensor([[[0.6, 0.4], [0.4, 0.6]]], dtype=torch.float64))
    assert out.argmax(dim=1).tolist() == [[0, 1]]


def test_hard_is_a_majority_vote_without_loss():
    # voxel 0: votes 0,0,1 -> 0 although the mean probability favours 1
    a, b, c = _logits([0.51, 0.49]), _logits([0.51, 0.49]), _logits([0.01, 0.99])
    out, with_loss = combine([a, b, c], "hard")
    assert not with_loss
    assert out.argmax(dim=1).tolist() == [[0]]
    assert sorted(out.unique().tolist()) == [0.0, 1.0]  # one-hot
    soft, _ = combine([a, b, c], "soft")
    assert soft.argmax(dim=1).tolist() == [[1]]


def test_hard_tie_goes_to_the_higher_mean_probability_then_the_lower_class():
    a, b = _logits([0.9, 0.1]), _logits([0.4, 0.6])  # one vote each; mean prob 0.65 vs 0.35
    assert combine([a, b], "hard")[0].argmax(dim=1).tolist() == [[0]]
    c, d = _logits([0.6, 0.4]), _logits([0.4, 0.6])  # one vote each, mean prob equal
    assert combine([c, d], "hard")[0].argmax(dim=1).tolist() == [[0]]


# -- ensemble_runs --------------------------------------------------------------------


def test_soft_ensemble_records_members_and_their_own_numbers(three):
    store, ids = three
    rec = ensemble_runs(store, ids, vote="soft", data=_data())
    assert rec.kind == "ensemble" and rec.method == {"vote": "soft"}
    assert [m.run_id for m in rec.members] == sorted(ids)
    assert set(rec.metrics) == {"loss", "dice/0", "dice/1", "dice/mean"}
    for run_id in ids:  # each member's own numbers are exactly what `predict` says for it
        assert (
            rec.member_metrics[run_id] == predict_run(store, run_id, data=_data(), out=InMemoryArtifactStore()).metrics
        )
    assert load_result(store, rec.result_id) == rec


def test_hard_ensemble_has_no_loss_and_its_own_id(three):
    store, ids = three
    hard = ensemble_runs(store, ids, vote="hard", data=_data())
    soft = ensemble_runs(store, ids, vote="soft", data=_data())
    assert "loss" not in hard.metrics and hard.result_id != soft.result_id


def test_order_of_the_runs_does_not_matter_and_reruns_are_identical(three):
    store, ids = three
    a = ensemble_runs(store, ids, vote="soft", data=_data())
    b = ensemble_runs(store, list(reversed(ids)), vote="soft", data=_data(), out=InMemoryArtifactStore())
    assert a == b


def test_identical_members_vote_like_one_model():
    store = InMemoryArtifactStore()
    spec = _spec()
    runs = [_train(store, spec, attempt=n) for n in (1, 2, 3)]  # deterministic: same weights
    single = predict_run(store, runs[0], data=_data())
    for vote in ("soft", "hard"):
        rec = ensemble_runs(store, runs, vote=vote, data=_data())
        for k in ("dice/0", "dice/1", "dice/mean"):
            assert rec.metrics[k] == single.metrics[k], (vote, k)
    soft = ensemble_runs(store, runs, vote="soft", data=_data())
    assert soft.metrics["loss"] == pytest.approx(single.metrics["loss"], rel=1e-5)


def test_one_member_soft_equals_predict(three):
    store, ids = three
    rec = ensemble_runs(store, ids[:1], vote="soft", data=_data())
    single = predict_run(store, ids[0], data=_data())
    assert rec.metrics["dice/mean"] == single.metrics["dice/mean"]
    assert rec.metrics["loss"] == pytest.approx(single.metrics["loss"], rel=1e-5)


def test_member_runs_are_never_written(three):
    store, ids = three
    before = {r: {n: store.get(r, n) for n in store.list(r)} for r in ids}
    ensemble_runs(store, ids, vote="hard", data=_data(), save_predictions=True)
    assert before == {r: {n: store.get(r, n) for n in store.list(r)} for r in ids}


def test_kfold_members_share_the_test_cases_only():
    store = InMemoryArtifactStore()
    folds = [_train(store, _spec(split={"kind": "kfold", "k": 3, "fold": f, "test": 0.3})) for f in range(3)]
    rec = ensemble_runs(store, folds, vote="soft", data=_data())
    assert len(rec.cases) == 3
    with pytest.raises(PredictError, match="different val cases"):
        ensemble_runs(store, folds, vote="soft", data=_data(), part="val")


def test_runs_that_measure_differently_are_refused_before_anything_runs():
    store = InMemoryArtifactStore()
    a = _train(store, _spec())
    b = _train(store, _spec(seed=1, metrics=[{"name": "dice", "params": {"include_background": False}}]))
    with pytest.raises(PredictError, match="differ in `metrics`"):
        ensemble_runs(store, [a, b], vote="soft", data=_data())


def test_a_run_listed_twice_is_refused(three):
    store, ids = three
    with pytest.raises(PredictError, match="twice"):
        ensemble_runs(store, [ids[0], ids[0]], vote="soft", data=_data())


def test_report_shows_members_and_vote(three):
    store, ids = three
    rec = ensemble_runs(store, ids, vote="soft", data=_data())
    md = store.get(rec.result_id, "report.md").decode()
    assert md.startswith("# tiny-cnn-smoke — ensemble (soft vote) on test")
    assert "## Each run alone" in md
    for r in ids:
        assert r[:12] in md


# -- CLI ---------------------------------------------------------------------------

from mlops.cli import main  # noqa: E402


def test_cli_ensemble(tmp_path, capsys):
    ids = []
    for seed in (0, 1):
        raw = yaml.safe_load(SMOKE_YAML.read_text())
        raw["split"], raw["training"]["seed"] = WITH_TEST, seed
        path = tmp_path / f"s{seed}.yaml"
        path.write_text(yaml.safe_dump(raw))
        assert main(["run", str(path), "--tracker", "inmemory", "--artifacts", str(tmp_path / "a"), "--json"]) == 0
        ids.append(json.loads(capsys.readouterr().out)["run_id"])
    assert main(["ensemble", *ids, "--artifacts", str(tmp_path / "a"), "--vote", "hard"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("tiny-cnn-smoke: ensemble (hard vote) of 2 runs on test (4 cases)")
    assert "each run alone:" in out
    assert main(["ensemble", ids[0], "--artifacts", str(tmp_path / "a"), "--vote", "soft"]) == 2
    assert "at least two runs" in capsys.readouterr().err
    assert main(["results", "--artifacts", str(tmp_path / "a"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["kind"] == "ensemble"
