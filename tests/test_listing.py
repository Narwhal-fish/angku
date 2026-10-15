"""``list_runs``: one summary per run under an artifact root, from spec + manifest only.

Answers "did this ExperimentID give the same result across its attempts?" without opening
a single weight file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.application.bundle import MANIFEST, MODEL
from mlops.application.listing import group_by_experiment, list_runs
from mlops.config import load_experiment_yaml
from mlops.domain import compute_experiment_id
from mlops.registry import discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


@pytest.fixture
def spec():
    return load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")


@pytest.fixture
def store():
    return InMemoryArtifactStore()


def _data() -> SyntheticBlobs:
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)


def _run(store, spec, **kw) -> str:
    return run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, **kw).run_id


def _seed(spec, seed):
    return spec.model_copy(update={"training": spec.training.model_copy(update={"seed": seed})})


def test_empty_root(store):
    assert list_runs(store) == []


def test_one_summary_per_run(spec, store):
    run_id = _run(store, spec)
    (s,) = list_runs(store)
    assert s.run_id == run_id
    assert s.experiment_id == compute_experiment_id(spec)
    assert s.name == "tiny-cnn-smoke"
    assert s.attempt == 1 and s.status == "FINISHED"
    assert (s.epochs_done, s.epochs_total) == (1, 1)
    assert set(s.final_metrics) == {"train/loss", "val/loss"}
    assert s.workers == 1 and s.resumes == 0 and not s.resumable


def test_sorted_by_name_experiment_then_attempt(spec, store):
    _run(store, spec, attempt=2)
    _run(store, _seed(spec, 1))
    _run(store, spec, attempt=1)
    runs = list_runs(store)
    exp = compute_experiment_id(spec)
    same = [r for r in runs if r.experiment_id == exp]
    assert [r.attempt for r in same] == [1, 2]
    assert runs == sorted(runs, key=lambda r: (r.name, r.experiment_id, r.attempt or 0, r.run_id))


def test_filter_by_experiment_prefix(spec, store):
    _run(store, spec)
    _run(store, _seed(spec, 1))
    exp = compute_experiment_id(spec)
    assert [r.experiment_id for r in list_runs(store, experiment=exp[:12])] == [exp]
    assert list_runs(store, experiment="ffffffffffff") == []


def test_failed_run_is_listed(spec, store):
    run_id = _run(store, spec)
    m = store.get(run_id, MANIFEST).replace(b'"FINISHED"', b'"FAILED"')
    store.put(run_id, MANIFEST, m)
    store._runs[run_id].pop(MODEL)
    (s,) = list_runs(store)
    assert s.status == "FAILED"


def test_killed_run_with_checkpoint_is_interrupted_and_resumable(spec, store):
    from mlops.application.bundle import CHECKPOINT

    run_id = _run(store, spec, checkpoint_every=1, keep_checkpoint=True)  # as if it never finished
    store._runs[run_id].pop(MANIFEST)
    store._runs[run_id].pop(MODEL)
    (s,) = list_runs(store)
    assert s.status == "INTERRUPTED" and s.resumable
    assert s.attempt is None and s.epochs_done is None
    assert CHECKPOINT in store.list(run_id)


def test_failed_run_with_checkpoint_is_resumable(spec, store):
    run_id = _run(store, spec, checkpoint_every=1, keep_checkpoint=True)  # as if it never finished
    store.put(run_id, MANIFEST, store.get(run_id, MANIFEST).replace(b'"FINISHED"', b'"FAILED"'))
    assert list_runs(store)[0].resumable


def test_finished_run_with_checkpoint_is_not_resumable(spec, store):
    _run(store, spec, checkpoint_every=1)
    assert not list_runs(store)[0].resumable


def test_directory_without_spec_is_ignored(spec, store):
    _run(store, spec)
    store.put("scratch", "notes.txt", b"not a bundle")
    assert len(list_runs(store)) == 1


def test_corrupt_bundle_is_listed_as_unreadable(spec, store):
    _run(store, spec)
    store.put("broken", "spec.json", b"{not json")
    broken = [r for r in list_runs(store) if r.run_id == "broken"]
    assert broken[0].status == "UNREADABLE" and broken[0].error


# --- grouping: did the experiment reproduce across attempts? ---


def test_attempts_of_one_experiment_agree(spec, store):
    for attempt in (1, 2, 3):
        _run(store, spec, attempt=attempt)
    (g,) = group_by_experiment(list_runs(store))
    assert g.experiment_id == compute_experiment_id(spec)
    assert [r.attempt for r in g.runs] == [1, 2, 3]
    assert g.finished == 3
    assert g.metrics_agree is True
    assert g.spread == {}


def test_disagreeing_attempts_name_the_metric_and_its_range(spec, store):
    from dataclasses import replace

    for attempt in (1, 2):
        _run(store, spec, attempt=attempt)
    runs = list_runs(store)
    runs[1] = replace(runs[1], final_metrics={**runs[1].final_metrics, "val/loss": 9.0})
    (g,) = group_by_experiment(runs)
    assert g.metrics_agree is False
    assert set(g.spread) == {"val/loss"}
    lo, hi = g.spread["val/loss"]
    assert hi == 9.0 and lo < hi


def test_agreement_needs_two_finished_runs(spec, store):
    _run(store, spec)
    (g,) = group_by_experiment(list_runs(store))
    assert g.metrics_agree is None


def test_groups_follow_list_order(spec, store):
    _run(store, spec)
    _run(store, _seed(spec, 1))
    groups = group_by_experiment(list_runs(store))
    assert len(groups) == 2
    assert all(len(g.runs) == 1 for g in groups)


def test_killed_ray_run_with_launch_record_is_resumable(spec, store):
    from mlops.application.bundle import LaunchRecord, save_launch

    run_id = _run(store, spec)
    store._runs[run_id].pop(MANIFEST)
    store._runs[run_id].pop(MODEL)
    save_launch(store, run_id, LaunchRecord(launcher="ray", attempt=1, checkpoint_every=1, storage_path="/s"))
    (s,) = list_runs(store)
    assert s.status == "INTERRUPTED" and s.resumable


# --- tolerance: runs from different launchers agree only to the last bits ---


def _two_with(store, spec, changes):
    """Two attempts of ``spec``; the second's final metrics updated with ``changes(final)``."""
    from dataclasses import replace

    for attempt in (1, 2):
        _run(store, spec, attempt=attempt)
    runs = list_runs(store)
    runs[1] = replace(runs[1], final_metrics={**runs[1].final_metrics, **changes(runs[1].final_metrics)})
    return runs


def test_atol_tolerates_last_bit_drift(spec, store):
    runs = _two_with(store, spec, changes=lambda m: {"val/loss": m["val/loss"] + 1e-9})
    (exact,) = group_by_experiment(runs)
    assert exact.metrics_agree is False and set(exact.spread) == {"val/loss"}
    (tolerant,) = group_by_experiment(runs, atol=1e-6)
    assert tolerant.metrics_agree is True and tolerant.spread == {} and tolerant.atol == 1e-6


def test_atol_still_flags_a_real_difference(spec, store):
    runs = _two_with(store, spec, changes=lambda m: {"val/loss": m["val/loss"] + 0.1})
    (g,) = group_by_experiment(runs, atol=1e-6)
    assert g.metrics_agree is False and set(g.spread) == {"val/loss"}


def test_nan_agrees_only_with_nan_whatever_the_tolerance(spec, store):
    import math

    runs = _two_with(store, spec, changes=lambda m: {"val/loss": math.nan})
    (g,) = group_by_experiment(runs, atol=1e9)
    assert g.metrics_agree is False
    runs = _two_with(InMemoryArtifactStore(), spec, changes=lambda m: {"val/x": math.nan})
    from dataclasses import replace

    runs[0] = replace(runs[0], final_metrics={**runs[0].final_metrics, "val/x": math.nan})
    (g,) = group_by_experiment(runs, atol=1e-6)
    assert g.metrics_agree is True


def test_summary_carries_the_keep_best_record(spec, store):
    from mlops.domain.best import BestCriterion

    _run(store, spec, keep_best=BestCriterion(metric="val/loss", mode="min"))
    (s,) = list_runs(store)
    assert s.best is not None and (s.best.metric, s.best.epoch) == ("val/loss", 1)
    assert s.to_dict()["best"] == {"metric": "val/loss", "mode": "min", "epoch": 1, "value": s.best.value}
