"""Keep-best: opt-in ``keep_best=BestCriterion("val/loss", "min")`` stores the weights of the best
epoch as ``best.pt`` and records which epoch that was in ``manifest.best``.

Not part of any id: choosing which epoch to keep does not change training. The best epoch is a
pure function of the history (strictly better wins, so ties keep the earlier epoch; NaN never
wins); only the weights have to be carried, in the checkpoint too, so a resumed run ends with the
same ``best.pt`` as one left alone.
"""

from __future__ import annotations

import io
import math

import pytest
import torch
from pydantic import ValidationError

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.application import RunnerError, run_experiment
from mlops.application.bundle import BEST, CHECKPOINT, MODEL, load_bundle
from mlops.application.report import render_report
from mlops.domain.best import BestCriterion, best_epoch
from mlops.registry import discover
from tests.test_checkpoint import KilledStore, _data, _interrupt, spec  # noqa: F401 - fixture


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


# --- the criterion and picking the epoch: pure domain ---


@pytest.mark.parametrize(
    "text,metric,mode",
    [
        ("val/loss:min", "val/loss", "min"),
        ("val/dice/mean:max", "val/dice/mean", "max"),
    ],
)
def test_parse(text, metric, mode):
    c = BestCriterion.parse(text)
    assert (c.metric, c.mode) == (metric, mode)
    assert str(c) == text


@pytest.mark.parametrize("bad", ["val/loss", "val/loss:median", ":max", "val/loss:", ""])
def test_parse_rejects(bad):
    with pytest.raises(ValueError, match="METRIC:min|max"):
        BestCriterion.parse(bad)


def test_criterion_rejects_typos():
    with pytest.raises(ValidationError):
        BestCriterion(metric="val/loss", mode="min", modee="max")


H = [{"val/loss": 0.5}, {"val/loss": 0.3}, {"val/loss": math.nan}, {"val/loss": 0.3}, {"val/loss": 0.4}]


def test_best_epoch_min_max_ties_and_nan():
    assert best_epoch(H, BestCriterion(metric="val/loss", mode="min")) == (2, 0.3)  # tie: earlier wins
    assert best_epoch(H, BestCriterion(metric="val/loss", mode="max")) == (1, 0.5)
    assert best_epoch(H[:0], BestCriterion(metric="val/loss", mode="min")) is None
    assert best_epoch([{"val/loss": math.nan}], BestCriterion(metric="val/loss", mode="min")) is None


# --- in a run ---


def _truncated(spec, epochs):
    return spec.model_copy(update={"training": spec.training.model_copy(update={"epochs": epochs})})


def _load(store, run_id, name):
    return torch.load(io.BytesIO(store.get(run_id, name)), map_location="cpu", weights_only=True)


@pytest.mark.parametrize("mode", ["min", "max"])
def test_best_pt_holds_the_weights_of_the_best_epoch(spec, mode):
    store = InMemoryArtifactStore()
    crit = BestCriterion(metric="val/loss", mode=mode)
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, keep_best=crit)
    best = load_bundle(store, result.run_id).manifest.best
    epoch, value = best_epoch(result.history, crit)
    assert (best.metric, best.mode, best.epoch, best.value) == ("val/loss", mode, epoch, value)

    # No LR schedule: the same spec stopped after `epoch` epochs ends with exactly those weights.
    short_store = InMemoryArtifactStore()
    short = run_experiment(_truncated(spec, epoch), data=_data(), tracker=InMemoryTracker(), artifacts=short_store)
    expected, got = _load(short_store, short.run_id, MODEL), _load(store, result.run_id, BEST)
    assert all(torch.equal(v, got[k]) for k, v in expected.items())


def test_keep_best_does_not_change_the_run(spec):
    plain_store, best_store = InMemoryArtifactStore(), InMemoryArtifactStore()
    plain = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=plain_store)
    best = run_experiment(
        spec,
        data=_data(),
        tracker=InMemoryTracker(),
        artifacts=best_store,
        keep_best=BestCriterion(metric="val/loss", mode="min"),
    )
    assert plain.run_id == best.run_id and plain.history == best.history
    a, b = _load(plain_store, plain.run_id, MODEL), _load(best_store, best.run_id, MODEL)
    assert all(torch.equal(v, b[k]) for k, v in a.items())


def test_no_keep_best_no_best_pt(spec):
    store = InMemoryArtifactStore()
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store)
    assert BEST not in store.list(result.run_id)
    assert load_bundle(store, result.run_id).manifest.best is None


def test_unknown_metric_fails_after_the_first_epoch_naming_the_known_ones(spec):
    store, tracker = InMemoryArtifactStore(), InMemoryTracker()
    with pytest.raises(RunnerError, match=r"val/dice.*val/loss"):
        run_experiment(
            spec, data=_data(), tracker=tracker, artifacts=store, keep_best=BestCriterion(metric="val/dice", mode="max")
        )
    assert tracker.runs[0].status == "FAILED" and len(tracker.runs[0].metrics) == 1


@pytest.mark.parametrize("mode", ["min", "max"])
def test_resumed_run_ends_with_the_same_best(spec, mode):
    crit = BestCriterion(metric="val/loss", mode=mode)
    ref_store = InMemoryArtifactStore()
    ref = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=ref_store, keep_best=crit)

    store = KilledStore()
    _interrupt(spec, store, fail_at=3, every=2, keep_best=crit)
    # the criterion rides in the checkpoint: resume does not need it again
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, resume=True)
    assert load_bundle(store, result.run_id).manifest.best == load_bundle(ref_store, ref.run_id).manifest.best
    a, b = _load(ref_store, ref.run_id, BEST), _load(store, result.run_id, BEST)
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    assert CHECKPOINT not in store.list(result.run_id)


def test_resume_with_another_criterion_is_refused(spec):
    store = InMemoryArtifactStore()
    _interrupt(spec, store, keep_best=BestCriterion(metric="val/loss", mode="min"))
    with pytest.raises(RunnerError, match="val/loss:min"):
        run_experiment(
            spec,
            data=_data(),
            tracker=InMemoryTracker(),
            artifacts=store,
            resume=True,
            keep_best=BestCriterion(metric="val/loss", mode="max"),
        )


def test_failed_run_writes_no_best_pt(spec):
    store = InMemoryArtifactStore()
    _interrupt(spec, store, keep_best=BestCriterion(metric="val/loss", mode="min"))
    run_id = next(iter(store._runs))
    assert BEST not in store.list(run_id)


def test_report_names_the_best_epoch(spec):
    store = InMemoryArtifactStore()
    result = run_experiment(
        spec,
        data=_data(),
        tracker=InMemoryTracker(),
        artifacts=store,
        keep_best=BestCriterion(metric="val/loss", mode="min"),
    )
    best = load_bundle(store, result.run_id).manifest.best
    md = render_report(load_bundle(store, result.run_id))
    assert f"| best | `val/loss` (min) = {best.value:.6g} at epoch {best.epoch}, `best.pt` |" in md


# --- reproduce verifies best.pt too; the CLI face ---


def _best_run(spec, store):
    return run_experiment(
        spec,
        data=_data(),
        tracker=InMemoryTracker(),
        artifacts=store,
        keep_best=BestCriterion(metric="val/loss", mode="max"),
    ).run_id


def test_reproduce_checks_best_pt(spec):
    from mlops.application.reproduce import reproduce_run

    store = InMemoryArtifactStore()
    run_id = _best_run(spec, store)
    report = reproduce_run(store, run_id, data=_data(), tracker=InMemoryTracker(), scratch=InMemoryArtifactStore())
    assert report.ok

    best = _load(store, run_id, BEST)
    k = next(iter(best))
    best[k] = best[k] + 1.0
    buf = io.BytesIO()
    torch.save(best, buf)
    store.put(run_id, BEST, buf.getvalue())
    report = reproduce_run(store, run_id, data=_data(), tracker=InMemoryTracker(), scratch=InMemoryArtifactStore())
    assert not report.ok and not report.checks["weights"]
    assert any(m.detail.startswith(f"best.pt {k}:") for m in report.mismatches)


def test_cli_keep_best(tmp_path, capsys):
    import json

    from mlops.cli import main
    from tests.test_cli import _write_yaml

    art = tmp_path / "artifacts"
    yml = _write_yaml(tmp_path / "e.yaml", **{"training.epochs": 2})
    assert (
        main(
            ["run", str(yml), "--tracker", "inmemory", "--artifacts", str(art), "--keep-best", "val/loss:min", "--json"]
        )
        == 0
    )
    run_id = json.loads(capsys.readouterr().out)["run_id"]
    assert (art / run_id / BEST).is_file()
    assert "| best | `val/loss` (min)" in (art / run_id / "report.md").read_text()


def test_cli_keep_best_bad_value_exits_2(capsys):
    from mlops.cli import main
    from tests.test_cli import SMOKE

    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--keep-best", "val/loss"]) == 2
    err = capsys.readouterr().err
    assert "METRIC:min|max" in err and "Traceback" not in err


def test_rayjob_passes_keep_best_to_the_in_cluster_run(capsys):
    import json

    from mlops.cli import main
    from tests.test_cli import BASELINE

    assert main(["rayjob", str(BASELINE), "--data-pvc", "msd", "--keep-best", "val/dice/mean:max"]) == 0
    job = json.loads(capsys.readouterr().out)["items"][1]
    assert "--keep-best val/dice/mean:max" in job["spec"]["entrypoint"]
