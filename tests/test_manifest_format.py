from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.application.bundle import MANIFEST, load_bundle
from mlops.application.diff import diff_runs
from mlops.application.gc import GcPolicy, apply_gc, plan_gc
from mlops.application.listing import list_runs
from mlops.application.predict import RESULT, predict_run
from mlops.cli import main
from mlops.config import load_experiment_yaml
from mlops.domain import FORMAT_VERSION, BundleFormatError, DomainError, RunManifest, parse_manifest
from mlops.domain.prediction import RESULT_FORMAT_VERSION, parse_result
from mlops.registry import discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"

OLD = {"experiment_id": "e", "run_id": "r", "attempt": 1, "status": "FINISHED", "device": "cpu"}


def _raw(**fields) -> str:
    return json.dumps({**OLD, **fields})


def test_manifest_without_a_version_is_format_0_and_reads_as_before():
    m = parse_manifest(json.dumps(OLD))
    assert m.format_version == 0
    assert m == RunManifest.model_validate_json(json.dumps(OLD))


def test_known_versions_read():
    assert parse_manifest(_raw(format_version=0)).format_version == 0
    assert parse_manifest(_raw(format_version=FORMAT_VERSION)).format_version == FORMAT_VERSION


def test_newer_version_says_upgrade_even_with_fields_this_code_does_not_know():
    raw = _raw(format_version=FORMAT_VERSION + 1, timings=[{"epoch": 1}])
    with pytest.raises(BundleFormatError) as e:
        parse_manifest(raw)
    assert e.value.version == FORMAT_VERSION + 1 and e.value.known == FORMAT_VERSION
    msg = str(e.value)
    assert "newer mlops" in msg and "upgrade mlops" in msg and f"format {FORMAT_VERSION + 1}" in msg
    assert isinstance(e.value, ValueError) and isinstance(e.value, DomainError)


@pytest.mark.parametrize("raw", [_raw(statuss="FINISHED"), _raw(format_version=FORMAT_VERSION, timings=[])])
def test_unknown_field_in_a_known_version_still_fails(raw):
    with pytest.raises(ValidationError):
        parse_manifest(raw)


@pytest.mark.parametrize("bad", [-1, "1", "99", 1.0, True, None])
def test_malformed_version_is_rejected_not_guessed(bad):
    with pytest.raises(ValidationError):
        parse_manifest(_raw(format_version=bad))


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", ""])
def test_non_json_or_non_object_fails_validation(raw):
    with pytest.raises(ValidationError):
        parse_manifest(raw)


def test_roundtrip_keeps_the_version_and_nan():
    m = RunManifest(**OLD, format_version=FORMAT_VERSION, history=[{"val/dice": float("nan")}])
    back = parse_manifest(m.model_dump_json().encode())
    assert back.format_version == FORMAT_VERSION and math.isnan(back.history[0]["val/dice"])


def _spec():
    discover("experiments")
    return load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")


def test_a_finished_run_writes_the_current_version():
    store, tracker = InMemoryArtifactStore(), InMemoryTracker()
    result = run_experiment(_spec(), data=SyntheticBlobs(n_train=4, n_val=2), tracker=tracker, artifacts=store)
    assert json.loads(store.get(result.run_id, MANIFEST))["format_version"] == FORMAT_VERSION


def test_a_failed_run_writes_the_current_version():
    class ExplodingLoader(SyntheticBlobs):
        def load(self, dataset, preprocessing):
            splits = super().load(dataset, preprocessing)
            bad = torch.utils.data.TensorDataset(torch.zeros(2, 3, 8, 8, 8), torch.zeros(2, 8, 8, 8, dtype=torch.long))
            return type(splits)(train=bad, val=splits.val)

    store, tracker = InMemoryArtifactStore(), InMemoryTracker()
    with pytest.raises(RuntimeError):
        run_experiment(_spec(), data=ExplodingLoader(n_train=2, n_val=0), tracker=tracker, artifacts=store)
    run_id = tracker.runs[0].run_id
    assert json.loads(store.get(run_id, MANIFEST))["format_version"] == FORMAT_VERSION


def _newer_manifest(store, run_id):
    m = json.loads(store.get(run_id, MANIFEST))
    store.put(run_id, MANIFEST, json.dumps({**m, "format_version": FORMAT_VERSION + 1, "timings": []}).encode())


def _newer_result(store, result_id):
    r = json.loads(store.get(result_id, RESULT))
    store.put(result_id, RESULT, json.dumps({**r, "format_version": RESULT_FORMAT_VERSION + 1, "extra": 1}).encode())


def _local_run(tmp_path, seed=0):
    spec = _spec()
    spec = spec.model_copy(update={"training": spec.training.model_copy(update={"seed": seed})})
    store = LocalArtifactStore(tmp_path)
    return store, run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store).run_id


def _data():
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)


def test_load_bundle_and_diff_refuse_a_newer_manifest(tmp_path):
    store, run_id = _local_run(tmp_path)
    _newer_manifest(store, run_id)
    with pytest.raises(BundleFormatError):
        load_bundle(store, run_id)
    with pytest.raises(BundleFormatError):
        diff_runs(store, run_id, store, run_id)


def test_list_shows_a_newer_manifest_as_unreadable_with_the_reason(tmp_path):
    store, run_id = _local_run(tmp_path)
    _newer_manifest(store, run_id)
    (s,) = list_runs(store)
    assert s.status == "UNREADABLE" and "upgrade mlops" in s.error


def _cli_commands(art, a, b):
    return [
        ["report", a, "--artifacts", art],
        ["reproduce", a, "--artifacts", art, "--tracker", "inmemory"],
        ["diff", a, b, "--artifacts", art],
        ["resume", a, "--artifacts", art, "--tracker", "inmemory"],
        ["predict", a, "--artifacts", art, "--part", "val"],
        ["ensemble", a, b, "--artifacts", art, "--part", "val", "--vote", "soft"],
    ]


@pytest.mark.parametrize("i", range(6))
def test_cli_exits_2_and_says_upgrade_on_a_newer_manifest(tmp_path, capsys, i):
    store, a = _local_run(tmp_path, seed=0)
    _, b = _local_run(tmp_path, seed=1)
    _newer_manifest(store, a)
    cmd = _cli_commands(str(tmp_path), a, b)[i]
    assert main(cmd) == 2, cmd
    err = capsys.readouterr().err
    assert "upgrade mlops" in err and "Traceback" not in err


def test_cli_list_and_gc_keep_a_newer_run(tmp_path, capsys):
    store, run_id = _local_run(tmp_path)
    _newer_manifest(store, run_id)
    assert main(["list", "--artifacts", str(tmp_path)]) == 0
    assert "upgrade mlops" in capsys.readouterr().out
    assert main(["gc", "--artifacts", str(tmp_path), "--apply"]) == 0
    assert MANIFEST in store.list(run_id)


def test_newer_result_is_refused():
    raw = json.dumps({"format_version": RESULT_FORMAT_VERSION + 1, "result_id": "x", "anything": 1})
    with pytest.raises(BundleFormatError) as e:
        parse_result(raw)
    assert "result.json" in str(e.value)
    with pytest.raises(ValidationError):
        parse_result(json.dumps({"format_version": "2"}))


def test_gc_deletes_no_run_while_a_result_is_unreadable(tmp_path):
    store, finished = _local_run(tmp_path, seed=0)
    _, used = _local_run(tmp_path, seed=1)
    _, failed = _local_run(tmp_path, seed=2)
    store.put(failed, MANIFEST, store.get(failed, MANIFEST).replace(b'"FINISHED"', b'"FAILED"'))
    store.delete(failed, "model.pt")
    store.put(finished, "checkpoint.pt", b"x")
    res = predict_run(store, used, data=_data(), part="val")
    _newer_result(store, res.result_id)

    plan = plan_gc(store, GcPolicy(keep_attempts=1))
    assert [i.run_id for i in plan.delete] == [finished] and not plan.delete[0].whole_run
    assert {i.run_id for i in plan.kept} >= {failed}
    assert any("no run is deleted" in w for w in plan.warnings)
    apply_gc(store, plan)
    assert MANIFEST in store.list(failed) and "checkpoint.pt" not in store.list(finished)


def test_gc_apply_skips_whole_runs_when_a_result_became_unreadable(tmp_path):
    store, failed = _local_run(tmp_path, seed=0)
    _, used = _local_run(tmp_path, seed=1)
    store.put(failed, MANIFEST, store.get(failed, MANIFEST).replace(b'"FINISHED"', b'"FAILED"'))
    store.delete(failed, "model.pt")
    res = predict_run(store, used, data=_data(), part="val")
    plan = plan_gc(store)
    assert [i.run_id for i in plan.delete] == [failed]
    _newer_result(store, res.result_id)
    outcome = apply_gc(store, plan)
    assert outcome.deleted == [] and "unreadable" in outcome.skipped[0][1]
    assert MANIFEST in store.list(failed)


def test_predict_writes_the_current_result_version_and_old_format_key_is_refused(tmp_path):
    store, run_id = _local_run(tmp_path)
    res = predict_run(store, run_id, data=_data(), part="val")
    raw = json.loads(store.get(res.result_id, RESULT))
    assert raw["format_version"] == RESULT_FORMAT_VERSION and "format" not in raw
    old = {k: v for k, v in raw.items() if k != "format_version"} | {"format": 1}
    with pytest.raises(ValidationError):
        parse_result(json.dumps(old))


FIXTURES = Path(__file__).parent / "fixtures/formats"
FORMATS = {"manifest": (parse_manifest, FORMAT_VERSION, 0), "result": (parse_result, RESULT_FORMAT_VERSION, 1)}


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.name)
def test_every_recorded_format_still_reads(path):
    kind, version = path.stem.rsplit("_v", 1)
    parse, current, _ = FORMATS[kind]
    record = parse(path.read_bytes())
    assert record.format_version == int(version) <= current


@pytest.mark.parametrize("kind", FORMATS)
def test_a_fixture_exists_for_every_readable_version(kind):
    _, current, oldest = FORMATS[kind]
    assert {int(p.stem.rsplit("_v", 1)[1]) for p in FIXTURES.glob(f"{kind}_v*.json")} == set(range(oldest, current + 1))


def test_format_0_manifest_reads_its_content():
    m = parse_manifest((FIXTURES / "manifest_v0.json").read_bytes())
    assert m.status == "FINISHED" and m.launcher == "local" and len(m.history) == 1
    assert m.best is not None and m.best.metric == "val/loss" and m.best.epoch == 1
    assert m.environment is not None and m.data_snapshot.status == "unknown"
