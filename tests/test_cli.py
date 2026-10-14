"""`mlops` command line: the composition root that wires adapters into run_experiment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mlops.application.bundle import MANIFEST, MODEL, SPEC
from mlops.application.report import REPORT
from mlops.cli import main
from mlops.domain import compute_experiment_id
from mlops.config import load_experiment_yaml

ROOT = Path(__file__).parent.parent
SMOKE = ROOT / "experiments/configs/tiny_cnn_smoke.yaml"


def _write_yaml(path: Path, **overrides) -> Path:
    data = yaml.safe_load(SMOKE.read_text())
    for dotted, value in overrides.items():
        section, key = dotted.split(".")
        data[section][key] = value
    path.write_text(yaml.safe_dump(data))
    return path


# --- id ---


def test_id_prints_experiment_id(capsys):
    assert main(["id", str(SMOKE)]) == 0
    out = capsys.readouterr().out.strip()
    assert out == compute_experiment_id(load_experiment_yaml(SMOKE))


def test_id_of_unpinned_spec_fails_with_exit_2(tmp_path, capsys):
    path = _write_yaml(tmp_path / "e.yaml", **{"code.commit": None})
    assert main(["id", str(path)]) == 2
    assert "code.commit" in capsys.readouterr().err


def test_typo_in_yaml_fails_with_exit_2(tmp_path, capsys):
    path = _write_yaml(tmp_path / "e.yaml", **{"training.epoch": 3})
    assert main(["id", str(path)]) == 2
    assert "epoch" in capsys.readouterr().err


def test_missing_file_fails_with_exit_2(tmp_path, capsys):
    assert main(["id", str(tmp_path / "nope.yaml")]) == 2
    assert "nope.yaml" in capsys.readouterr().err


# --- run ---


def test_run_writes_bundle_and_prints_summary(tmp_path, capsys):
    art = tmp_path / "artifacts"
    rc = main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(art)])
    assert rc == 0
    out = capsys.readouterr().out
    exp_id = compute_experiment_id(load_experiment_yaml(SMOKE))
    assert exp_id in out
    assert "train/loss" in out and "val/loss" in out
    assert "FINISHED" in out

    (run_dir,) = art.iterdir()
    assert sorted(p.name for p in run_dir.iterdir()) == sorted([MANIFEST, MODEL, REPORT, SPEC])
    assert run_dir.name in out


def test_run_json_output(tmp_path, capsys):
    rc = main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "FINISHED"
    assert set(payload) >= {"experiment_id", "run_id", "attempt", "device", "history", "artifacts"}
    assert payload["artifacts"] == str(tmp_path / payload["run_id"])


def test_run_without_artifacts_dir_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(SMOKE), "--tracker", "inmemory"]) == 0
    assert list(tmp_path.iterdir()) == []


def test_run_uses_mlflow_when_spec_says_so(tmp_path, capsys):
    pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    rc = main(["run", str(SMOKE), "--mlflow-uri", uri, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    client = MlflowClient(uri)
    exp = client.get_experiment_by_name("smoke")
    runs = client.search_runs([exp.experiment_id], f"tags.mlops.run_id = '{payload['run_id']}'")
    assert len(runs) == 1 and runs[0].info.status == "FINISHED"


def test_run_mlflow_without_uri_is_a_clean_error(capsys):
    pytest.importorskip("mlflow")
    rc = main(["run", str(SMOKE)])  # spec says mlflow but has no uri, and no --mlflow-uri
    assert rc == 2
    err = capsys.readouterr().err
    assert "tracking.uri" in err
    assert "Traceback" not in err


def test_run_unknown_dataset_is_a_clean_error(tmp_path, capsys):
    path = _write_yaml(tmp_path / "e.yaml", **{"dataset.name": "msd-task03-liver"})
    rc = main(["run", str(path), "--tracker", "inmemory"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "msd-task03-liver" in err and "DataProvider" in err


def test_run_attempt_and_device_are_forwarded(tmp_path, capsys):
    args = ["run", str(SMOKE), "--tracker", "inmemory", "--json", "--attempt", "3", "--device", "cpu"]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["attempt"] == 3 and payload["device"] == "cpu"


def test_training_failure_exits_1_and_keeps_failed_bundle(tmp_path, capsys):
    # wrong channel count makes the first conv raise inside the epoch loop
    path = _write_yaml(tmp_path / "e.yaml", **{"model.params": {"in_channels": 3, "out_channels": 2, "hidden": 4, "spatial_dims": 3}})
    art = tmp_path / "artifacts"
    rc = main(["run", str(path), "--tracker", "inmemory", "--artifacts", str(art)])
    assert rc == 1
    assert "FAILED" in capsys.readouterr().err
    (run_dir,) = art.iterdir()
    assert sorted(p.name for p in run_dir.iterdir()) == sorted([MANIFEST, REPORT, SPEC])
    assert "FAILED" in (run_dir / REPORT).read_text().splitlines()[0]


def test_console_entry_point():
    r = subprocess.run(
        [sys.executable, "-m", "mlops", "id", str(SMOKE)], capture_output=True, text=True, cwd=ROOT
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == compute_experiment_id(load_experiment_yaml(SMOKE))


# --- launcher selection ---


def test_multi_worker_spec_needs_a_distributed_launcher(tmp_path, capsys):
    path = _write_yaml(tmp_path / "e.yaml", **{"resources.workers": 2})
    rc = main(["run", str(path), "--tracker", "inmemory"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "workers=2" in err and "--launcher ray" in err


def test_run_with_ray_launcher(tmp_path, capsys):
    pytest.importorskip("ray.train")
    import ray

    ray.init(num_cpus=8, include_dashboard=False, log_to_driver=False)
    try:
        path = _write_yaml(tmp_path / "e.yaml", **{"resources.workers": 2, "resources.cpu": 1})
        art = tmp_path / "artifacts"
        rc = main(["run", str(path), "--tracker", "inmemory", "--launcher", "ray",
                   "--ray-storage", str(tmp_path / "ray"), "--artifacts", str(art), "--json"])
        assert rc == 0, capsys.readouterr().err
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "FINISHED" and payload["launcher"] == "ray"
        (run_dir,) = art.iterdir()
        assert (run_dir / MODEL).exists()
    finally:
        ray.shutdown()


# --- rayjob ---


BASELINE = ROOT / "experiments/configs/swinunetr_baseline.yaml"


def test_rayjob_renders_json_list(capsys):
    assert main(["rayjob", str(BASELINE), "--namespace", "ml", "--artifacts-pvc", "runs"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["kind"] == "List"
    kinds = [i["kind"] for i in body["items"]]
    assert kinds == ["ConfigMap", "RayJob"]
    job = body["items"][1]
    assert job["metadata"]["namespace"] == "ml"
    assert "--mlflow-uri http://localhost:5000" in job["spec"]["entrypoint"]  # from tracking.uri


def test_rayjob_submit_goes_through_kubectl(capsys, monkeypatch):
    import mlops.cli as cli

    calls = []

    def fake_run(cmd, *, input, capture_output, text):
        calls.append(cmd)

        class R:
            returncode, stdout, stderr = 0, "", ""

        return R()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert main(["rayjob", str(BASELINE), "--submit", "--kubectl", "kubectl --context dev"]) == 0
    assert calls == [["kubectl", "--context", "dev", "apply", "-n", "default", "-f", "-"]]
    out = capsys.readouterr().out
    assert out.strip().startswith("submitted RayJob mlops-")


def test_rayjob_unpinned_spec_exits_2(tmp_path, capsys):
    path = _write_yaml(tmp_path / "e.yaml", **{"environment.image_digest": None})
    assert main(["rayjob", str(path)]) == 2
    assert "image_digest" in capsys.readouterr().err


# --- reproduce ---
#
# `mlops reproduce RUN_ID --artifacts DIR`: re-run the bundle and compare.
# Exit 0 reproduced / 1 mismatch (details printed) / 2 bad input.


@pytest.fixture
def finished_run(tmp_path, capsys):
    """A FINISHED bundle under tmp_path/artifacts; returns (artifacts_dir, run_id)."""
    art = tmp_path / "artifacts"
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(art), "--json"]) == 0
    return art, json.loads(capsys.readouterr().out)["run_id"]


def test_reproduce_intact_bundle_exits_0(finished_run, capsys):
    art, run_id = finished_run
    rc = main(["reproduce", run_id, "--artifacts", str(art), "--tracker", "inmemory"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "REPRODUCED" in out and run_id in out
    for check in ("run_id", "history", "weights"):
        assert check in out


def test_reproduce_json_output(finished_run, capsys):
    art, run_id = finished_run
    rc = main(["reproduce", run_id, "--artifacts", str(art), "--tracker", "inmemory", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["run_id"] == run_id
    assert payload["checks"] == {"run_id": True, "history": True, "weights": True}
    assert payload["mismatches"] == []


def test_reproduce_tampered_bundle_exits_1_and_names_the_mismatch(finished_run, capsys):
    art, run_id = finished_run
    manifest = art / run_id / MANIFEST
    m = json.loads(manifest.read_text())
    m["history"][0]["train/loss"] += 1.0
    manifest.write_text(json.dumps(m))

    rc = main(["reproduce", run_id, "--artifacts", str(art), "--tracker", "inmemory"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "NOT REPRODUCED" in out
    assert "history" in out and "train/loss" in out


def test_reproduce_atol_is_forwarded(finished_run, capsys):
    import torch

    art, run_id = finished_run
    path = art / run_id / MODEL
    state = torch.load(path, map_location="cpu", weights_only=True)
    key = next(iter(state))
    state[key] = state[key] + 1e-7
    torch.save(state, path)

    assert main(["reproduce", run_id, "--artifacts", str(art), "--tracker", "inmemory"]) == 1
    capsys.readouterr()
    assert main(["reproduce", run_id, "--artifacts", str(art), "--tracker", "inmemory", "--atol", "1e-5"]) == 0


def test_reproduce_unknown_run_id_exits_2(tmp_path, capsys):
    rc = main(["reproduce", "deadbeef", "--artifacts", str(tmp_path), "--tracker", "inmemory"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "deadbeef" in err and "Traceback" not in err


def test_reproduce_failed_run_exits_2(tmp_path, capsys):
    path = _write_yaml(tmp_path / "e.yaml", **{"model.params": {"in_channels": 3, "out_channels": 2, "hidden": 4, "spatial_dims": 3}})
    art = tmp_path / "artifacts"
    assert main(["run", str(path), "--tracker", "inmemory", "--artifacts", str(art)]) == 1
    capsys.readouterr()
    (run_dir,) = art.iterdir()

    rc = main(["reproduce", run_dir.name, "--artifacts", str(art), "--tracker", "inmemory"])
    assert rc == 2
    assert "FAILED" in capsys.readouterr().err


# --- report ---
#
# `mlops run --artifacts` writes report.md next to the bundle; `mlops report RUN_ID`
# re-renders it from the bundle (stdout, or --out FILE).


def test_run_writes_a_report_rendered_from_the_bundle(finished_run):
    art, run_id = finished_run
    md = (art / run_id / REPORT).read_text()
    assert md.startswith("# tiny-cnn-smoke")
    assert run_id in md and "FINISHED" in md


def test_report_prints_markdown_from_the_bundle(finished_run, capsys):
    art, run_id = finished_run
    assert main(["report", run_id, "--artifacts", str(art)]) == 0
    out = capsys.readouterr().out
    assert out == (art / run_id / REPORT).read_text()


def test_report_out_writes_file(finished_run, tmp_path, capsys):
    art, run_id = finished_run
    out_file = tmp_path / "r.md"
    assert main(["report", run_id, "--artifacts", str(art), "--out", str(out_file)]) == 0
    assert out_file.read_text() == (art / run_id / REPORT).read_text()


def test_report_regenerates_when_report_md_is_missing(finished_run, capsys):
    art, run_id = finished_run
    expected = (art / run_id / REPORT).read_text()
    (art / run_id / REPORT).unlink()
    assert main(["report", run_id, "--artifacts", str(art)]) == 0
    assert capsys.readouterr().out == expected


def test_report_unknown_run_id_exits_2(tmp_path, capsys):
    assert main(["report", "deadbeef", "--artifacts", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "deadbeef" in err and "Traceback" not in err
