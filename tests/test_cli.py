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
    assert main(["rayjob", str(BASELINE), "--namespace", "ml", "--artifacts-pvc", "runs", "--data-pvc", "msd"]) == 0
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
    assert main(["rayjob", str(BASELINE), "--data-pvc", "msd", "--submit", "--kubectl", "kubectl --context dev"]) == 0
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


# --- environment verification ---
#
# `mlops run` records the actual environment in manifest.json and warns on stderr when it
# does not match the pins; `--strict-env` refuses to start unless every check is ok.


@pytest.fixture
def probe(monkeypatch):
    """Swap the CLI's probe for a FixedProbe; returns a setter."""
    import mlops.cli as cli
    from tests.test_environment import FixedProbe

    def use(**kw):
        monkeypatch.setattr(cli, "_probe", lambda: FixedProbe(**kw))
    use()
    return use


def test_run_records_environment_in_manifest(tmp_path, capsys):
    art = tmp_path / "artifacts"
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(art), "--json"]) == 0
    run_id = json.loads(capsys.readouterr().out)["run_id"]
    m = json.loads((art / run_id / MANIFEST).read_text())
    assert m["environment"]["actual"]["packages"]["torch"]
    assert set(m["environment"]["checks"]) == {"code.commit", "environment.image_digest"}


def test_run_warns_when_environment_is_not_verified(tmp_path, capsys, probe):
    probe(commit_ok=False)
    assert main(["run", str(SMOKE), "--tracker", "inmemory"]) == 0
    err = capsys.readouterr().err
    assert "code.commit" in err and "mismatch" in err and "mlops/x.py" in err


def test_strict_env_refuses_an_unverified_environment(tmp_path, capsys, probe):
    probe(commit_ok=False)
    art = tmp_path / "artifacts"
    rc = main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(art), "--strict-env"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--strict-env" in err and "code.commit" in err and "Traceback" not in err
    assert not art.exists() or not any(art.iterdir())


def test_strict_env_runs_when_everything_verifies(tmp_path, capsys, probe, monkeypatch):
    import mlops.cli as cli
    from tests.test_data_snapshot import HashedBlobs

    spec = load_experiment_yaml(SMOKE)
    probe(digest=spec.environment.image_digest)
    monkeypatch.setitem(cli._DATA_PROVIDERS, "synthetic-blobs", lambda: HashedBlobs(spec.dataset.snapshot))
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--strict-env"]) == 0
    assert "code.commit" not in capsys.readouterr().err


# --- checkpoint + resume ---


def _killed_run(tmp_path, capsys, monkeypatch, *, epochs=3, fail_at=3):
    """`mlops run --checkpoint-every 1` that dies at epoch ``fail_at``; returns (art, run_id)."""
    import mlops.cli as cli
    from tests.test_checkpoint import FlakyTracker

    yml = _write_yaml(tmp_path / "e.yaml", **{"training.epochs": epochs})
    art = tmp_path / "artifacts"
    monkeypatch.setattr(cli, "_tracker", lambda *a: FlakyTracker(fail_at))
    assert main(["run", str(yml), "--artifacts", str(art), "--checkpoint-every", "1"]) == 1
    monkeypatch.undo()
    capsys.readouterr()
    (run_dir,) = art.iterdir()
    return art, run_dir.name


def test_run_checkpoint_every_writes_checkpoint(tmp_path, capsys):
    from mlops.application.bundle import CHECKPOINT

    art = tmp_path / "artifacts"
    yml = _write_yaml(tmp_path / "e.yaml", **{"training.epochs": 2})
    assert main(["run", str(yml), "--tracker", "inmemory", "--artifacts", str(art),
                 "--checkpoint-every", "1", "--json"]) == 0
    run_id = json.loads(capsys.readouterr().out)["run_id"]
    assert (art / run_id / CHECKPOINT).is_file()


def test_run_checkpoint_every_needs_artifacts(capsys):
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--checkpoint-every", "1"]) == 2
    assert "--artifacts" in capsys.readouterr().err


def test_run_checkpoint_every_refused_with_ray(tmp_path, capsys):
    rc = main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(tmp_path),
               "--launcher", "ray", "--checkpoint-every", "1"])
    assert rc == 2
    assert "local launcher" in capsys.readouterr().err


def test_resume_finishes_a_killed_run(tmp_path, capsys, monkeypatch):
    art, run_id = _killed_run(tmp_path, capsys, monkeypatch)
    assert main(["resume", run_id, "--artifacts", str(art), "--tracker", "inmemory", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "FINISHED" and out["run_id"] == run_id
    assert out["resumed_from_epoch"] == 2
    assert len(out["history"]) == 3
    m = json.loads((art / run_id / MANIFEST).read_text())
    assert m["status"] == "FINISHED" and [r["epoch"] for r in m["resumes"]] == [2]
    assert "resumed" in (art / run_id / REPORT).read_text()


def test_resume_human_output_names_the_resume_point(tmp_path, capsys, monkeypatch):
    art, run_id = _killed_run(tmp_path, capsys, monkeypatch)
    assert main(["resume", run_id, "--artifacts", str(art), "--tracker", "inmemory"]) == 0
    assert "resumed after epoch 2" in capsys.readouterr().out


def test_resume_unknown_run_exits_2(tmp_path, capsys):
    rc = main(["resume", "f" * 64, "--artifacts", str(tmp_path), "--tracker", "inmemory"])
    assert rc == 2
    assert "Traceback" not in capsys.readouterr().err


def test_resume_finished_run_exits_2(finished_run, capsys):
    art, run_id = finished_run
    assert main(["resume", run_id, "--artifacts", str(art), "--tracker", "inmemory"]) == 2
    assert "no checkpoint" in capsys.readouterr().err


# --- diff ---
#
# Exit 0 same experiment (fingerprints identical) / 1 different experiments / 2 bad input.


def _run_json(capsys, *argv) -> str:
    assert main(["run", *argv, "--tracker", "inmemory", "--json"]) == 0
    return json.loads(capsys.readouterr().out)["run_id"]


def test_diff_same_experiment_exits_0(tmp_path, capsys):
    art = tmp_path / "artifacts"
    a = _run_json(capsys, str(SMOKE), "--artifacts", str(art))
    b = _run_json(capsys, str(SMOKE), "--artifacts", str(art), "--attempt", "2")
    assert main(["diff", a, b, "--artifacts", str(art)]) == 0
    out = capsys.readouterr().out
    assert "same experiment" in out and "attempt" in out


def test_diff_names_the_changed_input_and_exits_1(tmp_path, capsys):
    art = tmp_path / "artifacts"
    a = _run_json(capsys, str(SMOKE), "--artifacts", str(art))
    b = _run_json(capsys, str(_write_yaml(tmp_path / "s.yaml", **{"training.seed": 7})), "--artifacts", str(art))
    assert main(["diff", a, b, "--artifacts", str(art)]) == 1
    out = capsys.readouterr().out
    assert "different experiments" in out
    assert "training.seed" in out and "0 -> 7" in out
    assert "val/loss" in out


def test_diff_json(tmp_path, capsys):
    art = tmp_path / "artifacts"
    a = _run_json(capsys, str(SMOKE), "--artifacts", str(art))
    b = _run_json(capsys, str(_write_yaml(tmp_path / "s.yaml", **{"training.seed": 7})), "--artifacts", str(art))
    assert main(["diff", a, b, "--artifacts", str(art), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["experiment"] == [{"path": "training.seed", "a": 0, "b": 7}]


def test_diff_across_two_artifact_roots(tmp_path, capsys):
    a = _run_json(capsys, str(SMOKE), "--artifacts", str(tmp_path / "one"))
    b = _run_json(capsys, str(SMOKE), "--artifacts", str(tmp_path / "two"))
    assert main(["diff", a, b, "--artifacts", str(tmp_path / "one"), "--artifacts-b", str(tmp_path / "two")]) == 0


def test_diff_unknown_run_exits_2(tmp_path, capsys):
    assert main(["diff", "a" * 64, "b" * 64, "--artifacts", str(tmp_path)]) == 2
    assert "Traceback" not in capsys.readouterr().err


# --- list ---


def test_list_groups_attempts_and_says_they_agree(tmp_path, capsys):
    art = tmp_path / "artifacts"
    a = _run_json(capsys, str(SMOKE), "--artifacts", str(art))
    _run_json(capsys, str(SMOKE), "--artifacts", str(art), "--attempt", "2")
    _run_json(capsys, str(_write_yaml(tmp_path / "s.yaml", **{"training.seed": 7})), "--artifacts", str(art))
    assert main(["list", "--artifacts", str(art)]) == 0
    out = capsys.readouterr().out
    assert out.count("tiny-cnn-smoke") == 2  # one header per experiment
    assert "2 finished, final metrics identical" in out
    assert a[:12] in out and "FINISHED" in out


def test_list_filters_by_experiment(tmp_path, capsys):
    art = tmp_path / "artifacts"
    _run_json(capsys, str(SMOKE), "--artifacts", str(art))
    _run_json(capsys, str(_write_yaml(tmp_path / "s.yaml", **{"training.seed": 7})), "--artifacts", str(art))
    exp = compute_experiment_id(load_experiment_yaml(SMOKE))
    assert main(["list", "--artifacts", str(art), "--experiment", exp[:12], "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [g["experiment_id"] for g in out] == [exp]
    assert out[0]["runs"][0]["status"] == "FINISHED"


def test_list_missing_root_exits_2(tmp_path, capsys):
    assert main(["list", "--artifacts", str(tmp_path / "nope")]) == 2
    assert "no such directory" in capsys.readouterr().err


def test_list_no_match_says_so(tmp_path, capsys):
    tmp_path.joinpath("artifacts").mkdir()
    assert main(["list", "--artifacts", str(tmp_path / "artifacts")]) == 0
    assert "no runs" in capsys.readouterr().out


# --- following a RayJob ---
#
# `mlops rayjob-status NAME`: exit 0 succeeded or still going / 1 failed or stopped / 2 kubectl error.


def _fake_kubectl(monkeypatch, stdout="", returncode=0, stderr=""):
    import mlops.cli as cli

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)

        class R:
            pass

        R.returncode, R.stdout, R.stderr = returncode, stdout, stderr
        return R()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls


def _job(status, deployment="Running", message=""):
    return json.dumps({"kind": "RayJob", "metadata": {"name": "mlops-abc-a1", "annotations": {"mlops.io/run-id": "r" * 64}},
                       "status": {"jobStatus": status, "jobDeploymentStatus": deployment, "message": message}})


@pytest.mark.parametrize("status,rc", [("RUNNING", 0), ("SUCCEEDED", 0), ("FAILED", 1), ("STOPPED", 1)])
def test_rayjob_status_exit_codes(monkeypatch, capsys, status, rc):
    calls = _fake_kubectl(monkeypatch, _job(status, message="boom" if status == "FAILED" else ""))
    assert main(["rayjob-status", "mlops-abc-a1", "--namespace", "ml", "--kubectl", "kubectl --context dev"]) == rc
    assert calls[0][:3] == ["kubectl", "--context", "dev"]
    out = capsys.readouterr().out
    assert status in out and ("r" * 12) in out
    if status == "FAILED":
        assert "boom" in out


def test_rayjob_status_json(monkeypatch, capsys):
    _fake_kubectl(monkeypatch, _job("SUCCEEDED", "Complete"))
    assert main(["rayjob-status", "mlops-abc-a1", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["job_status"] == "SUCCEEDED" and out["run_id"] == "r" * 64


def test_rayjob_status_kubectl_error_exits_2(monkeypatch, capsys):
    _fake_kubectl(monkeypatch, returncode=1, stderr="Error from server (NotFound)")
    assert main(["rayjob-status", "nope"]) == 2
    err = capsys.readouterr().err
    assert "NotFound" in err and "Traceback" not in err


def test_rayjob_logs_follows_and_passes_the_exit_code(monkeypatch):
    calls = _fake_kubectl(monkeypatch, returncode=0)
    assert main(["rayjob-logs", "mlops-abc-a1", "--namespace", "ml", "-f"]) == 0
    assert calls == [["kubectl", "logs", "-n", "ml", "job/mlops-abc-a1", "--follow"]]


def test_rayjob_submit_says_how_to_follow_it(capsys, monkeypatch):
    _fake_kubectl(monkeypatch)
    assert main(["rayjob", str(BASELINE), "--data-pvc", "msd", "--submit", "--namespace", "ml"]) == 0
    out = capsys.readouterr().out
    assert "mlops rayjob-status mlops-" in out and "mlops rayjob-logs mlops-" in out and "--namespace ml" in out


def test_rayjob_for_a_file_dataset_needs_a_data_pvc(capsys):
    # msd-task09-spleen reads files; without a volume the in-cluster run could only fail later.
    assert main(["rayjob", str(BASELINE)]) == 2
    err = capsys.readouterr().err
    assert "--data-pvc" in err and "msd-task09-spleen" in err


def test_rayjob_data_pvc_reaches_the_manifest(capsys):
    assert main(["rayjob", str(BASELINE), "--data-pvc", "msd", "--data-path", "/mnt/msd"]) == 0
    job = json.loads(capsys.readouterr().out)["items"][1]
    assert "--data-root /mnt/msd" in job["spec"]["entrypoint"]


def test_rayjob_for_synthetic_data_needs_no_volume(capsys):
    assert main(["rayjob", str(SMOKE)]) == 0
