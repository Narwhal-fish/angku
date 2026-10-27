"""Verify the data snapshot instead of trusting the yaml.

``DataProvider.snapshot(dataset)`` returns the content id of what the provider would serve
(``sha256:<64hex>``), or None when it cannot attest to its data (e.g. generated data).
``check_snapshot(pinned, served)`` (domain, pure) -> ``EnvCheck``:

    ok        pinned is a content hash and equals what is served
    mismatch  both are content hashes and differ  -> the runner raises DataError, nothing runs
    unknown   provider does not hash, or the pin is not a content hash (e.g. ``dvc:...``)

The result is stored as ``manifest.data_snapshot``. ``mlops snapshot <yaml>`` prints the
served id so it can be pinned.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.application.bundle import load_bundle
from mlops.cli import main
from mlops.config import load_experiment_yaml
from mlops.domain import compute_experiment_id, compute_run_id
from mlops.domain.environment import check_snapshot
from mlops.ports.artifacts import ArtifactError
from mlops.ports.data import DataError
from mlops.registry import discover

ROOT = Path(__file__).parent.parent
SMOKE = ROOT / "experiments/configs/tiny_cnn_smoke.yaml"
A = "sha256:" + "a" * 64
B = "sha256:" + "b" * 64


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


@pytest.fixture
def spec():
    return load_experiment_yaml(SMOKE)


def _with_snapshot(spec, snapshot: str):
    return spec.model_copy(update={"dataset": spec.dataset.model_copy(update={"snapshot": snapshot})})


class HashedBlobs(SyntheticBlobs):
    """SyntheticBlobs that claims to serve ``served``."""

    def __init__(self, served: str | None) -> None:
        super().__init__(n_train=4, n_val=2)
        self.served = served

    def snapshot(self, dataset):
        return self.served


# --- check_snapshot is pure ---


def test_equal_content_hashes_are_ok():
    assert check_snapshot(A, A).status == "ok"


def test_different_content_hashes_are_a_mismatch_naming_both():
    c = check_snapshot(A, B)
    assert c.status == "mismatch" and A in c.detail and B in c.detail


def test_provider_that_does_not_hash_is_unknown():
    assert check_snapshot(A, None).status == "unknown"


def test_non_content_pin_is_unknown_and_suggests_the_served_hash():
    c = check_snapshot("dvc:8d1f2c3a", B)
    assert c.status == "unknown" and B in c.detail


# --- runner ---


def test_mismatched_data_never_starts_the_run(spec):
    tracker, store = InMemoryTracker(), InMemoryArtifactStore()
    with pytest.raises(DataError, match="dataset.snapshot"):
        run_experiment(_with_snapshot(spec, A), data=HashedBlobs(B), tracker=tracker, artifacts=store)
    assert tracker.runs == []
    run_id = compute_run_id(compute_experiment_id(_with_snapshot(spec, A)), spec.resources, 1)
    with pytest.raises(ArtifactError):
        store.list(run_id)


def test_verified_snapshot_is_recorded(spec):
    store = InMemoryArtifactStore()
    r = run_experiment(_with_snapshot(spec, A), data=HashedBlobs(A), tracker=InMemoryTracker(), artifacts=store)
    assert load_bundle(store, r.run_id).manifest.data_snapshot.status == "ok"
    assert r.data_snapshot.status == "ok"


def test_unhashed_data_is_recorded_as_unknown(spec):
    store = InMemoryArtifactStore()
    r = run_experiment(spec, data=SyntheticBlobs(n_train=4, n_val=2), tracker=InMemoryTracker(), artifacts=store)
    assert load_bundle(store, r.run_id).manifest.data_snapshot.status == "unknown"


# --- msd-task09-spleen hashes its files ---


@pytest.fixture
def spleen_root(tmp_path):
    pytest.importorskip("nibabel")
    from tests.test_msd_spleen import _write_case

    task = tmp_path / "data" / "Task09_Spleen"
    (task / "imagesTr").mkdir(parents=True)
    (task / "labelsTr").mkdir()
    training = [_write_case(task, i, spleen_at=(4 + i, 5, 3)) for i in range(3)]
    (task / "dataset.json").write_text(json.dumps({"training": training}))
    return tmp_path / "data"


def _spleen_snapshot(root: Path) -> str:
    from mlops.domain.experiment import Dataset
    from mlops.registry import DATASETS

    p = DATASETS.build("msd-task09-spleen", {"root": str(root)})
    return p.snapshot(Dataset(name="msd-task09-spleen", snapshot="x"))


def test_spleen_snapshot_is_a_stable_content_hash(spleen_root):
    s = _spleen_snapshot(spleen_root)
    assert s.startswith("sha256:") and len(s) == len("sha256:") + 64
    assert _spleen_snapshot(spleen_root) == s


def test_spleen_snapshot_does_not_depend_on_where_the_data_lives(spleen_root, tmp_path):
    moved = tmp_path / "elsewhere"
    shutil.copytree(spleen_root, moved)
    assert _spleen_snapshot(moved) == _spleen_snapshot(spleen_root)


def test_spleen_snapshot_changes_when_a_label_changes(spleen_root):
    before = _spleen_snapshot(spleen_root)
    label = spleen_root / "Task09_Spleen/labelsTr/spleen_1.nii.gz"
    label.write_bytes(label.read_bytes() + b"\0")
    assert _spleen_snapshot(spleen_root) != before


def test_spleen_snapshot_changes_when_a_case_is_added(spleen_root):
    from tests.test_msd_spleen import _write_case

    before = _spleen_snapshot(spleen_root)
    task = spleen_root / "Task09_Spleen"
    index = json.loads((task / "dataset.json").read_text())
    index["training"].append(_write_case(task, 9, spleen_at=(4, 5, 3)))
    (task / "dataset.json").write_text(json.dumps(index))
    assert _spleen_snapshot(spleen_root) != before


def test_spleen_snapshot_ignores_files_it_does_not_serve(spleen_root):
    before = _spleen_snapshot(spleen_root)
    (spleen_root / "Task09_Spleen/imagesTs").mkdir()
    (spleen_root / "Task09_Spleen/imagesTs/test_1.nii.gz").write_bytes(b"not used")
    (spleen_root / "Task09_Spleen/.DS_Store").write_bytes(b"junk")
    assert _spleen_snapshot(spleen_root) == before


# --- CLI ---


def _spleen_yaml(tmp_path: Path, snapshot: str) -> Path:
    import yaml

    from tests.test_msd_spleen import PRE

    d = yaml.safe_load(SMOKE.read_text())
    d["dataset"] = {"name": "msd-task09-spleen", "version": "v1", "snapshot": snapshot}
    d["preprocessing"] = PRE
    path = tmp_path / "spleen.yaml"
    path.write_text(yaml.safe_dump(d))
    return path


def test_cli_snapshot_prints_the_id_to_pin(spleen_root, tmp_path, capsys):
    path = _spleen_yaml(tmp_path, "dvc:unpinned-yet")
    assert main(["snapshot", str(path), "--data-root", str(spleen_root)]) == 0
    assert capsys.readouterr().out.strip() == _spleen_snapshot(spleen_root)


def test_cli_snapshot_for_unhashed_data_exits_2(capsys):
    assert main(["snapshot", str(SMOKE)]) == 2
    assert "synthetic-blobs" in capsys.readouterr().err


def test_cli_run_with_pinned_spleen_snapshot_verifies(spleen_root, tmp_path, capsys):
    path = _spleen_yaml(tmp_path, _spleen_snapshot(spleen_root))
    art = tmp_path / "art"
    rc = main(
        ["run", str(path), "--tracker", "inmemory", "--data-root", str(spleen_root), "--artifacts", str(art), "--json"]
    )
    assert rc == 0
    run_id = json.loads(capsys.readouterr().out)["run_id"]
    assert json.loads((art / run_id / "manifest.json").read_text())["data_snapshot"]["status"] == "ok"


def test_cli_run_with_wrong_snapshot_exits_2(spleen_root, tmp_path, capsys):
    path = _spleen_yaml(tmp_path, A)
    art = tmp_path / "art"
    rc = main(["run", str(path), "--tracker", "inmemory", "--data-root", str(spleen_root), "--artifacts", str(art)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "dataset.snapshot" in err and A in err and "Traceback" not in err
    assert not art.exists() or not any(art.iterdir())


def test_cli_warns_about_unverified_data(capsys):
    assert main(["run", str(SMOKE), "--tracker", "inmemory"]) == 0
    assert "dataset.snapshot: unknown" in capsys.readouterr().err


def test_cli_strict_env_refuses_unverified_data(capsys, monkeypatch):
    import mlops.cli as cli
    from tests.test_environment import FixedProbe

    spec = load_experiment_yaml(SMOKE)
    monkeypatch.setattr(cli, "_probe", lambda: FixedProbe(digest=spec.environment.image_digest))
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--strict-env"]) == 2
    assert "dataset.snapshot" in capsys.readouterr().err
