"""``msd-task09-spleen``: the first real DataProvider, registered from ``experiments/`` as a plugin.

Layout it reads (Medical Segmentation Decathlon, Task09):

    <data_root>/Task09_Spleen/dataset.json         {"training": [{"image": "./imagesTr/x.nii.gz", "label": ...}]}
    <data_root>/Task09_Spleen/imagesTr/*.nii.gz
    <data_root>/Task09_Spleen/labelsTr/*.nii.gz

Deterministic preprocessing (done once, at load): orient RAS -> resample to ``spacing`` ->
clip+scale ``intensity_range`` to [0, 1] -> crop to body foreground -> pad up to ``roi_size``.
Train items are random ``roi_size`` crops (pos/neg label sampling, torch global RNG, so the
run seed decides them); val items are a fixed crop centred on the label. Split: cases sorted
by file name, the last ``num_val`` are validation.

These tests build a tiny fake Task09 tree; the real 1.5 GB dataset is never needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")
pytest.importorskip("monai")

from mlops.adapters.inmemory_tracker import InMemoryTracker  # noqa: E402
from mlops.application import run_experiment  # noqa: E402
from mlops.cli import main  # noqa: E402
from mlops.config import load_experiment_yaml  # noqa: E402
from mlops.domain.experiment import Dataset  # noqa: E402
from mlops.domain.split import SplitAssignment  # noqa: E402
from mlops.ports.data import DataError  # noqa: E402
from mlops.registry import DATASETS, RegistryError, discover  # noqa: E402

ROOT = Path(__file__).parent.parent
NAME = "msd-task09-spleen"
SPACING = (4.0, 4.0, 4.0)  # voxel size of the fake source volumes (mm)
SHAPE = (20, 20, 12)
PRE = {
    "spacing": [4.0, 4.0, 4.0],
    "intensity_range": [-57, 164],
    "roi_size": [8, 8, 8],
    "num_val": 1,
}


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


def _write_case(task: Path, i: int, *, spleen_at: tuple[int, int, int]) -> dict[str, str]:
    """CT-like volume: air (-1000) around a body box (40 HU) with a 'spleen' cube (120 HU)."""
    img = np.full(SHAPE, -1000.0, dtype=np.float32)
    img[2:-2, 2:-2, 1:-1] = 40.0
    lbl = np.zeros(SHAPE, dtype=np.uint8)
    x, y, z = spleen_at
    img[x:x + 3, y:y + 3, z:z + 3] = 120.0
    lbl[x:x + 3, y:y + 3, z:z + 3] = 1
    affine = np.diag([*SPACING, 1.0])
    name = f"spleen_{i}.nii.gz"
    nib.save(nib.Nifti1Image(img, affine), task / "imagesTr" / name)
    nib.save(nib.Nifti1Image(lbl, affine), task / "labelsTr" / name)
    return {"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"}


@pytest.fixture
def data_root(tmp_path) -> Path:
    task = tmp_path / "data" / "Task09_Spleen"
    (task / "imagesTr").mkdir(parents=True)
    (task / "labelsTr").mkdir()
    training = [_write_case(task, i, spleen_at=(4 + i, 5, 3)) for i in range(4)]
    (task / "dataset.json").write_text(json.dumps({"name": "Spleen", "training": training}))
    return tmp_path / "data"


@pytest.fixture
def provider(data_root):
    return DATASETS.build(NAME, {"root": str(data_root)})


def _ds(name: str = NAME) -> Dataset:
    return Dataset(name=name, version="v1", snapshot="test-snapshot")


# --- registration & config ---


def test_registered_as_a_dataset_plugin():
    assert NAME in DATASETS


def test_provider_config_needs_a_root():
    with pytest.raises(RegistryError, match="root"):
        DATASETS.build(NAME, {})


def test_typo_in_preprocessing_is_rejected(provider):
    with pytest.raises(DataError, match="roi_sise"):
        provider.load(_ds(), {**PRE, "roi_sise": [8, 8, 8]})


def test_other_dataset_name_is_refused(provider):
    with pytest.raises(DataError, match="synthetic-blobs"):
        provider.load(_ds("synthetic-blobs"), PRE)


def test_missing_dataset_dir_names_the_path(tmp_path):
    p = DATASETS.build(NAME, {"root": str(tmp_path)})
    with pytest.raises(DataError, match="Task09_Spleen"):
        p.load(_ds(), PRE)


def test_num_val_must_leave_training_cases(provider):
    with pytest.raises(DataError, match="num_val"):
        provider.load(_ds(), {**PRE, "num_val": 4})


# --- split ---


def test_last_num_val_cases_by_name_are_validation(provider):
    splits = provider.load(_ds(), {**PRE, "num_val": 2})
    assert len(splits.train) == 2 and len(splits.val) == 2
    assert [c.name for c in splits.train.cases] == ["spleen_0", "spleen_1"]
    assert [c.name for c in splits.val.cases] == ["spleen_2", "spleen_3"]



# Without num_val: with `split` the platform chooses val.
NO_NUM_VAL = {k: v for k, v in PRE.items() if k != "num_val"}


def test_cases_are_image_names_without_extension(provider):
    assert provider.cases(_ds()) == ["spleen_0", "spleen_1", "spleen_2", "spleen_3"]


def test_spec_split_picks_the_cases_and_serves_test_like_val(provider):
    split = SplitAssignment(train=["spleen_1", "spleen_3"], val=["spleen_0"], test=["spleen_2"])
    splits = provider.load(_ds(), NO_NUM_VAL, split=split)
    assert [c.name for c in splits.train.cases] == ["spleen_1", "spleen_3"]
    assert [c.name for c in splits.val.cases] == ["spleen_0"]
    assert [c.name for c in splits.test.cases] == ["spleen_2"]
    assert type(splits.test) is type(splits.val)
    full = provider.load(_ds(), {**NO_NUM_VAL, "val": "full_volume"}, split=split)
    assert type(full.test).__name__ == "FullVolumes"


def test_spec_split_without_test_serves_none(provider):
    splits = provider.load(_ds(), NO_NUM_VAL, split=SplitAssignment(train=["spleen_0", "spleen_1"], val=["spleen_2", "spleen_3"]))
    assert splits.test is None


def test_spec_split_and_num_val_together_are_refused(provider):
    split = SplitAssignment(train=["spleen_0", "spleen_1"], val=["spleen_2"], test=["spleen_3"])
    with pytest.raises(DataError, match="num_val"):
        provider.load(_ds(), {**PRE, "num_val": 1}, split=split)


def test_spec_split_naming_an_unknown_case_is_refused(provider):
    with pytest.raises(DataError, match="spleen_9"):
        provider.load(_ds(), NO_NUM_VAL, split=SplitAssignment(train=["spleen_9"], val=["spleen_0"]))


# --- samples ---


def test_train_sample_matches_runner_contract(provider):
    torch.manual_seed(0)
    x, y = provider.load(_ds(), PRE).train[0]
    assert type(x) is torch.Tensor and type(y) is torch.Tensor  # plain tensors, not MetaTensor
    assert x.shape == (1, 8, 8, 8) and x.dtype == torch.float32
    assert y.shape == (8, 8, 8) and y.dtype == torch.long
    assert set(y.unique().tolist()) <= {0, 1}


def test_intensity_is_clipped_and_scaled_to_unit_range(provider):
    torch.manual_seed(0)
    train = provider.load(_ds(), PRE).train
    x = torch.cat([train[i][0].flatten() for i in range(len(train))])
    assert x.min() >= 0.0 and x.max() <= 1.0
    # spleen 120 HU -> (120 + 57) / (164 + 57)
    assert torch.isclose(x.max(), torch.tensor(177 / 221), atol=1e-4)


def test_positive_only_sampling_always_hits_the_label(provider):
    torch.manual_seed(0)
    train = provider.load(_ds(), {**PRE, "pos": 1.0, "neg": 0.0}).train
    for _ in range(5):
        for i in range(len(train)):
            assert train[i][1].sum() > 0


def test_train_crops_follow_the_torch_seed(provider):
    train = provider.load(_ds(), {**PRE, "pos": 1.0, "neg": 1.0}).train

    def draw(seed: int) -> list[torch.Tensor]:
        torch.manual_seed(seed)
        return [train[i % len(train)][0] for i in range(12)]

    a, b, c = draw(0), draw(0), draw(1)
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert not all(torch.equal(p, q) for p, q in zip(a, c))


def test_val_crop_is_fixed_and_contains_the_label(provider):
    val = provider.load(_ds(), PRE).val
    torch.manual_seed(0)
    x0, y0 = val[0]
    torch.manual_seed(1)
    x1, y1 = val[0]
    assert torch.equal(x0, x1) and torch.equal(y0, y1)
    assert y0.sum() > 0


def test_resampling_to_finer_spacing_grows_the_volume(provider):
    coarse = provider.load(_ds(), PRE).val.cases[0].image.shape
    fine = provider.load(_ds(), {**PRE, "spacing": [2.0, 2.0, 2.0]}).val.cases[0].image.shape
    assert all(f > c for f, c in zip(fine[1:], coarse[1:]))


def test_volume_smaller_than_roi_is_padded(provider):
    torch.manual_seed(0)
    x, y = provider.load(_ds(), {**PRE, "roi_size": [32, 32, 32]}).train[0]
    assert x.shape == (1, 32, 32, 32) and y.shape == (32, 32, 32)


# --- end to end ---


def _spleen_spec(path: Path):
    spec = load_experiment_yaml(ROOT / "experiments/configs/tiny_cnn_smoke.yaml")
    return spec.model_copy(update={
        "dataset": _ds(),
        "preprocessing": PRE,
        "training": spec.training.model_copy(update={"epochs": 2}),
    })


def test_runs_end_to_end_and_reproduces(provider, tmp_path):
    spec = _spleen_spec(tmp_path)
    a = run_experiment(spec, data=provider, tracker=InMemoryTracker())
    b = run_experiment(spec, data=provider, tracker=InMemoryTracker())
    assert len(a.history) == 2 and "val/loss" in a.history[0]
    assert a.history == b.history


def test_cli_run_with_data_root(data_root, tmp_path, capsys):
    import yaml

    spec = _spleen_spec(tmp_path)
    path = tmp_path / "spleen.yaml"
    path.write_text(yaml.safe_dump(json.loads(spec.model_dump_json())))
    rc = main(["run", str(path), "--tracker", "inmemory", "--data-root", str(data_root), "--json"])
    assert rc == 0, capsys.readouterr().err
    assert json.loads(capsys.readouterr().out)["status"] == "FINISHED"


def test_cli_run_without_data_root_is_a_clean_error(tmp_path, capsys, monkeypatch):
    import yaml

    monkeypatch.delenv("MLOPS_DATA_ROOT", raising=False)
    path = tmp_path / "spleen.yaml"
    path.write_text(yaml.safe_dump(json.loads(_spleen_spec(tmp_path).model_dump_json())))
    assert main(["run", str(path), "--tracker", "inmemory"]) == 2
    err = capsys.readouterr().err
    assert "--data-root" in err and "MLOPS_DATA_ROOT" in err and "Traceback" not in err


def test_cli_data_root_from_env(data_root, tmp_path, capsys, monkeypatch):
    import yaml

    monkeypatch.setenv("MLOPS_DATA_ROOT", str(data_root))
    path = tmp_path / "spleen.yaml"
    path.write_text(yaml.safe_dump(json.loads(_spleen_spec(tmp_path).model_dump_json())))
    assert main(["run", str(path), "--tracker", "inmemory"]) == 0


def test_baseline_yaml_preprocessing_is_valid_for_the_provider(provider):
    from experiments.datasets.msd_spleen import SpleenPreprocessing

    spec = load_experiment_yaml(ROOT / "experiments/configs/swinunetr_baseline.yaml")
    assert spec.dataset.name == NAME
    SpleenPreprocessing.model_validate(spec.preprocessing)


# --- full-volume validation (with spec.inference doing the tiling) ---


def test_val_mode_typo_is_rejected(provider):
    with pytest.raises(DataError, match="val"):
        provider.load(_ds(), {**PRE, "val": "full_volumes"})


def test_full_volume_val_serves_whole_preprocessed_cases(provider):
    splits = provider.load(_ds(), {**PRE, "val": "full_volume"})
    case = splits.val.cases[0]
    x, y = splits.val[0]
    assert type(x) is torch.Tensor and type(y) is torch.Tensor
    assert x.shape == case.image.shape and y.shape == case.label.shape
    assert x.shape[1:] != (8, 8, 8)  # bigger than one roi: not a crop
    assert torch.equal(y, case.label) and y.sum() > 0


def test_full_volume_val_with_sliding_window_runs_end_to_end(provider, tmp_path):
    from mlops.config import load_experiment_dict

    spec = _spleen_spec(tmp_path)
    spec = load_experiment_dict({
        **spec.model_dump(),
        "preprocessing": {**PRE, "val": "full_volume"},
        "inference": {"name": "sliding_window", "params": {"roi_size": [8, 8, 8], "overlap": 0.25}},
        "metrics": [{"name": "dice", "params": {"include_background": False, "class_names": ["bg", "spleen"]}}],
    })
    a = run_experiment(spec, data=provider, tracker=InMemoryTracker())
    b = run_experiment(spec, data=provider, tracker=InMemoryTracker())
    assert a.history == b.history
    assert 0.0 <= a.final_metrics["val/dice/spleen"] <= 1.0


def test_baseline_yaml_validates_on_full_volumes():
    spec = load_experiment_yaml(ROOT / "experiments/configs/swinunetr_baseline.yaml")
    assert spec.preprocessing["val"] == "full_volume"
    assert spec.inference.params["roi_size"] == spec.preprocessing["roi_size"]
