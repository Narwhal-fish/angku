"""End-to-end: experiment.yaml -> registries -> train loop -> tracker."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import RunnerError, run_experiment
from mlops.config import load_experiment_dict, load_experiment_yaml
from mlops.domain import UnpinnedInputError, compute_experiment_id, compute_run_id
from mlops.domain.experiment import Dataset
from mlops.ports.data import DataError
from mlops.registry import AUGMENTATIONS, Params, discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


@pytest.fixture
def spec():
    return load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")


def _data() -> SyntheticBlobs:
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)


# --- happy path ---


def test_smoke_yaml_trains_one_epoch(spec):
    tracker = InMemoryTracker()
    result = run_experiment(spec, data=_data(), tracker=tracker)

    assert result.experiment_id == compute_experiment_id(spec)
    assert result.run_id == compute_run_id(result.experiment_id, spec.resources, attempt=1)
    assert result.device == "cpu"
    assert len(result.history) == spec.training.epochs == 1
    assert set(result.final_metrics) == {"train/loss", "val/loss"}
    assert all(math.isfinite(v) for v in result.final_metrics.values())

    run = tracker.runs[0]
    assert run.status == "FINISHED"
    assert run.run_id == result.run_id
    assert run.metrics == [(1, result.history[0])]


def test_loss_goes_down_over_epochs(spec):
    spec = spec.model_copy(update={"training": spec.training.model_copy(update={"epochs": 5})})
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    losses = [m["train/loss"] for m in result.history]
    assert losses[-1] < losses[0]


def test_same_seed_same_history(spec):
    a = run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    b = run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    assert a.history == b.history


def test_attempt_changes_run_id_not_experiment_id(spec):
    a = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), attempt=1)
    b = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), attempt=2)
    assert a.experiment_id == b.experiment_id
    assert a.run_id != b.run_id


def test_augmentations_are_built_and_applied(spec):
    calls: list[int] = []

    class _NoParams(Params):
        pass

    if "_test_identity" not in AUGMENTATIONS:

        @AUGMENTATIONS.register("_test_identity", params=_NoParams)
        def _identity(p: _NoParams):
            def aug(x, y):
                calls.append(x.shape[0])
                return x, y

            return aug

    spec = load_experiment_dict({**spec.model_dump(), "augmentation": [{"name": "_test_identity", "params": {}}]})
    run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    assert sum(calls) == 4  # every training sample went through the augmentation once


# --- failure paths ---


def test_unpinned_spec_is_refused_before_anything_runs(spec):
    spec = spec.model_copy(update={"code": spec.code.model_copy(update={"commit": None})})
    tracker = InMemoryTracker()
    with pytest.raises(UnpinnedInputError):
        run_experiment(spec, data=_data(), tracker=tracker)
    assert tracker.runs == []


def test_gpu_requested_without_cuda_is_an_error(spec):
    if torch.cuda.is_available():
        pytest.skip("CUDA present; nothing to refuse")
    spec = spec.model_copy(update={"resources": spec.resources.model_copy(update={"gpu": 1})})
    with pytest.raises(RunnerError, match="CUDA is not available"):
        run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    # explicit override lets the same spec run on the dev box
    assert run_experiment(spec, data=_data(), tracker=InMemoryTracker(), device="cpu").device == "cpu"


def test_unknown_optimizer(spec):
    spec = spec.model_copy(update={"training": spec.training.model_copy(update={"optimizer": "lion"})})
    with pytest.raises(RunnerError, match=r"unknown optimizer 'lion'; known: adam, adamw, sgd"):
        run_experiment(spec, data=_data(), tracker=InMemoryTracker())


def test_data_provider_refuses_other_dataset(spec):
    provider = SyntheticBlobs(name="something-else")
    with pytest.raises(DataError, match="synthetic-blobs"):
        run_experiment(spec, data=provider, tracker=InMemoryTracker())


def test_failure_during_training_marks_run_failed(spec):
    class Boom:
        def snapshot(self, dataset: Dataset) -> None:
            return None

        def load(self, dataset: Dataset, preprocessing: dict[str, Any]):
            raise OSError("disk on fire")

    tracker = InMemoryTracker()
    with pytest.raises(OSError, match="disk on fire"):
        run_experiment(spec, data=Boom(), tracker=tracker)
    # data.load happens before start_run, so nothing was recorded
    assert tracker.runs == []


def test_failure_inside_epoch_marks_run_failed(spec):
    class ExplodingLoader(SyntheticBlobs):
        def load(self, dataset, preprocessing):
            splits = super().load(dataset, preprocessing)
            bad = torch.utils.data.TensorDataset(
                torch.zeros(2, 3, 8, 8, 8),  # wrong channel count -> conv raises
                torch.zeros(2, 8, 8, 8, dtype=torch.long),
            )
            return type(splits)(train=bad, val=splits.val)

    tracker = InMemoryTracker()
    with pytest.raises(RuntimeError):
        run_experiment(spec, data=ExplodingLoader(n_train=2, n_val=0), tracker=tracker)
    assert tracker.runs[0].status == "FAILED"


# --- the MONAI stack end to end: only when the `medical` extra is installed ---


def test_swinunetr_baseline_stack_trains_on_synthetic_data():
    pytest.importorskip("monai", reason="medical extra not installed")
    pytest.importorskip("einops", reason="medical extra not installed")
    from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
    from mlops.application.bundle import load_bundle

    base = load_experiment_yaml(CONFIGS / "swinunetr_baseline.yaml")
    # Same model/loss/augmentations as the baseline; shrunk to run on a CPU in seconds.
    spec = load_experiment_dict(
        {
            **base.model_dump(),
            "dataset": {**base.dataset.model_dump(), "name": "synthetic-blobs"},
            "preprocessing": {},
            "model": {"name": "swinunetr", "params": {**base.model.params, "feature_size": 12}},
            "training": {**base.training.model_dump(), "epochs": 1, "batch_size": 2},
            "inference": {**base.inference.model_dump(), "params": {**base.inference.params, "roi_size": [64, 64, 64]}},
            "resources": {**base.resources.model_dump(), "gpu": 0},
        }
    )
    assert [a.name for a in spec.augmentation] == ["rand_flip", "rand_rotate90"]
    assert spec.loss.name == "dice_ce"
    assert spec.inference.name == "sliding_window" and spec.metrics[0].name == "dice"

    data = SyntheticBlobs(n_train=2, n_val=1, shape=(64, 64, 64), in_channels=1, num_classes=2)
    store = InMemoryArtifactStore()
    result = run_experiment(spec, data=data, tracker=InMemoryTracker(), artifacts=store)

    assert all(math.isfinite(v) for v in result.final_metrics.values())
    bundle = load_bundle(store, result.run_id)
    assert bundle.manifest.status == "FINISHED"
    assert any(k.startswith("swinViT") for k in bundle.model_state)


# --- per-worker training loop (what a distributed launcher runs on every rank) ---


# Set by the tests that register them: a registered component outlives one test, so it reads these.
_COUNT_SEEN: dict[int, list[int]] = {}
_COUNT_RANK: dict[str, int] = {}
_INFER_CALLS: list[tuple[int, ...]] = []


def test_train_worker_shards_data_by_rank(spec):
    from mlops.application.runner import Worker, train_worker

    seen: dict[int, list[int]] = {0: [], 1: []}
    current = {"rank": 0}

    class _NoParams(Params):
        pass

    if "_test_count" not in AUGMENTATIONS:

        @AUGMENTATIONS.register("_test_count", params=_NoParams)
        def _count(p: _NoParams):
            def aug(x, y):
                _COUNT_SEEN[_COUNT_RANK["rank"]].append(x.shape[0])
                return x, y

            return aug

    global _COUNT_SEEN, _COUNT_RANK
    _COUNT_SEEN, _COUNT_RANK = seen, current
    spec = load_experiment_dict(
        {
            **spec.model_dump(),
            "augmentation": [{"name": "_test_count", "params": {}}],
            "training": {**spec.training.model_dump(), "batch_size": 4},
            "resources": {**spec.resources.model_dump(), "workers": 2},
        }
    )
    data = SyntheticBlobs(n_train=8, n_val=0, shape=(8, 8, 8), in_channels=1, num_classes=2)
    for rank in (0, 1):
        current["rank"] = rank
        model, history, _ = train_worker(
            spec, data=data, device=torch.device("cpu"), worker=Worker(rank=rank, world_size=2)
        )
        assert len(history) == 1 and math.isfinite(history[0]["train/loss"])
    # 8 samples, 2 ranks -> 4 each, per-worker batch = 4 // 2 = 2
    assert seen == {0: [2, 2], 1: [2, 2]}


def test_train_worker_single_rank_matches_run_experiment(spec):
    from mlops.application.runner import train_worker

    epochs: list[tuple[int, dict[str, float]]] = []
    model, history, _ = train_worker(
        spec, data=_data(), device=torch.device("cpu"), on_epoch=lambda e, m: epochs.append((e, m))
    )
    assert history == run_experiment(spec, data=_data(), tracker=InMemoryTracker()).history
    assert epochs == [(1, history[0])]
    assert isinstance(model, torch.nn.Module)


def test_train_worker_wrap_model_hook_is_used(spec):
    from mlops.application.runner import train_worker

    wrapped: list[torch.nn.Module] = []

    def wrap(m):
        wrapped.append(m)
        return m

    model, _, _ = train_worker(spec, data=_data(), device=torch.device("cpu"), wrap_model=wrap)
    assert wrapped == [model]


def test_local_launcher_is_run_experiment(spec):
    from mlops.application.launch import LocalLauncher

    tracker = InMemoryTracker()
    result = LocalLauncher().launch(spec, data=_data(), tracker=tracker)
    assert result.run_id == compute_run_id(result.experiment_id, spec.resources, attempt=1)
    assert tracker.runs[0].status == "FINISHED"
    assert result.history == run_experiment(spec, data=_data(), tracker=InMemoryTracker()).history


def test_swinunetr_stack_is_reproducible_with_same_seed():
    pytest.importorskip("monai", reason="medical extra not installed")
    pytest.importorskip("einops", reason="medical extra not installed")
    from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
    from mlops.application.bundle import load_bundle

    base = load_experiment_yaml(CONFIGS / "swinunetr_baseline.yaml")
    spec = load_experiment_dict(
        {
            **base.model_dump(),
            "dataset": {**base.dataset.model_dump(), "name": "synthetic-blobs"},
            "preprocessing": {},
            "model": {"name": "swinunetr", "params": {**base.model.params, "feature_size": 12, "drop_rate": 0.1}},
            "training": {**base.training.model_dump(), "epochs": 2, "batch_size": 2},
            "augmentation": [  # prob 1.0 / 0.5 so the RNG path is actually exercised
                {"name": "rand_flip", "params": {"prob": 0.5, "spatial_axis": [0, 1, 2]}},
                {"name": "rand_rotate90", "params": {"prob": 0.5, "max_k": 3}},
            ],
            "inference": {**base.inference.model_dump(), "params": {**base.inference.params, "roi_size": [64, 64, 64]}},
            "resources": {**base.resources.model_dump(), "gpu": 0},
        }
    )

    def once():
        data = SyntheticBlobs(n_train=2, n_val=1, shape=(64, 64, 64), in_channels=1, num_classes=2)
        store = InMemoryArtifactStore()
        result = run_experiment(spec, data=data, tracker=InMemoryTracker(), artifacts=store)
        return result.history, load_bundle(store, result.run_id).model_state

    h1, w1 = once()
    h2, w2 = once()
    assert h1 == h2
    assert all(torch.equal(w1[k], w2[k]) for k in w1)


def test_seed_everything_switches_torch_deterministic_mode():
    from mlops.application.runner import seed_everything

    seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    seed_everything(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    seed_everything(0)  # default is deterministic: the reproducibility contract
    assert torch.are_deterministic_algorithms_enabled()


# --- environment record ---


def test_probe_result_is_recorded_in_the_manifest(spec):
    from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
    from mlops.application.bundle import load_bundle
    from tests.test_environment import FixedProbe

    store = InMemoryArtifactStore()
    r = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, probe=FixedProbe())
    env = load_bundle(store, r.run_id).manifest.environment
    assert env.actual.packages == {"torch": "2.x"}
    assert env.checks["code.commit"].status == "ok"
    assert env.checks["environment.image_digest"].status == "unknown"
    assert not env.verified
    assert r.environment == env


def test_failed_run_still_records_the_environment(spec):
    from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
    from mlops.application.bundle import load_bundle
    from tests.test_environment import FixedProbe

    bad = spec.model_copy(
        update={"model": spec.model.model_copy(update={"params": {**spec.model.params, "in_channels": 3}})}
    )
    store = InMemoryArtifactStore()
    with pytest.raises(RuntimeError):
        run_experiment(bad, data=_data(), tracker=InMemoryTracker(), artifacts=store, probe=FixedProbe())
    run_id = compute_run_id(compute_experiment_id(bad), bad.resources, attempt=1)
    assert load_bundle(store, run_id).manifest.environment is not None


def test_without_probe_no_environment_is_recorded(spec):
    from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
    from mlops.application.bundle import load_bundle

    store = InMemoryArtifactStore()
    r = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store)
    assert load_bundle(store, r.run_id).manifest.environment is None


def test_manifest_written_before_environment_records_still_loads():
    from mlops.domain import RunManifest

    old = '{"experiment_id": "e", "run_id": "r", "attempt": 1, "status": "FINISHED", "device": "cpu", "history": []}'
    assert RunManifest.model_validate_json(old).environment is None


# --- validation metrics (METRICS registry) ---


def _with_metrics(spec, metrics):
    return load_experiment_dict({**spec.model_dump(), "metrics": metrics})


def test_metrics_are_logged_under_val_prefix(spec):
    spec = _with_metrics(spec, [{"name": "dice", "params": {"class_names": ["bg", "blob"]}}])
    tracker = InMemoryTracker()
    result = run_experiment(spec, data=_data(), tracker=tracker)
    assert set(result.final_metrics) == {
        "train/loss",
        "val/loss",
        "val/dice/bg",
        "val/dice/blob",
        "val/dice/mean",
    }
    assert all(0.0 <= result.final_metrics[k] <= 1.0 for k in result.final_metrics if "dice" in k)
    assert tracker.runs[0].metrics == [(1, result.history[0])]


def test_metrics_do_not_change_training(spec):
    plain = run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    measured = run_experiment(
        _with_metrics(spec, [{"name": "dice", "params": {}}]), data=_data(), tracker=InMemoryTracker()
    )
    assert measured.history[0]["train/loss"] == plain.history[0]["train/loss"]
    assert measured.history[0]["val/loss"] == plain.history[0]["val/loss"]


def test_metric_mean_is_over_samples_where_defined(spec):
    from mlops.registry import METRICS

    class _NoParams(Params):
        pass

    if "_test_nan_half" not in METRICS:

        @METRICS.register("_test_nan_half", params=_NoParams)
        def _nan_half(p: _NoParams):
            # sample i -> value i, except odd positions in the batch are undefined
            def metric(logits, target):
                n = logits.shape[0]
                v = torch.arange(n, dtype=torch.float64)
                v[1::2] = float("nan")
                return {"half": v}

            return metric

    spec = _with_metrics(spec, [{"name": "_test_nan_half", "params": {}}])
    data = SyntheticBlobs(n_train=2, n_val=4, shape=(8, 8, 8), in_channels=1, num_classes=2)
    result = run_experiment(spec, data=data, tracker=InMemoryTracker())
    # batch_size 2 -> two val batches, values [0, nan] and [0, nan] -> mean of the defined ones
    assert result.final_metrics["val/half"] == 0.0


def test_metric_undefined_everywhere_is_nan(spec):
    from mlops.registry import METRICS

    class _NoParams(Params):
        pass

    if "_test_all_nan" not in METRICS:

        @METRICS.register("_test_all_nan", params=_NoParams)
        def _all_nan(p: _NoParams):
            return lambda logits, target: {"never": torch.full((logits.shape[0],), float("nan"))}

    spec = _with_metrics(spec, [{"name": "_test_all_nan", "params": {}}])
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    assert math.isnan(result.final_metrics["val/never"])


def test_metrics_without_val_split_are_refused(spec):
    spec = _with_metrics(spec, [{"name": "dice", "params": {}}])
    data = SyntheticBlobs(n_train=2, n_val=0, shape=(8, 8, 8), in_channels=1, num_classes=2)
    tracker = InMemoryTracker()
    with pytest.raises(RunnerError, match="metrics need a validation split"):
        run_experiment(spec, data=data, tracker=tracker)
    assert tracker.runs == []


def test_metric_key_clash_is_refused(spec):
    spec = _with_metrics(spec, [{"name": "dice", "params": {}}, {"name": "dice", "params": {}}])
    with pytest.raises(RunnerError, match="dice/0"):
        run_experiment(spec, data=_data(), tracker=InMemoryTracker())


def test_unknown_metric_name_is_refused(spec):
    from mlops.registry import RegistryError

    spec = _with_metrics(spec, [{"name": "dcie", "params": {}}])
    with pytest.raises(RegistryError, match="unknown metric 'dcie'"):
        run_experiment(spec, data=_data(), tracker=InMemoryTracker())


# --- inference hook (spec.inference, INFERERS registry) ---


def _register_counting_inferer(calls: list[tuple[int, ...]]) -> None:
    from mlops.registry import INFERERS

    class _NoParams(Params):
        pass

    if "_test_counting" not in INFERERS:

        @INFERERS.register("_test_counting", params=_NoParams)
        def _counting(p: _NoParams):
            def infer(model, x):
                _INFER_CALLS.append(tuple(x.shape))
                return model(x)

            return infer

    global _INFER_CALLS
    _INFER_CALLS = calls
    calls.clear()


def test_inferer_replaces_the_plain_forward_on_validation(spec):
    calls: list[tuple[int, ...]] = []
    _register_counting_inferer(calls)
    plain = run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    calls.clear()
    spec = load_experiment_dict({**spec.model_dump(), "inference": {"name": "_test_counting", "params": {}}})
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    # validation only, one volume at a time (volumes may differ in shape)
    assert calls == [(1, 1, 8, 8, 8), (1, 1, 8, 8, 8)]
    assert result.history[0]["train/loss"] == plain.history[0]["train/loss"]
    assert result.history[0]["val/loss"] == pytest.approx(plain.history[0]["val/loss"], abs=1e-6)


def test_unknown_inferer_is_refused_before_the_run_starts(spec):
    from mlops.registry import RegistryError

    spec = load_experiment_dict({**spec.model_dump(), "inference": {"name": "sliding_windw", "params": {}}})
    tracker = InMemoryTracker()
    with pytest.raises(RegistryError, match="unknown inferer 'sliding_windw'"):
        run_experiment(spec, data=_data(), tracker=tracker)
    assert tracker.runs == []


def test_sliding_window_validation_end_to_end(spec):
    pytest.importorskip("monai", reason="medical extra not installed")
    spec = load_experiment_dict(
        {
            **spec.model_dump(),
            "training": {**spec.training.model_dump(), "epochs": 2},
            "inference": {
                "name": "sliding_window",
                "params": {"roi_size": [4, 4, 4], "overlap": 0.5, "mode": "gaussian"},
            },
            "metrics": [{"name": "dice", "params": {"include_background": False}}],
        }
    )
    a = run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    b = run_experiment(spec, data=_data(), tracker=InMemoryTracker())
    assert a.history == b.history
    assert {"val/loss", "val/dice/1"} <= set(a.final_metrics)


# --- validation sharding: every val case exactly once across ranks (no padding) ---


@pytest.mark.parametrize("n,world_size", [(5, 2), (2, 2), (1, 3), (9, 4), (0, 2)])
def test_val_shards_cover_every_case_exactly_once(n, world_size):
    from mlops.application.runner import ShardSampler

    shards = [list(ShardSampler(n, rank=r, world_size=world_size)) for r in range(world_size)]
    assert sorted(i for s in shards for i in s) == list(range(n))
    assert [len(ShardSampler(n, rank=r, world_size=world_size)) for r in range(world_size)] == [len(s) for s in shards]
