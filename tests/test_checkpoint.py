"""Mid-run checkpoint + resume.

The promise: a run interrupted after epoch k and resumed from its checkpoint ends with the
same run_id, history and weights (bit-exact) as the same run left alone, and its manifest
says where it was resumed. ``angku reproduce`` (a replay from scratch) is the honesty check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import RunnerError, run_experiment
from mlops.application.bundle import CHECKPOINT, MANIFEST, load_bundle, load_checkpoint
from mlops.application.reproduce import reproduce_run
from mlops.config import load_experiment_dict, load_experiment_yaml
from mlops.registry import discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


@pytest.fixture
def spec():
    """4 epochs with a random augmentation, so the resumed RNG state actually matters."""
    base = load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")
    return load_experiment_dict(
        {
            **base.model_dump(),
            "training": {**base.training.model_dump(), "epochs": 4},
            "augmentation": [{"name": "rand_flip", "params": {"prob": 0.5, "spatial_axis": [0, 1, 2]}}],
        }
    )


def _data() -> SyntheticBlobs:
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)


class _Interrupted(Exception):
    pass


class FlakyTracker(InMemoryTracker):
    """Blows up when epoch ``fail_at`` is logged, once: the run dies mid-training."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at

    def log_metrics(self, metrics, *, step):
        if step == self.fail_at:
            self.fail_at = -1
            raise _Interrupted(f"killed at epoch {step}")
        super().log_metrics(metrics, step=step)


class KilledStore(InMemoryArtifactStore):
    """A store whose process gets SIGKILLed: the FAILED manifest never lands. ``revive()``
    is the machine coming back: writes work again."""

    dead = True

    def put(self, run_id, name, data):
        if not (self.dead and name == MANIFEST):
            super().put(run_id, name, data)

    def revive(self):
        self.dead = False


def _interrupt(spec, store, *, fail_at=3, every=2, tracker=None, **kw):
    with pytest.raises(_Interrupted):
        run_experiment(
            spec, data=_data(), tracker=tracker or FlakyTracker(fail_at), artifacts=store, checkpoint_every=every, **kw
        )
    if isinstance(store, KilledStore):
        assert MANIFEST not in store.list(next(iter(store._runs)))
        store.revive()


# --- writing checkpoints ---


def test_checkpoint_written_every_n_epochs(spec):
    store = InMemoryArtifactStore()
    spec = spec.model_copy(update={"training": spec.training.model_copy(update={"epochs": 3})})
    result = run_experiment(
        spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, checkpoint_every=2, keep_checkpoint=True
    )
    ckpt = load_checkpoint(store, result.run_id)
    assert ckpt.epoch == 2
    assert ckpt.history == result.history[:2]
    assert ckpt.run_id == result.run_id and ckpt.attempt == 1
    assert ckpt.checkpoint_every == 2


def test_no_checkpoint_unless_asked(spec):
    store = InMemoryArtifactStore()
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store)
    assert CHECKPOINT not in store.list(result.run_id)


def test_checkpointing_does_not_change_the_run(spec):
    plain_store, ckpt_store = InMemoryArtifactStore(), InMemoryArtifactStore()
    plain = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=plain_store)
    ckpt = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=ckpt_store, checkpoint_every=1)
    assert plain.history == ckpt.history
    a, b = load_bundle(plain_store, plain.run_id), load_bundle(ckpt_store, ckpt.run_id)
    assert all(torch.equal(v, b.model_state[k]) for k, v in a.model_state.items())


def test_checkpoint_every_needs_an_artifact_store(spec):
    with pytest.raises(RunnerError, match="checkpoint_every needs an artifact store"):
        run_experiment(spec, data=_data(), tracker=InMemoryTracker(), checkpoint_every=1)


def test_checkpoint_every_must_be_positive(spec):
    with pytest.raises(RunnerError, match="checkpoint_every"):
        run_experiment(
            spec, data=_data(), tracker=InMemoryTracker(), artifacts=InMemoryArtifactStore(), checkpoint_every=0
        )


def test_spec_is_stored_with_the_first_checkpoint(spec):
    # A killed run never reaches save_bundle; resume (and a stranger) still needs the spec.
    store = KilledStore()
    _interrupt(spec, store)
    run_id = next(iter(store._runs))
    assert set(store.list(run_id)) == {"spec.json", CHECKPOINT}


# --- resuming ---


@pytest.mark.parametrize("store_cls", [InMemoryArtifactStore, KilledStore], ids=["failed", "killed"])
def test_resumed_run_equals_uninterrupted_run(spec, store_cls):
    ref_store = InMemoryArtifactStore()
    ref = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=ref_store)

    store = store_cls()
    _interrupt(spec, store)
    resumed = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, resume=True)

    assert resumed.run_id == ref.run_id
    assert resumed.history == ref.history
    a, b = load_bundle(ref_store, ref.run_id), load_bundle(store, resumed.run_id)
    assert b.manifest.status == "FINISHED"
    assert b.manifest.history == ref.history
    assert all(torch.equal(v, b.model_state[k]) for k, v in a.model_state.items())
    assert [r.epoch for r in b.manifest.resumes] == [2]
    assert a.manifest.resumes == []


def test_resume_twice_records_both_points(spec):
    store = InMemoryArtifactStore()
    _interrupt(spec, store, fail_at=2, every=1)
    _interrupt(spec, store, fail_at=4, every=1, resume=True)
    run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, resume=True)
    run_id = next(iter(store._runs))
    assert [r.epoch for r in load_bundle(store, run_id).manifest.resumes] == [1, 3]


def test_resume_keeps_checkpointing_at_the_stored_interval(spec):
    store = InMemoryArtifactStore()
    _interrupt(spec, store, fail_at=3, every=2)
    result = run_experiment(
        spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, resume=True, keep_checkpoint=True
    )
    assert load_checkpoint(store, result.run_id).epoch == 4


def test_resume_continues_the_same_tracker_run(spec):
    tracker = FlakyTracker(fail_at=3)
    store = InMemoryArtifactStore()
    _interrupt(spec, store, tracker=tracker)
    run_experiment(spec, data=_data(), tracker=tracker, artifacts=store, resume=True)
    assert len(tracker.runs) == 1
    run = tracker.runs[0]
    assert run.status == "FINISHED"
    assert [step for step, _ in run.metrics] == [1, 2, 3, 4]


def test_resumed_bundle_reproduces_from_scratch(spec):
    store = InMemoryArtifactStore()
    _interrupt(spec, store)
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, resume=True)
    report = reproduce_run(
        store, result.run_id, data=_data(), tracker=InMemoryTracker(), scratch=InMemoryArtifactStore()
    )
    assert report.ok, report.mismatches


def test_environment_is_recorded_per_segment(spec):
    from tests.test_environment import FixedProbe

    store = InMemoryArtifactStore()
    _interrupt(spec, store, probe=FixedProbe(commit_ok=True))
    run_experiment(
        spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, resume=True, probe=FixedProbe(commit_ok=False)
    )
    m = load_bundle(store, next(iter(store._runs))).manifest
    # The first segment's record survives the interruption; the resumed one is recorded separately.
    assert m.environment.checks["code.commit"].status == "ok"
    assert m.resumes[0].environment.checks["code.commit"].status == "mismatch"


# --- refusals ---


def test_resume_needs_an_artifact_store(spec):
    with pytest.raises(RunnerError, match="resume needs an artifact store"):
        run_experiment(spec, data=_data(), tracker=InMemoryTracker(), resume=True)


def test_resume_without_checkpoint_is_refused(spec):
    store = InMemoryArtifactStore()
    with pytest.raises(RunnerError, match="no checkpoint"):
        run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, resume=True)


@pytest.mark.parametrize("keep", [False, True])
def test_resume_of_a_finished_run_is_refused(spec, keep):
    # Whether or not the checkpoint survived, the reason to give is "already FINISHED".
    store = InMemoryArtifactStore()
    run_experiment(
        spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, checkpoint_every=1, keep_checkpoint=keep
    )
    tracker = InMemoryTracker()
    with pytest.raises(RunnerError, match="already FINISHED"):
        run_experiment(spec, data=_data(), tracker=tracker, artifacts=store, resume=True)
    assert tracker.runs == []


def test_checkpoint_of_another_run_is_refused(spec):
    store = InMemoryArtifactStore()
    _interrupt(spec, store)
    run_id = next(iter(store._runs))
    other = spec.model_copy(update={"training": spec.training.model_copy(update={"seed": 1})})
    from mlops.domain import compute_experiment_id, compute_run_id

    other_id = compute_run_id(compute_experiment_id(other), other.resources, 1)
    store.put(other_id, CHECKPOINT, store.get(run_id, CHECKPOINT))
    with pytest.raises(RunnerError, match="belongs to run"):
        run_experiment(other, data=_data(), tracker=InMemoryTracker(), artifacts=store, resume=True)


# --- retention: the checkpoint is for resuming; a FINISHED run has nothing left to resume ---


def test_finished_run_drops_its_checkpoint(spec):
    store = InMemoryArtifactStore()
    result = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, checkpoint_every=1)
    assert CHECKPOINT not in store.list(result.run_id)
    assert load_bundle(store, result.run_id).manifest.status == "FINISHED"


def test_keep_checkpoint_keeps_it(spec):
    store = InMemoryArtifactStore()
    result = run_experiment(
        spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, checkpoint_every=1, keep_checkpoint=True
    )
    assert load_checkpoint(store, result.run_id).epoch == spec.training.epochs


def test_resumed_run_drops_its_checkpoint_once_finished(spec):
    store = KilledStore()
    _interrupt(spec, store)
    run_id = next(iter(store._runs))
    run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, resume=True)
    assert CHECKPOINT not in store.list(run_id)


def test_failed_run_keeps_its_checkpoint(spec):
    store = InMemoryArtifactStore()
    _interrupt(spec, store)
    run_id = next(iter(store._runs))
    assert CHECKPOINT in store.list(run_id)
