"""`render_report`: a human-readable Markdown report rendered from the run bundle.

The bundle is the source of truth, not the tracker: the report is a pure
function of ``(spec.json, manifest.json, model.pt)``. Same bundle -> same bytes,
so a report can be regenerated and diffed at any time.

API (mlops/application/report.py):

    render_report(bundle) -> str             Markdown
    write_report(store, run_id) -> str       render from the stored bundle, put it as report.md
"""

from __future__ import annotations

import pytest

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.application.bundle import Bundle, load_bundle
from mlops.application.report import REPORT, render_report, write_report
from mlops.config import load_experiment_yaml
from mlops.domain import RunManifest
from mlops.registry import discover

from tests.test_reproduce import CONFIGS


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


@pytest.fixture
def spec():
    return load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")


def _data() -> SyntheticBlobs:
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)


@pytest.fixture
def stored(spec):
    """(store, run_id) holding a FINISHED 3-epoch bundle."""
    spec = spec.model_copy(update={"training": spec.training.model_copy(update={"epochs": 3})})
    store = InMemoryArtifactStore()
    run_id = run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store).run_id
    return store, run_id


@pytest.fixture
def bundle(stored) -> Bundle:
    return load_bundle(*stored)


def _section(md: str, heading: str) -> str:
    """Body of the ``## heading`` section (up to the next ``## ``)."""
    marker = f"\n## {heading}\n"
    assert marker in md, f"missing section {heading!r}"
    return md.split(marker, 1)[1].split("\n## ", 1)[0]


# --- identity & status ---


def test_title_names_experiment_and_status(bundle):
    first = render_report(bundle).splitlines()[0]
    assert first.startswith("# ") and "tiny-cnn-smoke" in first and "FINISHED" in first


def test_full_ids_attempt_and_device_are_present(bundle):
    md = render_report(bundle)
    m = bundle.manifest
    assert m.experiment_id in md and m.run_id in md
    assert "attempt" in md and str(m.attempt) in md
    assert m.device in md


def test_description_and_tags_are_shown(bundle):
    md = render_report(bundle)
    assert bundle.spec.meta.description in md
    assert "smoke" in md


# --- fingerprint: every input that defines the experiment ---


def test_fingerprint_section_lists_every_pin_and_setting(bundle):
    body = _section(render_report(bundle), "Experiment")
    s = bundle.spec
    for expected in (
        s.code.repo, s.code.commit,
        s.environment.image, s.environment.image_digest,
        s.dataset.name, s.dataset.version, s.dataset.snapshot,
        s.model.name, s.loss.name, s.training.optimizer,
    ):
        assert expected in body, expected
    for param in s.model.params:
        assert param in body
    assert "seed" in body and str(s.training.seed) in body
    assert "deterministic" in body


def test_augmentations_are_listed_in_order(spec):
    from mlops.domain.experiment import Augmentation

    spec = spec.model_copy(update={"augmentation": [
        Augmentation(name="rand_flip", params={"prob": 0.5, "spatial_axis": [0]}),
        Augmentation(name="rand_rotate90", params={"prob": 0.1}),
    ]})
    body = _section(render_report(_fake_bundle(spec)), "Experiment")
    assert body.index("rand_flip") < body.index("rand_rotate90")
    assert "prob" in body


def test_resources_show_world_size_and_per_worker_batch(bundle):
    body = _section(render_report(bundle), "Resources")
    r = bundle.spec.resources
    assert "workers" in body and str(r.workers) in body
    assert r.memory in body
    assert str(bundle.spec.per_worker_batch_size()) in body


# --- metrics & curves ---


def test_per_epoch_table_has_a_row_per_epoch(bundle):
    body = _section(render_report(bundle), "Training curves")
    rows = [l for l in body.splitlines() if l.startswith("| ") and l.split("|")[1].strip().isdigit()]
    assert [int(r.split("|")[1]) for r in rows] == [1, 2, 3]
    assert "train/loss" in body and "val/loss" in body


def test_final_metrics_show_last_value_and_best_epoch(spec):
    history = [{"val/loss": 0.9}, {"val/loss": 0.3}, {"val/loss": 0.5}]
    body = _section(render_report(_fake_bundle(spec, history=history)), "Final metrics")
    row = next(l for l in body.splitlines() if "val/loss" in l)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[0] == "val/loss"
    assert "0.5" in cells[1]               # final
    assert "0.3" in cells[2] and "2" in cells[2]   # min, at epoch 2
    assert "0.9" in cells[3] and "1" in cells[3]   # max, at epoch 1


def test_curves_include_a_sparkline(spec):
    history = [{"val/loss": v} for v in (1.0, 0.5, 0.0)]
    body = _section(render_report(_fake_bundle(spec, history=history)), "Training curves")
    assert "█▄▁" in body


def test_flat_metric_sparkline_does_not_divide_by_zero(spec):
    history = [{"val/loss": 0.5}] * 3
    render_report(_fake_bundle(spec, history=history))


# --- model ---


def test_model_section_counts_parameters(bundle):
    body = _section(render_report(bundle), "Model")
    n = sum(t.numel() for t in bundle.model_state.values())
    assert f"{n:,}" in body


# --- environment honesty & reproduction ---


def test_environment_section_says_pins_are_declared_not_verified(bundle):
    body = _section(render_report(bundle), "Environment")
    assert "not verified" in body.lower()
    assert bundle.spec.environment.image_digest in body


def test_reproduce_command_is_given(bundle):
    body = _section(render_report(bundle), "Reproduce")
    assert f"mlops reproduce {bundle.manifest.run_id}" in body


# --- FAILED runs ---


def test_failed_run_report_says_so_and_has_no_weights(spec):
    md = render_report(_fake_bundle(spec, status="FAILED", history=[{"train/loss": 1.0}]))
    assert "FAILED" in md.splitlines()[0]
    assert "no weights" in _section(md, "Model").lower()
    assert "train/loss" in md  # epochs completed before the failure are still reported


def test_run_with_no_completed_epoch_renders(spec):
    md = render_report(_fake_bundle(spec, status="FAILED", history=[]))
    assert "no epoch completed" in _section(md, "Final metrics").lower()


# --- purity & storage ---


def test_report_is_deterministic(stored):
    store, run_id = stored
    assert render_report(load_bundle(store, run_id)) == render_report(load_bundle(store, run_id))


def test_write_report_puts_report_md_into_the_bundle(stored):
    store, run_id = stored
    md = write_report(store, run_id)
    assert REPORT == "report.md"
    assert REPORT in store.list(run_id)
    assert store.get(run_id, REPORT).decode() == md


def test_write_report_does_not_touch_the_evidence(stored):
    store, run_id = stored
    before = {n: store.get(run_id, n) for n in store.list(run_id)}
    write_report(store, run_id)
    assert {n: store.get(run_id, n) for n in before} == before


# --- helpers ---


def _fake_bundle(spec, *, status="FINISHED", history=None) -> Bundle:
    from mlops.domain import compute_experiment_id, compute_run_id

    exp = compute_experiment_id(spec)
    manifest = RunManifest(
        experiment_id=exp,
        run_id=compute_run_id(exp, spec.resources, 1),
        attempt=1,
        status=status,
        device="cpu",
        history=history if history is not None else [{"train/loss": 1.0, "val/loss": 1.1}],
    )
    import torch

    state = {"w": torch.zeros(2, 3)} if status == "FINISHED" else None
    return Bundle(spec=spec, manifest=manifest, model_state=state)


# --- environment record ---


def test_environment_section_shows_checks_and_versions(spec):
    from mlops.domain.environment import record_environment
    from tests.test_environment import FixedProbe

    b = _fake_bundle(spec)
    env = record_environment(spec, FixedProbe(commit_ok=False).capture(spec))
    b = Bundle(spec=b.spec, manifest=b.manifest.model_copy(update={"environment": env}), model_state=b.model_state)
    body = _section(render_report(b), "Environment")
    assert "mismatch" in body and "mlops/x.py" in body
    assert "unknown" in body  # image digest
    assert "torch" in body and "2.x" in body
    assert "3.12.0" in body
    assert "not verified" in body.lower()


def test_header_flags_an_unverified_environment(spec):
    from mlops.domain.environment import record_environment
    from tests.test_environment import FixedProbe

    b = _fake_bundle(spec)
    env = record_environment(spec, FixedProbe(commit_ok=False).capture(spec))
    b = Bundle(spec=b.spec, manifest=b.manifest.model_copy(update={"environment": env}), model_state=b.model_state)
    header = render_report(b).split("\n## ", 1)[0]
    assert "environment" in header and "mismatch" in header
