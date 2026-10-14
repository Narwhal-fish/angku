"""The 8 properties the ExperimentID / RunID design must satisfy."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from mlops.config import ConfigError, load_experiment_dict, load_experiment_yaml
from mlops.domain import UnpinnedInputError, compute_experiment_id, compute_run_id

EXAMPLE_YAML = Path(__file__).parent.parent / "experiments/configs/swinunetr_baseline.yaml"
HEX64 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(EXAMPLE_YAML.read_text())


def _shuffle_keys(d):
    """Recursively reverse key order to prove order-independence."""
    if isinstance(d, dict):
        return {k: _shuffle_keys(d[k]) for k in reversed(list(d))}
    if isinstance(d, list):
        return [_shuffle_keys(x) for x in d]
    return d


# 1) example yaml parses and validates
def test_1_example_yaml_loads():
    spec = load_experiment_yaml(EXAMPLE_YAML)
    assert spec.meta.name == "swinunetr-baseline"
    assert spec.model.name == "swinunetr"
    assert spec.training.learning_rate == 1e-4
    assert spec.is_pinned()


# 2) both ids can be computed and look like sha256 hex
def test_2_ids_are_computable(raw):
    spec = load_experiment_dict(raw)
    eid = compute_experiment_id(spec)
    rid = compute_run_id(eid, spec.resources)
    assert len(eid) == 64 and int(eid, 16)
    assert len(rid) == 64 and int(rid, 16)
    assert eid != rid


# 3) deterministic and independent of key order
def test_3_experiment_id_deterministic_and_order_independent(raw):
    a = load_experiment_dict(raw)
    b = load_experiment_dict(copy.deepcopy(raw))
    c = load_experiment_dict(_shuffle_keys(raw))
    assert compute_experiment_id(a) == compute_experiment_id(b) == compute_experiment_id(c)


# 4) changing learning_rate changes ExperimentID
def test_4_learning_rate_changes_experiment_id(raw):
    base = compute_experiment_id(load_experiment_dict(raw))
    changed = copy.deepcopy(raw)
    changed["training"]["learning_rate"] = 3e-4
    assert compute_experiment_id(load_experiment_dict(changed)) != base


# 5) changing meta only does NOT change ExperimentID
def test_5_meta_does_not_affect_experiment_id(raw):
    base = compute_experiment_id(load_experiment_dict(raw))
    changed = copy.deepcopy(raw)
    changed["meta"]["name"] = "renamed"
    changed["meta"]["tags"] = ["something", "else"]
    changed["meta"]["description"] = "totally different text"
    assert compute_experiment_id(load_experiment_dict(changed)) == base


# 5b) other non-scientific fields (image tag, dataset version label, tracking) also excluded
def test_5b_labels_and_tracking_do_not_affect_experiment_id(raw):
    base = compute_experiment_id(load_experiment_dict(raw))
    changed = copy.deepcopy(raw)
    changed["environment"]["image"] = "ghcr.io/example/mlops-train:9.9.9"
    changed["dataset"]["version"] = "v99"
    changed["tracking"]["uri"] = "http://elsewhere:5000"
    assert compute_experiment_id(load_experiment_dict(changed)) == base


# 6) changing GPU count keeps ExperimentID, changes RunID
def test_6_resources_change_run_id_only(raw):
    spec1 = load_experiment_dict(raw)
    changed = copy.deepcopy(raw)
    changed["resources"]["gpu"] = 4
    spec4 = load_experiment_dict(changed)

    eid1, eid4 = compute_experiment_id(spec1), compute_experiment_id(spec4)
    assert eid1 == eid4
    assert compute_run_id(eid1, spec1.resources) != compute_run_id(eid4, spec4.resources)
    # same hardware, different attempt -> different RunID too
    assert compute_run_id(eid1, spec1.resources, attempt=1) != compute_run_id(
        eid1, spec1.resources, attempt=2
    )


# 7) unpinned inputs -> refuse to compute ExperimentID
@pytest.mark.parametrize(
    "section,field",
    [("code", "commit"), ("environment", "image_digest"), ("dataset", "snapshot")],
)
def test_7_unpinned_input_is_rejected(raw, section, field):
    changed = copy.deepcopy(raw)
    changed[section][field] = None
    spec = load_experiment_dict(changed)  # loading a draft is fine...
    assert not spec.is_pinned()
    with pytest.raises(UnpinnedInputError) as exc:  # ...computing an id is not
        compute_experiment_id(spec)
    assert f"{section}.{field}" in str(exc.value)


# 7b) "pinned" to a moving target is rejected already at load time
@pytest.mark.parametrize(
    "section,field,value",
    [
        ("code", "commit", "main"),
        ("code", "commit", "abc123"),  # short sha
        ("environment", "image_digest", "latest"),
        ("environment", "image_digest", HEX64),  # missing sha256: prefix
        ("dataset", "snapshot", "latest"),
        ("dataset", "snapshot", ""),
    ],
)
def test_7b_moving_targets_rejected_at_load(raw, section, field, value):
    changed = copy.deepcopy(raw)
    changed[section][field] = value
    with pytest.raises(ConfigError):
        load_experiment_dict(changed)


# 8) typo in a field name is rejected at load time
def test_8_typo_field_rejected(raw):
    changed = copy.deepcopy(raw)
    changed["training"]["learing_rate"] = changed["training"].pop("learning_rate")
    with pytest.raises(ConfigError) as exc:
        load_experiment_dict(changed)
    msg = str(exc.value)
    assert "learing_rate" in msg and "learning_rate" in msg


def test_8b_typo_in_top_level_section_rejected(raw):
    changed = copy.deepcopy(raw)
    changed["trainning"] = changed.pop("training")
    with pytest.raises(ConfigError):
        load_experiment_dict(changed)


def test_spec_is_immutable(raw):
    spec = load_experiment_dict(raw)
    with pytest.raises(Exception):
        spec.training.learning_rate = 1.0  # type: ignore[misc]


# --- resources.workers: the world size, a RunID matter, never an ExperimentID matter ---


def test_workers_defaults_to_one_and_changes_run_id_only(raw):
    spec1 = load_experiment_dict(raw)
    assert spec1.resources.workers == 1
    two = copy.deepcopy(raw)
    two["resources"]["workers"] = 2
    spec2 = load_experiment_dict(two)
    assert compute_experiment_id(spec1) == compute_experiment_id(spec2)
    assert compute_run_id(compute_experiment_id(spec1), spec1.resources) != compute_run_id(
        compute_experiment_id(spec2), spec2.resources
    )


def test_global_batch_must_split_evenly_across_workers(raw):
    raw["training"]["batch_size"] = 4
    raw["resources"]["workers"] = 3
    with pytest.raises(ConfigError, match="batch_size=4.*workers=3"):
        load_experiment_dict(raw)


def test_per_worker_batch(raw):
    raw["training"]["batch_size"] = 8
    raw["resources"]["workers"] = 4
    assert load_experiment_dict(raw).per_worker_batch_size() == 2


def test_workers_must_be_positive(raw):
    raw["resources"]["workers"] = 0
    with pytest.raises(ConfigError):
        load_experiment_dict(raw)


def test_training_deterministic_defaults_on_and_is_part_of_experiment_id(raw):
    spec = load_experiment_dict(raw)
    assert spec.training.deterministic is True
    off = copy.deepcopy(raw)
    off["training"]["deterministic"] = False
    assert compute_experiment_id(spec) != compute_experiment_id(load_experiment_dict(off))


def test_metrics_default_empty_and_old_ids_unchanged(raw):
    raw.pop("metrics", None)
    spec = load_experiment_dict(raw)
    assert spec.metrics == []
    # Absent metrics stay out of the fingerprint, so ids from before the field existed hold.
    assert "metrics" not in spec.fingerprint()


def test_metrics_are_part_of_experiment_id(raw):
    raw["metrics"] = [{"name": "dice", "params": {"include_background": False}}]
    a = load_experiment_dict(raw)
    raw["metrics"][0]["params"]["include_background"] = True
    b = load_experiment_dict(raw)
    assert compute_experiment_id(a) != compute_experiment_id(b)


def test_metrics_entry_typo_rejected(raw):
    raw["metrics"] = [{"name": "dice", "parms": {}}]
    with pytest.raises(ConfigError):
        load_experiment_dict(raw)
