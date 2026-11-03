"""Data split as a spec option: `split` in the yaml, `assign_split` (pure) decides which case
goes where, providers build the datasets for the assignment, the bundle records it."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from angku.config import ConfigError, load_experiment_dict
from angku.domain import compute_experiment_id
from angku.domain.split import SplitError, assign_split, count_of

SMOKE_YAML = Path(__file__).parent.parent / "experiments/configs/tiny_cnn_smoke.yaml"
IDS = [f"case-{i:03d}" for i in range(41)]


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(SMOKE_YAML.read_text())


def _split(**kw):
    raw = yaml.safe_load(SMOKE_YAML.read_text())
    raw["split"] = kw
    return load_experiment_dict(raw).split


# -- spec ----------------------------------------------------------------------


def test_defaults_are_resolved_into_the_spec():
    s = _split(kind="holdout")
    assert s.model_dump() == {
        "kind": "holdout",
        "val": 0.2,
        "test": 0.0,
        "k": None,
        "fold": None,
        "order": "shuffled",
        "seed": 0,
    }
    s = _split(kind="kfold", fold=2)
    assert (s.k, s.fold, s.val, s.test, s.order) == (5, 2, None, 0.0, "shuffled")


def test_kind_defaults_to_holdout():
    assert _split().kind == "holdout"


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "holdout", "vall": 0.2},  # typo
        {"kind": "holdout", "k": 5},  # kfold-only field
        {"kind": "holdout", "fold": 1},
        {"kind": "kfold", "val": 0.2, "fold": 0},  # holdout-only field
        {"kind": "kfold"},  # fold is required
        {"kind": "kfold", "k": 5, "fold": 5},  # fold out of range
        {"kind": "kfold", "k": 1, "fold": 0},
        {"kind": "holdout", "val": 0.0},
        {"kind": "holdout", "val": 1.0},
        {"kind": "holdout", "test": -0.1},
        {"kind": "holdout", "val": 0.5, "test": 0.5},
        {"kind": "random"},
        {"order": "reversed"},
    ],
)
def test_invalid_split_rejected(bad):
    with pytest.raises(ConfigError):
        _split(**bad)


def test_absent_split_stays_out_of_the_fingerprint(raw):
    before = compute_experiment_id(load_experiment_dict(raw))
    assert "split" not in load_experiment_dict(raw).fingerprint()
    raw["split"] = {"kind": "holdout"}
    assert compute_experiment_id(load_experiment_dict(raw)) != before


@pytest.mark.parametrize("change", [{"val": 0.3}, {"test": 0.1}, {"order": "sorted"}, {"seed": 1}])
def test_every_split_field_is_part_of_experiment_id(raw, change):
    raw["split"] = {"kind": "holdout"}
    base = compute_experiment_id(load_experiment_dict(raw))
    raw["split"] = {"kind": "holdout", **change}
    assert compute_experiment_id(load_experiment_dict(raw)) != base


def test_each_fold_is_its_own_experiment(raw):
    ids = set()
    for fold in range(5):
        raw["split"] = {"kind": "kfold", "fold": fold}
        ids.add(compute_experiment_id(load_experiment_dict(raw)))
    assert len(ids) == 5


def test_split_seed_is_separate_from_training_seed(raw):
    raw["split"] = {"kind": "holdout", "test": 0.15}
    a = assign_split(IDS, load_experiment_dict(raw).split)
    raw["training"]["seed"] = 123
    assert assign_split(IDS, load_experiment_dict(raw).split) == a


# -- assign_split ----------------------------------------------------------------


def test_count_rounds_half_up_and_is_at_least_one_unless_zero():
    assert count_of(0.0, 41) == 0
    assert count_of(0.15, 41) == 6  # 6.15
    assert count_of(0.2, 41) == 8  # 8.2
    assert count_of(0.25, 10) == 3  # 2.5 rounds up (Python's round() would give 2)
    assert count_of(0.01, 10) == 1


def test_holdout_counts_and_partition():
    a = assign_split(IDS, _split(kind="holdout", val=0.2, test=0.15))
    assert (len(a.train), len(a.val), len(a.test)) == (27, 8, 6)
    assert sorted(a.train + a.val + a.test) == IDS
    assert not set(a.train) & set(a.val) and not set(a.val) & set(a.test)
    assert a.counts() == {"train": 27, "val": 8, "test": 6}


def test_no_test_by_default():
    a = assign_split(IDS, _split(kind="holdout"))
    assert a.test == [] and len(a.val) == 8


def test_each_part_is_sorted_by_case_id():
    a = assign_split(IDS, _split(kind="holdout", test=0.15))
    for part in (a.train, a.val, a.test):
        assert part == sorted(part)


def test_sorted_order_takes_test_then_val_from_the_front():
    a = assign_split([f"c{i}" for i in range(10)], _split(order="sorted", val=0.2, test=0.1))
    assert (a.test, a.val, a.train) == (["c0"], ["c1", "c2"], [f"c{i}" for i in range(3, 10)])


def test_shuffled_is_deterministic_and_depends_on_seed_only():
    s = _split(kind="holdout", test=0.15)
    a = assign_split(IDS, s)
    assert assign_split(list(reversed(IDS)), s) == a  # input order does not matter
    assert assign_split(IDS, _split(kind="holdout", test=0.15, seed=1)) != a
    assert a != assign_split(IDS, _split(kind="holdout", test=0.15, order="sorted"))


def test_shuffled_assignment_is_pinned():
    # A shuffle that changed with the Python version would silently move cases between splits.
    a = assign_split([f"c{i}" for i in range(10)], _split(val=0.2, test=0.2))
    assert (a.test, a.val) == (["c6", "c9"], ["c2", "c4"])


def test_kfold_vals_cover_everything_but_test_once():
    s0 = _split(kind="kfold", k=5, fold=0, test=0.15)
    folds = [assign_split(IDS, s0.model_copy(update={"fold": f})) for f in range(5)]
    test = folds[0].test
    assert len(test) == 6 and all(f.test == test for f in folds)
    vals = [c for f in folds for c in f.val]
    assert sorted(vals) == sorted(set(IDS) - set(test))  # each non-test case is val exactly once
    assert [len(f.val) for f in folds] == [7, 7, 7, 7, 7]  # 35 / 5
    for f in folds:
        assert sorted(f.train + f.val) == sorted(set(IDS) - set(test))


def test_kfold_uneven_sizes_differ_by_at_most_one():
    s = _split(kind="kfold", k=4, fold=0)
    sizes = [len(assign_split(IDS, s.model_copy(update={"fold": f})).val) for f in range(4)]
    assert sorted(sizes) == [10, 10, 10, 11] and sizes[0] == 11  # the first folds take the remainder


@pytest.mark.parametrize(
    "ids, split, words",
    [
        ([], {"kind": "holdout"}, "no cases"),
        (["a", "a", "b"], {"kind": "holdout"}, "duplicate"),
        (["a", "b"], {"kind": "holdout", "val": 0.5, "test": 0.4}, "no training case"),
        (["a", "b", "c"], {"kind": "kfold", "k": 5, "fold": 0}, "fewer cases"),
    ],
)
def test_impossible_splits_raise(ids, split, words):
    with pytest.raises(SplitError, match=words):
        assign_split(ids, _split(**split))


# -- providers, runner, bundle -----------------------------------------------------

import torch  # noqa: E402

from angku.adapters.inmemory_artifacts import InMemoryArtifactStore  # noqa: E402
from angku.adapters.inmemory_tracker import InMemoryTracker  # noqa: E402
from angku.adapters.synthetic_data import SyntheticBlobs  # noqa: E402
from angku.application import run_experiment  # noqa: E402
from angku.application.bundle import SPLIT, load_bundle  # noqa: E402
from angku.application.reproduce import reproduce_run  # noqa: E402
from angku.domain.experiment import Dataset  # noqa: E402
from angku.domain.split import SplitAssignment  # noqa: E402
from angku.ports.data import DataError, DataSplits  # noqa: E402
from angku.registry import discover  # noqa: E402

DS = Dataset(name="synthetic-blobs", version="v1", snapshot="sha256:" + "0" * 64)


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


def _spec(raw: dict, **split):
    raw["split"] = split
    return load_experiment_dict(raw)


def test_synthetic_cases_and_legacy_load_unchanged():
    p = SyntheticBlobs(n_train=6, n_val=2)
    assert p.cases(DS) == [f"case-{i:03d}" for i in range(8)]
    legacy = p.load(DS, {})
    by_split = p.load(DS, {}, split=SplitAssignment(train=p.cases(DS)[:6], val=p.cases(DS)[6:]))
    # The legacy split is "first n_train train, rest val" over the same generated samples.
    for a, b in ((legacy.train, by_split.train), (legacy.val, by_split.val)):
        assert all(torch.equal(x, y) for x, y in zip(a.tensors, b.tensors, strict=True))
    assert legacy.test is None and by_split.test is None


def test_synthetic_serves_exactly_the_assigned_cases():
    p = SyntheticBlobs(n_train=6, n_val=2)
    pool = p.load(DS, {}, split=SplitAssignment(train=p.cases(DS), val=[]))
    s = p.load(DS, {}, split=SplitAssignment(train=["case-001", "case-004"], val=["case-000"], test=["case-007"]))
    assert len(s.train) == 2 and len(s.val) == 1 and len(s.test) == 1
    assert torch.equal(s.test.tensors[0][0], pool.train.tensors[0][7])
    with pytest.raises(DataError, match="case-099"):
        p.load(DS, {}, split=SplitAssignment(train=["case-099"], val=[]))


class _TestIsOffLimits(torch.utils.data.Dataset):
    def __len__(self):
        raise AssertionError("training touched the test split")

    def __getitem__(self, i):
        raise AssertionError("training touched the test split")


class _Guarded(SyntheticBlobs):
    def load(self, dataset, preprocessing, split=None):
        s = super().load(dataset, preprocessing, split=split)
        return DataSplits(train=s.train, val=s.val, test=_TestIsOffLimits() if s.test is not None else None)


def test_run_records_the_split_and_never_reads_test(raw):
    spec = _spec(raw, kind="holdout", val=0.25, test=0.25)
    store = InMemoryArtifactStore()
    result = run_experiment(spec, data=_Guarded(n_train=6, n_val=2), tracker=InMemoryTracker(), artifacts=store)
    bundle = load_bundle(store, result.run_id)
    assert bundle.manifest.split == {"train": 4, "val": 2, "test": 2} == result.split.counts()
    assert bundle.split == result.split
    assert sorted(bundle.split.train + bundle.split.val + bundle.split.test) == [f"case-{i:03d}" for i in range(8)]


def test_run_without_split_writes_no_split_file(raw):
    store = InMemoryArtifactStore()
    result = run_experiment(
        load_experiment_dict(raw), data=SyntheticBlobs(), tracker=InMemoryTracker(), artifacts=store
    )
    assert SPLIT not in store.list(result.run_id)
    assert load_bundle(store, result.run_id).manifest.split is None


def test_split_changes_training_and_reproduces(raw):
    store = InMemoryArtifactStore()
    a = run_experiment(
        _spec(raw, kind="holdout", val=0.25),
        data=SyntheticBlobs(n_train=6, n_val=2),
        tracker=InMemoryTracker(),
        artifacts=store,
    )
    b = run_experiment(
        _spec(raw, kind="holdout", val=0.25, seed=7),
        data=SyntheticBlobs(n_train=6, n_val=2),
        tracker=InMemoryTracker(),
        artifacts=store,
    )
    assert a.split != b.split and a.history != b.history
    report = reproduce_run(
        store,
        a.run_id,
        data=SyntheticBlobs(n_train=6, n_val=2),
        tracker=InMemoryTracker(),
        scratch=InMemoryArtifactStore(),
    )
    assert report.ok, report.mismatches


class _NoCases:
    def snapshot(self, dataset):
        return None

    def load(self, dataset, preprocessing):
        return SyntheticBlobs().load(dataset, preprocessing)


@pytest.mark.parametrize(
    "data, words",
    [
        (_NoCases(), "cannot list its cases"),
        (SyntheticBlobs(n_train=1, n_val=1), "no training case"),
    ],
)
def test_split_problems_stop_the_run_before_it_starts(raw, data, words):
    tracker, store = InMemoryTracker(), InMemoryArtifactStore()
    with pytest.raises(DataError, match=words):
        run_experiment(
            _spec(raw, kind="holdout", val=0.5, test=0.5 - 1e-9), data=data, tracker=tracker, artifacts=store
        )
    assert not tracker.runs and not store.runs()


# -- report, diff, CLI ----------------------------------------------------------------

from angku.application.diff import diff_lines, diff_runs  # noqa: E402
from angku.application.report import render_report, split_counts, split_rule  # noqa: E402
from angku.cli import main  # noqa: E402


def test_split_rule_and_counts_read_plainly():
    assert split_rule(None) == "the data provider's own split"
    assert split_rule(_split(val=0.2, test=0.15)) == "holdout: val 0.2, test 0.15, shuffled (seed 0)"
    assert split_rule(_split(kind="kfold", fold=1, order="sorted")) == "5-fold, fold 1, sorted by case id"
    assert split_counts({"train": 27, "val": 8, "test": 6}) == "train 27, val 8, test 6"
    assert split_counts({"train": 6, "val": 2, "test": 0}) == "train 6, val 2"


def test_report_shows_the_rule_and_the_counts(raw):
    store = InMemoryArtifactStore()
    r = run_experiment(
        _spec(raw, val=0.25, test=0.25),
        data=SyntheticBlobs(n_train=6, n_val=2),
        tracker=InMemoryTracker(),
        artifacts=store,
    )
    md = render_report(load_bundle(store, r.run_id))
    assert "| split | train 4, val 2, test 2, cases in `split.json` |" in md
    assert "| split | holdout: val 0.25, test 0.25, shuffled (seed 0) |" in md


def test_diff_names_the_cases_that_moved(raw):
    store = InMemoryArtifactStore()
    data = SyntheticBlobs(n_train=6, n_val=2)
    a = run_experiment(_spec(raw, val=0.25), data=data, tracker=InMemoryTracker(), artifacts=store)
    b = run_experiment(_spec(raw, val=0.25, seed=7), data=data, tracker=InMemoryTracker(), artifacts=store)
    d = diff_runs(store, a.run_id, store, b.run_id)
    moved = {c.part: c for c in d.split}
    assert set(moved) == {"train", "val"}
    assert set(moved["val"].only_a) == set(a.split.val) - set(b.split.val)
    lines = diff_lines(d)
    assert "split (cases in a different part):" in lines
    assert any("split.seed" in line for line in lines)  # the rule itself is a fingerprint change
    assert d.to_dict()["split"][0]["part"] in {"train", "val"}


def test_diff_without_split_files_has_no_split_section(raw):
    store = InMemoryArtifactStore()
    a = run_experiment(load_experiment_dict(raw), data=SyntheticBlobs(), tracker=InMemoryTracker(), artifacts=store)
    assert diff_runs(store, a.run_id, store, a.run_id).split == []


def test_cli_run_prints_the_split(raw, tmp_path, capsys):
    raw["split"] = {"kind": "holdout", "val": 0.25, "test": 0.25}
    raw["tracking"]["backend"] = "mlflow"
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert main(["run", str(path), "--tracker", "inmemory"]) == 0
    out = capsys.readouterr().out
    assert "  split          train 6, val 3, test 3  (holdout: val 0.25, test 0.25, shuffled (seed 0))" in out


def test_ui_overview_shows_the_split_like_the_report(raw):
    from angku.application.listing import list_runs
    from angku.ui.rows import overview_rows

    store = InMemoryArtifactStore()
    r = run_experiment(
        _spec(raw, val=0.25, test=0.25),
        data=SyntheticBlobs(n_train=6, n_val=2),
        tracker=InMemoryTracker(),
        artifacts=store,
    )
    b = load_bundle(store, r.run_id)
    rows = dict(overview_rows(list_runs(store)[0], spec=b.spec, manifest=b.manifest))
    assert rows["split"] == "train 4, val 2, test 2; holdout: val 0.25, test 0.25, shuffled (seed 0)"
