"""``mlops gc`` (roadmap 18): clean an artifact root without touching anything that can still be
reproduced, resumed, or that a prediction result depends on."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.adapters.ray_storage import LocalRayStorage
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.application.bundle import (
    CHECKPOINT,
    MANIFEST,
    MODEL,
    SPEC,
    LaunchRecord,
    ray_run_name,
    save_launch,
)
from mlops.application.gc import GcError, GcPolicy, apply_gc, format_bytes, gc_lines, plan_gc
from mlops.application.listing import list_runs
from mlops.application.predict import ensemble_runs, load_result, predict_run
from mlops.application.progress import PROGRESS, Progress, save_progress
from mlops.application.reproduce import reproduce_run
from mlops.cli import build_parser, main
from mlops.config import load_experiment_yaml
from mlops.domain import compute_experiment_id
from mlops.registry import discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"
NOW = 1_000_000.0


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments")


@pytest.fixture
def spec():
    return load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")


@pytest.fixture
def store():
    return InMemoryArtifactStore()


def _data() -> SyntheticBlobs:
    return SyntheticBlobs(n_train=4, n_val=2, shape=(8, 8, 8), in_channels=1, num_classes=2)


def _run(store, spec, **kw) -> str:
    return run_experiment(spec, data=_data(), tracker=InMemoryTracker(), artifacts=store, **kw).run_id


def _seed(spec, seed):
    return spec.model_copy(update={"training": spec.training.model_copy(update={"seed": seed})})


def _fail(store, run_id):
    store.put(run_id, MANIFEST, store.get(run_id, MANIFEST).replace(b'"FINISHED"', b'"FAILED"'))
    store.delete(run_id, MODEL)


def _failed(store, spec, **kw) -> str:
    run_id = _run(store, spec, **kw)
    _fail(store, run_id)
    return run_id


def _interrupted(store, spec) -> str:
    run_id = _run(store, spec, checkpoint_every=1, keep_checkpoint=True)
    store.delete(run_id, MANIFEST)
    store.delete(run_id, MODEL)
    return run_id


def _incomplete(store, spec) -> str:
    run_id = _run(store, spec)
    store.delete(run_id, MANIFEST)
    store.delete(run_id, MODEL)
    return run_id


def _beat(store, run_id, updated=NOW, state="running"):
    save_progress(
        store, Progress(run_id=run_id, state=state, epoch=0, epochs=1, started=updated, updated=updated, history=[])
    )


def _ids(items):
    return sorted(i.run_id for i in items)


# --- what the plan selects ------------------------------------------------------


def test_empty_root(store):
    plan = plan_gc(store, now=NOW)
    assert plan.delete == [] and plan.kept == [] and plan.warnings == []
    assert gc_lines(plan) == ["would delete 0 run(s) and leftovers of 0 finished run(s) (0 B)"]


def test_clean_finished_run_is_left_alone(spec, store):
    _run(store, spec)
    plan = plan_gc(store, now=NOW)
    assert plan.delete == [] and plan.kept == []


def test_failed_and_incomplete_runs_go_whole(spec, store):
    failed = _failed(store, spec)
    incomplete = _incomplete(store, _seed(spec, 1))
    plan = plan_gc(store, now=NOW)
    assert _ids(plan.delete) == sorted([failed, incomplete])
    assert all(i.whole_run for i in plan.delete)
    by_id = {i.run_id: i for i in plan.delete}
    assert by_id[failed].reason == "epoch 1/1, not resumable"
    assert by_id[incomplete].reason == "no manifest, no checkpoint"
    # spec.json goes last, the manifest before the remaining files
    assert by_id[failed].files[-1] == SPEC and by_id[failed].files[0] == MANIFEST


def test_resumable_runs_need_the_flag(spec, store):
    failed = _failed(store, spec, checkpoint_every=1, keep_checkpoint=True)
    interrupted = _interrupted(store, _seed(spec, 1))
    plan = plan_gc(store, now=NOW)
    assert plan.delete == []
    assert _ids(plan.kept) == sorted([failed, interrupted])
    assert all(i.reason == "resumable (--resumable to delete)" for i in plan.kept)

    plan = plan_gc(store, GcPolicy(resumable=True), now=NOW)
    assert _ids(plan.delete) == sorted([failed, interrupted])
    order = {i.run_id: i.files for i in plan.delete}[failed]
    assert order[:2] == (CHECKPOINT, MANIFEST)  # nothing half-resumable is ever left behind


def test_live_runs_are_never_touched(spec, store):
    running = _incomplete(store, spec)
    _beat(store, running, updated=NOW - 5)
    # A FAILED run being resumed: its old manifest still says FAILED, but the heartbeat is live.
    resuming = _failed(store, _seed(spec, 1), checkpoint_every=1, keep_checkpoint=True)
    _beat(store, resuming, updated=NOW - 1)
    plan = plan_gc(store, GcPolicy(resumable=True), now=NOW)
    assert plan.delete == []
    reasons = {i.run_id: i.reason for i in plan.kept}
    assert reasons == {running: "live, heartbeat 5s ago", resuming: "live, heartbeat 1s ago"}


def test_stale_heartbeat_is_not_live(spec, store):
    run_id = _incomplete(store, spec)
    _beat(store, run_id, updated=NOW - 3600)
    assert _ids(plan_gc(store, now=NOW).delete) == [run_id]


def test_unreadable_runs_are_kept(spec, store):
    store.put("broken", SPEC, b"{not json")
    plan = plan_gc(store, now=NOW)
    assert plan.delete == []
    assert plan.kept[0].run_id == "broken" and plan.kept[0].reason.startswith("unreadable: ")


def test_finished_leftovers_go_and_the_bundle_stays(spec, store):
    run_id = _run(store, spec, checkpoint_every=1, keep_checkpoint=True)
    _beat(store, run_id, updated=NOW - 3600, state="ended")
    plan = plan_gc(store, now=NOW)
    (item,) = plan.delete
    assert not item.whole_run and item.files == (CHECKPOINT, PROGRESS)
    assert item.reason == "leftover checkpoint.pt, progress.json"
    before = set(store.list(run_id))
    apply_gc(store, plan, now=NOW)
    assert set(store.list(run_id)) == before - {CHECKPOINT, PROGRESS}
    # still a complete, reproducible bundle
    report = reproduce_run(store, run_id, data=_data(), tracker=InMemoryTracker(), scratch=InMemoryArtifactStore())
    assert report.ok


def test_experiment_prefix_limits_the_scope(spec, store):
    a = _failed(store, spec)
    _failed(store, _seed(spec, 1))
    exp = compute_experiment_id(spec)
    assert _ids(plan_gc(store, GcPolicy(experiment=exp[:12]), now=NOW).delete) == [a]


# --- old attempts -----------------------------------------------------------------


def test_keep_attempts_keeps_the_newest(spec, store):
    ids = {a: _run(store, spec, attempt=a) for a in (1, 2, 3, 4)}
    other = _run(store, _seed(spec, 1))  # another experiment, one attempt: untouched
    plan = plan_gc(store, GcPolicy(keep_attempts=2), now=NOW)
    assert _ids(plan.delete) == sorted([ids[1], ids[2]])
    assert all(i.whole_run and i.reason == "older attempt (keeping the newest 2)" for i in plan.delete)
    apply_gc(store, plan, now=NOW)
    assert sorted(r.run_id for r in list_runs(store)) == sorted([ids[3], ids[4], other])


def test_keep_attempts_must_be_positive(store):
    with pytest.raises(GcError, match="keep_attempts"):
        plan_gc(store, GcPolicy(keep_attempts=0))


def test_runs_used_by_a_result_are_protected(spec, store):
    ids = {a: _run(store, spec, attempt=a) for a in (1, 2, 3, 4)}
    result = predict_run(store, ids[2], data=_data(), part="val")
    plan = plan_gc(store, GcPolicy(keep_attempts=2), now=NOW)
    # attempt 2 does not count towards the two kept, and stays
    assert _ids(plan.delete) == [ids[1]]
    assert [(i.run_id, i.reason) for i in plan.kept] == [(ids[2], f"used by result {result.result_id[:12]}")]
    apply_gc(store, plan, now=NOW)
    assert load_result(store, result.result_id).members[0].run_id == ids[2]
    assert result.result_id in store.runs()  # result directories are never touched


def test_ensemble_members_are_protected(spec, store):
    a = _run(store, spec, attempt=1)
    b = _run(store, spec, attempt=2)
    ens = ensemble_runs(store, [a, b], vote="soft", data=_data(), part="val")
    plan = plan_gc(store, GcPolicy(keep_attempts=1), now=NOW)
    assert plan.delete == []
    assert _ids(plan.kept) == sorted([a, b])
    assert ens.result_id in store.runs()


def test_unreadable_result_is_a_warning(spec, store):
    _run(store, spec)
    store.put("res", "result.json", b"{bad")
    plan = plan_gc(store, now=NOW)
    assert plan.warnings and "result res" in plan.warnings[0]


# --- apply ------------------------------------------------------------------------


def test_plan_alone_changes_nothing(spec, tmp_path):
    store = LocalArtifactStore(tmp_path)
    _failed(store, spec)
    _run(store, _seed(spec, 1), checkpoint_every=1, keep_checkpoint=True)

    def tree():
        return {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    before = tree()
    plan = plan_gc(store, GcPolicy(keep_attempts=1), now=NOW)
    assert len(plan.delete) == 2
    assert tree() == before


def test_apply_removes_the_run_directory(spec, tmp_path):
    store = LocalArtifactStore(tmp_path)
    run_id = _failed(store, spec)
    store.put(run_id, "extra/nested/notes.txt", b"x")
    outcome = apply_gc(store, plan_gc(store, now=NOW), now=NOW)
    assert _ids(outcome.deleted) == [run_id] and outcome.skipped == []
    assert not (tmp_path / run_id).exists()
    assert list_runs(store) == []


def test_run_that_went_live_since_the_plan_is_skipped(spec, store):
    run_id = _failed(store, spec, checkpoint_every=1, keep_checkpoint=True)
    plan = plan_gc(store, GcPolicy(resumable=True), now=NOW)
    assert _ids(plan.delete) == [run_id]
    _beat(store, run_id, updated=NOW)  # someone ran `mlops resume` meanwhile
    outcome = apply_gc(store, plan, now=NOW)
    assert outcome.deleted == []
    assert [(i.run_id, why) for i, why in outcome.skipped] == [(run_id, "now live (started or resumed since the plan)")]
    assert MANIFEST in store.list(run_id)


def test_run_whose_status_changed_is_skipped(spec, store):
    run_id = _failed(store, spec)
    plan = plan_gc(store, now=NOW)
    store.put(run_id, MANIFEST, store.get(run_id, MANIFEST).replace(b'"FAILED"', b'"FINISHED"'))
    outcome = apply_gc(store, plan, now=NOW)
    assert [why for _, why in outcome.skipped] == ["now FINISHED"]


def test_crash_halfway_leaves_an_incomplete_run_the_next_gc_removes(spec, store):
    run_id = _failed(store, spec, checkpoint_every=1, keep_checkpoint=True)
    (item,) = plan_gc(store, GcPolicy(resumable=True), now=NOW).delete
    for name in item.files[:2]:  # checkpoint.pt, manifest.json: then the process dies
        store.delete(run_id, name)
    (s,) = list_runs(store)
    assert s.status == "INCOMPLETE"
    apply_gc(store, plan_gc(store, now=NOW), now=NOW)
    assert store.runs() == []


# --- Ray storage ------------------------------------------------------------------


def _ray_run(store, spec, storage: str, *, finished: bool) -> tuple[str, str]:
    run_id = _run(store, spec)
    if not finished:
        _fail(store, run_id)
    save_launch(store, run_id, LaunchRecord(launcher="ray", attempt=1, checkpoint_every=1, storage_path=storage))
    return run_id, ray_run_name(run_id, 1)


def test_ray_storage_of_a_deleted_run_goes_too(spec, store, tmp_path):
    run_id, name = _ray_run(store, spec, str(tmp_path), finished=False)
    (tmp_path / name / "checkpoint_000001").mkdir(parents=True)
    (tmp_path / name / "checkpoint_000001" / "state.pt").write_bytes(b"x" * 10)
    ray = LocalRayStorage()
    plan = plan_gc(store, GcPolicy(resumable=True), ray_storage=ray, now=NOW)
    (item,) = plan.delete
    assert item.ray_dir is not None and str(item.ray_dir) == f"{tmp_path}/{name}"
    assert ray.size(item.ray_dir.storage_path, item.ray_dir.name) == 10
    apply_gc(store, plan, ray_storage=ray, now=NOW)
    assert not (tmp_path / name).exists() and store.runs() == []


def test_ray_storage_is_left_alone_without_the_flag(spec, store, tmp_path):
    run_id, name = _ray_run(store, spec, str(tmp_path), finished=False)
    (tmp_path / name).mkdir()
    plan = plan_gc(store, GcPolicy(resumable=True), now=NOW)
    assert plan.delete[0].ray_dir is None
    assert plan.warnings == [f"run {run_id[:12]}: Ray checkpoints at {tmp_path}/{name} stay (pass --ray-storage)"]


def test_finished_ray_run_with_kept_checkpoints(spec, store, tmp_path):
    _, name = _ray_run(store, spec, "/ray-storage", finished=True)  # path as the pod saw it
    (tmp_path / name).mkdir()
    ray = LocalRayStorage(mount=tmp_path)  # same PVC, mounted here
    (item,) = plan_gc(store, ray_storage=ray, now=NOW).delete
    assert not item.whole_run and item.files == () and item.reason == "leftover Ray checkpoints"
    apply_gc(store, plan_gc(store, ray_storage=ray, now=NOW), ray_storage=ray, now=NOW)
    assert not (tmp_path / name).exists()
    assert list_runs(store)[0].status == "FINISHED"


def test_finished_ray_run_whose_checkpoints_are_gone_is_left_alone(spec, store, tmp_path):
    _ray_run(store, spec, str(tmp_path), finished=True)
    assert plan_gc(store, ray_storage=LocalRayStorage(), now=NOW).delete == []


def test_remote_ray_storage_is_only_reported(spec, store):
    run_id, name = _ray_run(store, spec, "s3://bucket/ray", finished=False)
    plan = plan_gc(store, GcPolicy(resumable=True), ray_storage=LocalRayStorage(), now=NOW)
    assert plan.delete[0].ray_dir is None
    assert "s3://bucket/ray/" + name in plan.warnings[0] and "by hand" in plan.warnings[0]


def test_ray_storage_refuses_a_foreign_directory_name(tmp_path):
    with pytest.raises(ValueError, match="not a Ray run name"):
        LocalRayStorage().delete(str(tmp_path), "../home")


# --- text + CLI -------------------------------------------------------------------


def test_format_bytes():
    assert [format_bytes(n) for n in (None, 0, 1023, 2048, 3 * 1024**3)] == ["?", "0 B", "1023 B", "2.0 KB", "3.0 GB"]


def test_cli_parses():
    args = build_parser().parse_args(
        ["gc", "--artifacts", "a", "--keep-attempts", "2", "--resumable", "--ray-storage-mount", "m", "--apply"]
    )
    assert (args.keep_attempts, args.resumable, args.ray_storage_mount, args.apply) == (2, True, "m", True)


def test_cli_dry_run_then_apply(spec, tmp_path, capsys):
    store = LocalArtifactStore(tmp_path)
    failed = _failed(store, spec)
    kept = _run(store, _seed(spec, 1), checkpoint_every=1, keep_checkpoint=True)

    assert main(["gc", "--artifacts", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("would delete 1 run(s) and leftovers of 1 finished run(s) (")
    assert f"run   {failed[:12]}" in out and f"file  {kept[:12]}" in out
    assert "dry run: nothing deleted" in out
    assert (tmp_path / failed).exists()

    assert main(["gc", "--artifacts", str(tmp_path), "--apply", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["applied"] is True and doc["bytes"][failed] > 0
    assert sorted(i["run_id"] for i in doc["outcome"]["deleted"]) == sorted([failed, kept])
    assert not (tmp_path / failed).exists() and not (tmp_path / kept / CHECKPOINT).exists()
    assert (tmp_path / kept / MODEL).exists()


def test_cli_bad_input_exits_2(tmp_path, capsys):
    assert main(["gc", "--artifacts", str(tmp_path / "nope")]) == 2
    assert main(["gc", "--artifacts", str(tmp_path), "--keep-attempts", "0"]) == 2
    err = capsys.readouterr().err
    assert "no such directory" in err and "keep_attempts" in err and "Traceback" not in err


def test_live_check_uses_the_real_clock_by_default(spec, store):
    run_id = _incomplete(store, spec)
    _beat(store, run_id, updated=time.time())
    assert plan_gc(store).delete == []
