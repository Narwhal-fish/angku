"""Use case: clean up an artifact root (``angku gc``, roadmap 18).

``plan_gc`` decides, ``apply_gc`` deletes; nothing is deleted without the second call, so the
CLI's default is a dry run. Statuses come from ``list_runs``, so gc sees exactly what
``angku list`` / ``angku ui`` show:

    RUNNING       always kept (a live process is writing it); so is any run with a live
                  heartbeat, whatever its manifest says: resuming a FAILED run keeps the
                  FAILED manifest until the new one lands
    UNREADABLE    always kept (nobody knows what it is)
    FAILED        deleted; kept when resumable unless ``resumable=True``
    INTERRUPTED   (always resumable) kept unless ``resumable=True``
    INCOMPLETE    deleted (no manifest, no checkpoint, no live heartbeat)
    FINISHED      the bundle stays; only leftovers go: ``checkpoint.pt`` (kept with
                  ``--keep-checkpoint`` or a crash before its delete), a stale ``progress.json``
                  and, with Ray storage, the Ray run dir (kept with ``--keep-checkpoint``).
                  ``keep_attempts=N`` also deletes all but the newest N FINISHED attempts of
                  each ExperimentID.

A run that is a member of a prediction result (``result.json``) is never deleted: the result
must stay reproducible from its bundles. Result directories themselves are never touched.

Deletion order inside a run keeps every intermediate state a valid listing: Ray storage first,
then ``checkpoint.pt`` / ``launch.json`` (nothing resumable is left half-deleted), then the
manifest (the run stops claiming FINISHED/FAILED before its files go), then the rest,
``spec.json`` last. A crash in between leaves an INCOMPLETE run the next gc removes.

``apply_gc`` re-reads every run just before deleting it and skips one whose status changed
since the plan (resumed, started, finished meanwhile).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from mlops.application.bundle import CHECKPOINT, LAUNCH, MANIFEST, SPEC, load_launch, ray_run_name
from mlops.application.listing import RunSummary, list_runs, summarise_run
from mlops.application.predict import RESULT
from mlops.application.progress import PROGRESS, is_live, read_progress
from mlops.domain.prediction import parse_result
from mlops.ports.artifacts import ArtifactError, ArtifactStore
from mlops.ports.ray_storage import RayStorage


class GcError(Exception):
    """Bad policy (e.g. keep_attempts < 1)."""


@dataclass(frozen=True)
class GcPolicy:
    resumable: bool = False  # also delete FAILED-resumable and INTERRUPTED runs
    keep_attempts: int | None = None  # keep the newest N FINISHED attempts per experiment
    experiment: str | None = None  # only ExperimentIDs starting with this


@dataclass(frozen=True)
class RayDir:
    storage_path: str
    name: str

    def __str__(self) -> str:
        return f"{self.storage_path.rstrip('/')}/{self.name}"


@dataclass(frozen=True)
class GcItem:
    run_id: str
    name: str
    status: str
    resumable: bool
    reason: str
    whole_run: bool = False  # True: every file of the run goes; False: only ``files``
    files: tuple[str, ...] = ()
    ray_dir: RayDir | None = None  # Ray storage to remove as well

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["files"] = list(self.files)
        d["ray_dir"] = str(self.ray_dir) if self.ray_dir is not None else None
        return d


@dataclass(frozen=True)
class GcPlan:
    delete: list[GcItem] = field(default_factory=list)
    kept: list[GcItem] = field(default_factory=list)  # what one might expect gone, and why not
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delete": [i.to_dict() for i in self.delete],
            "kept": [i.to_dict() for i in self.kept],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class GcOutcome:
    deleted: list[GcItem] = field(default_factory=list)
    skipped: list[tuple[GcItem, str]] = field(default_factory=list)  # (item, why)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deleted": [i.to_dict() for i in self.deleted],
            "skipped": [{**i.to_dict(), "skipped": why} for i, why in self.skipped],
        }


def plan_gc(
    store: ArtifactStore,
    policy: GcPolicy | None = None,
    *,
    ray_storage: RayStorage | None = None,
    now: float | None = None,
) -> GcPlan:
    """What ``apply_gc`` would delete. Reads spec, manifest, launch.json and result.json only.
    ``ray_storage`` None = leave Ray storage alone (and warn about what stays there)."""
    policy = policy or GcPolicy()
    if policy.keep_attempts is not None and policy.keep_attempts < 1:
        raise GcError(f"keep_attempts must be >= 1, got {policy.keep_attempts}")
    now = time.time() if now is None else now
    warnings: list[str] = []
    used_by, blind = _result_members(store, warnings)
    delete: list[GcItem] = []
    kept: list[GcItem] = []
    runs = list_runs(store, experiment=policy.experiment, now=now)

    def whole(s: RunSummary, reason: str) -> None:
        if blind:
            kept.append(_item(s, f"{reason}; kept: a result.json is unreadable"))
            return
        if s.run_id in used_by:
            kept.append(_item(s, f"used by result {used_by[s.run_id][:12]}"))
            return
        files = store.list(s.run_id)
        ray_dir = _ray_dir(store, s, files, ray_storage, warnings)
        delete.append(_item(s, reason, whole_run=True, files=_deletion_order(files), ray_dir=ray_dir))

    old = _old_attempts(runs, policy.keep_attempts, used_by, kept) if policy.keep_attempts else set()
    for s in runs:
        heartbeat = _heartbeat(store, s.run_id, now)
        if heartbeat is not None:  # RUNNING, or a FAILED run being resumed (its old manifest still says FAILED)
            kept.append(_item(s, f"live, heartbeat {now - heartbeat:.0f}s ago"))
        elif s.status == "UNREADABLE":
            kept.append(_item(s, f"unreadable: {s.error}"))
        elif s.status in ("FAILED", "INTERRUPTED") and s.resumable and not policy.resumable:
            kept.append(_item(s, "resumable (--resumable to delete)"))
        elif s.status in ("FAILED", "INTERRUPTED"):
            whole(s, _progress_text(s) + (", resumable" if s.resumable else ", not resumable"))
        elif s.status == "INCOMPLETE":
            whole(s, "no manifest, no checkpoint")
        elif s.run_id in old:
            whole(s, f"older attempt (keeping the newest {policy.keep_attempts})")
        else:  # FINISHED
            files = store.list(s.run_id)
            leftovers = tuple(f for f in (CHECKPOINT, PROGRESS) if f in files)
            ray_dir = _ray_dir(store, s, files, ray_storage, warnings)
            if leftovers or ray_dir is not None:
                what = [*leftovers] + (["Ray checkpoints"] if ray_dir is not None else [])
                delete.append(_item(s, "leftover " + ", ".join(what), files=leftovers, ray_dir=ray_dir))
    return GcPlan(delete=delete, kept=kept, warnings=warnings)


def apply_gc(
    store: ArtifactStore,
    plan: GcPlan,
    *,
    ray_storage: RayStorage | None = None,
    now: float | None = None,
) -> GcOutcome:
    """Delete what ``plan`` says, re-checking each run first; a run whose status or
    resumability changed since the plan, or that a result now uses, is skipped."""
    used_by, blind = _result_members(store, [])
    deleted: list[GcItem] = []
    skipped: list[tuple[GcItem, str]] = []
    for item in plan.delete:
        t = time.time() if now is None else now
        s = summarise_run(store, item.run_id, now=t)
        if s is None:
            skipped.append((item, "gone"))
            continue
        if _heartbeat(store, item.run_id, t) is not None:
            skipped.append((item, "now live (started or resumed since the plan)"))
            continue
        if (s.status, s.resumable) != (item.status, item.resumable):
            skipped.append((item, f"now {s.status}" + (" (resumable)" if s.resumable else "")))
            continue
        if item.whole_run and blind:
            skipped.append((item, "a result.json is now unreadable"))
            continue
        if item.whole_run and item.run_id in used_by:
            skipped.append((item, f"now used by result {used_by[item.run_id][:12]}"))
            continue
        if item.ray_dir is not None:
            if ray_storage is None:
                skipped.append((item, f"Ray storage {item.ray_dir} needs a RayStorage"))
                continue
            ray_storage.delete(item.ray_dir.storage_path, item.ray_dir.name)
        # whole run: the files as they are now, so nothing written since the plan stays behind
        files = _deletion_order(store.list(item.run_id)) if item.whole_run else item.files
        for name in files:
            store.delete(item.run_id, name)
        deleted.append(item)
    return GcOutcome(deleted=deleted, skipped=skipped)


# -- helpers -----------------------------------------------------------------------


def _item(s: RunSummary, reason: str, **kw: Any) -> GcItem:
    return GcItem(run_id=s.run_id, name=s.name, status=s.status, resumable=s.resumable, reason=reason, **kw)


def _heartbeat(store: ArtifactStore, run_id: str, now: float) -> float | None:
    """Time of the last heartbeat when a live process is writing ``run_id``, else None. Checked
    even when a manifest exists: ``angku resume`` of a FAILED run keeps the FAILED manifest."""
    p = read_progress(store, run_id)
    return p.updated if p is not None and is_live(p, now=now) else None


def _progress_text(s: RunSummary) -> str:
    done = "?" if s.epochs_done is None else s.epochs_done
    return f"epoch {done}/{s.epochs_total or '?'}"


def _deletion_order(files: list[str]) -> tuple[str, ...]:
    first = [f for f in (CHECKPOINT, LAUNCH) if f in files]
    manifest = [MANIFEST] if MANIFEST in files else []
    last = [SPEC] if SPEC in files else []
    rest = [f for f in files if f not in (*first, MANIFEST, SPEC)]
    return (*first, *manifest, *rest, *last)


def _result_members(store: ArtifactStore, warnings: list[str]) -> tuple[dict[str, str], bool]:
    """{run_id: result_id} of every run some result.json lists as a member, and whether any
    result.json was unreadable."""
    out: dict[str, str] = {}
    blind = False
    for rid in store.runs():
        try:
            if RESULT not in store.list(rid):
                continue
            record = parse_result(store.get(rid, RESULT))
        except (ArtifactError, ValueError) as e:
            warnings.append(f"result {rid[:12]} is unreadable ({str(e).splitlines()[0]}): no run is deleted")
            blind = True
            continue
        for m in record.members:
            out.setdefault(m.run_id, rid)
    return out, blind


def _old_attempts(runs: list[RunSummary], keep: int, used_by: dict[str, str], kept: list[GcItem]) -> set[str]:
    """Run ids of FINISHED attempts beyond the newest ``keep`` of their experiment. Runs used by
    a result never count towards ``keep`` and are never returned."""
    old: set[str] = set()
    groups: dict[str, list[RunSummary]] = {}
    for s in runs:
        if s.status == "FINISHED":
            groups.setdefault(s.experiment_id, []).append(s)
    for group in groups.values():
        newest_first = sorted(group, key=lambda r: (r.attempt or 0, r.run_id), reverse=True)
        free = [r for r in newest_first if r.run_id not in used_by]
        old.update(r.run_id for r in free[keep:])
        if len(group) > keep:
            kept.extend(_item(r, f"used by result {used_by[r.run_id][:12]}") for r in group if r.run_id in used_by)
    return old


def _ray_dir(
    store: ArtifactStore,
    s: RunSummary,
    files: list[str],
    ray_storage: RayStorage | None,
    warnings: list[str],
) -> RayDir | None:
    """The run's Ray storage dir when it exists and may go; warns about what has to stay."""
    if LAUNCH not in files:
        return None
    try:
        launch = load_launch(store, s.run_id)
    except (ArtifactError, ValueError, TypeError):
        warnings.append(f"run {s.run_id[:12]}: launch.json is unreadable; its Ray storage is not checked")
        return None
    if launch.storage_path is None:
        return None
    d = RayDir(launch.storage_path, ray_run_name(s.run_id, launch.attempt))
    if ray_storage is None:
        if s.status != "FINISHED":  # FINISHED: usually already deleted by the launcher
            warnings.append(f"run {s.run_id[:12]}: Ray checkpoints at {d} stay (pass --ray-storage)")
        return None
    exists = ray_storage.exists(d.storage_path, d.name)
    if exists is None:
        warnings.append(f"run {s.run_id[:12]}: Ray storage {d} is not reachable from here; delete it by hand")
        return None
    return d if exists else None


# -- text --------------------------------------------------------------------------


def format_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def gc_lines(
    plan: GcPlan, *, sizes: dict[str, int | None] | None = None, outcome: GcOutcome | None = None
) -> list[str]:
    """What ``angku gc`` prints: the plan (dry run) or, with ``outcome``, what was deleted.
    ``sizes`` = bytes per run_id of what the item frees (None/missing = unknown)."""
    sizes = sizes or {}
    done = plan.delete if outcome is None else outcome.deleted
    runs = sum(i.whole_run for i in done)
    leftovers = len(done) - runs
    known = [sizes.get(i.run_id) for i in done]
    total = format_bytes(sum(s for s in known if s is not None)) + ("+" if None in known and done else "")
    verb = "would delete" if outcome is None else "deleted"
    lines = [f"{verb} {runs} run(s) and leftovers of {leftovers} finished run(s) ({total})"]
    if done or plan.kept:
        lines.append("")
    for i in done:
        kind = "run " if i.whole_run else "file"
        size = format_bytes(sizes.get(i.run_id))
        lines.append(f"{kind}  {i.run_id[:12]}  {i.name}  {i.status}  {i.reason}  ({size})")
    if outcome is not None:
        for i, why in outcome.skipped:
            lines.append(f"skip  {i.run_id[:12]}  {i.name}  {i.status}  {why}")
    for i in plan.kept:
        lines.append(f"kept  {i.run_id[:12]}  {i.name}  {i.status}  {i.reason}")
    if plan.warnings:
        lines.append("")
        lines.extend(f"warning: {w}" for w in plan.warnings)
    if outcome is None and plan.delete:
        lines += ["", "dry run: nothing deleted. Re-run with --apply to delete."]
    return lines
