"""``mlops`` command line. The composition root: the one place that picks
concrete adapters (tracker, data provider, artifact store) and hands them to
``run_experiment``. Nothing under ``mlops/application`` knows this exists.

    mlops id     experiment.yaml
    mlops run    experiment.yaml [--artifacts DIR] [--data-root DIR] [--tracker mlflow|inmemory]
                                 [--mlflow-uri URI] [--launcher local|ray] [--ray-address ADDR]
                                 [--device cpu] [--attempt N] [--checkpoint-every N] [--keep-checkpoint]
                                 [--keep-best METRIC:min|max] [--strict-env] [--json]
    mlops resume RUN_ID --artifacts DIR [--data-root DIR] [--tracker mlflow|inmemory] [--mlflow-uri URI]
                                 [--device cpu] [--ray-address ADDR] [--checkpoint-every N] [--keep-checkpoint]
                                 [--json]
    mlops rayjob experiment.yaml [--namespace NS] [--artifacts-pvc PVC] [--data-pvc PVC] [--attempt N]
                                 [--checkpoint-every N --ray-storage-pvc PVC] [--resume]
                                 [--submit [--kubectl "kubectl --context X"]]
    mlops rayjob-status NAME [--namespace NS] [--kubectl CMD] [--json]
    mlops rayjob-logs   NAME [--namespace NS] [--kubectl CMD] [-f]
    mlops reproduce RUN_ID --artifacts DIR [--data-root DIR] [--tracker mlflow|inmemory] [--mlflow-uri URI]
                                 [--device cpu] [--atol X] [--json]
    mlops report RUN_ID --artifacts DIR [--out FILE]
    mlops diff   RUN_A RUN_B --artifacts DIR [--artifacts-b DIR] [--json]
    mlops list   --artifacts DIR [--experiment ID_PREFIX] [--json]
    mlops snapshot experiment.yaml [--data-root DIR]

Exit codes: 0 ok, 1 training failed / bundle not reproduced / diff: different experiments,
2 bad input (yaml, pins, wiring).

Environment: every run records what it actually ran on (code vs code.commit, image digest,
library versions, GPUs) and the data provider's content hash vs dataset.snapshot in
manifest.json, and warns when they cannot be verified; ``--strict-env`` refuses to start
instead. Data that verifiably differs from dataset.snapshot always refuses (exit 2).

Datasets: built-in ``synthetic-blobs``, plus whatever ``experiments/`` registers in
``mlops.registry.DATASETS``; those read files under ``--data-root`` (default: $MLOPS_DATA_ROOT).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.kuberay import (
    KubeRayError,
    active_rayjobs,
    follow_rayjob_logs,
    rayjob_status,
    render_rayjob,
    submit_rayjob,
)
from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.adapters.local_environment import LocalEnvironmentProbe
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import RunnerError
from mlops.application.bundle import LAUNCH, MANIFEST, SPEC, load_bundle, load_checkpoint, load_launch
from mlops.application.launch import Launcher, LocalLauncher
from mlops.application.runner import verify_data
from mlops.application.diff import ABSENT, Change, RunDiff, diff_runs
from mlops.application.listing import ExperimentGroup, group_by_experiment, list_runs
from mlops.application.report import render_report, write_report
from mlops.application.reproduce import ReproduceError, ReproduceReport, reproduce_run
from mlops.config import ConfigError, load_experiment_yaml
from mlops.domain import ExperimentSpec, RunManifest, UnpinnedInputError, compute_experiment_id, compute_run_id
from mlops.domain.best import BestCriterion
from mlops.domain.environment import EnvCheck, record_environment
from mlops.ports.artifacts import ArtifactError
from mlops.ports.data import DataError, DataProvider
from mlops.ports.environment import EnvironmentProbe
from mlops.ports.tracking import Tracker, TrackerError
from mlops.registry import DATASETS, RegistryError, discover

# Built-in dataset name -> provider factory. Real datasets are plugins registered in DATASETS.
_DATA_PROVIDERS: dict[str, Callable[[], DataProvider]] = {
    "synthetic-blobs": SyntheticBlobs,
}


class CliError(Exception):
    """Bad input; reported without a traceback, exit 2."""


def _load(path: str) -> ExperimentSpec:
    if not Path(path).is_file():
        raise CliError(f"no such file: {path}")
    try:
        return load_experiment_yaml(path)
    except ConfigError as e:
        raise CliError(str(e)) from e


def _discover(package: str) -> None:
    # `mlops` as a console script does not put the cwd on sys.path; `python -m` does.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        discover(package)
    except ImportError as e:
        raise CliError(f"cannot import experiments package {package!r}: {e}") from e


def _data_provider(spec: ExperimentSpec, data_root: str | None) -> DataProvider:
    name = spec.dataset.name
    if name in _DATA_PROVIDERS:
        return _DATA_PROVIDERS[name]()
    if name not in DATASETS:
        known = ", ".join(sorted({*_DATA_PROVIDERS, *DATASETS.names()}))
        raise CliError(f"no DataProvider for dataset {name!r}; known: {known}")
    if not data_root:
        raise CliError(f"dataset {name!r} reads files: pass --data-root DIR or set MLOPS_DATA_ROOT")
    try:
        return DATASETS.build(name, {"root": data_root})
    except RegistryError as e:
        raise CliError(str(e)) from e


def _tracker(spec: ExperimentSpec, choice: str | None, mlflow_uri: str | None) -> Tracker:
    backend = choice or spec.tracking.backend
    if backend == "inmemory":
        return InMemoryTracker()
    if backend == "mlflow":
        try:
            from mlops.adapters.mlflow_tracker import MlflowTracker
        except ImportError as e:
            raise CliError("MLflow tracking needs the 'tracking' extra: uv pip install -e '.[tracking]'") from e
        if not (mlflow_uri or spec.tracking.uri):
            raise CliError("MLflow needs a uri: pass --mlflow-uri or set tracking.uri in the yaml")
        return MlflowTracker(uri=mlflow_uri)
    raise CliError(f"unknown tracker {backend!r}; known: inmemory, mlflow")


def _probe() -> EnvironmentProbe:
    return LocalEnvironmentProbe()


def _unverified(checks: dict[str, EnvCheck]) -> list[str]:
    return [f"{name}: {c.status}: {c.detail}" for name, c in checks.items() if c.status != "ok"]


def _launcher(spec: ExperimentSpec, args: argparse.Namespace) -> Launcher:
    if args.launcher == "ray":
        try:
            from mlops.adapters.ray_launcher import RayLauncher
        except ImportError as e:
            raise CliError("the Ray launcher needs the 'distributed' extra: uv pip install -e '.[distributed]'") from e
        return RayLauncher(address=args.ray_address, storage_path=args.ray_storage, experiments=args.experiments)
    if spec.resources.workers > 1:
        raise CliError(
            f"resources.workers={spec.resources.workers} needs a distributed launcher: pass --launcher ray"
        )
    return LocalLauncher()


# -- commands ------------------------------------------------------------------


def cmd_id(args: argparse.Namespace) -> int:
    spec = _load(args.experiment)
    try:
        print(compute_experiment_id(spec))
    except UnpinnedInputError as e:
        raise CliError(str(e)) from e
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = _load(args.experiment)
    if spec.unpinned_inputs():
        raise CliError("cannot run: unpinned inputs: " + ", ".join(spec.unpinned_inputs()))
    _check_checkpoint_args(args)
    keep_best = _keep_best(args.keep_best)
    _discover(args.experiments)
    data = _data_provider(spec, args.data_root)
    tracker = _tracker(spec, args.tracker, args.mlflow_uri)
    launcher = _launcher(spec, args)
    probe = _probe()
    if args.strict_env:
        # Pre-flight on this machine; the record in the manifest is captured where training runs.
        try:
            data_check = verify_data(spec, data)
        except DataError as e:
            raise CliError(str(e)) from e
        checks = {**record_environment(spec, probe.capture(spec)).checks, "dataset.snapshot": data_check}
        problems = _unverified(checks)
        if problems:
            raise CliError("--strict-env: run does not verifiably match the pins:\n  " + "\n  ".join(problems))
    store = LocalArtifactStore(args.artifacts) if args.artifacts else None
    return _launch_and_report(spec, args, launcher, store, attempt=args.attempt, probe=probe, data=data,
                              tracker=tracker, checkpoint_every=args.checkpoint_every,
                              keep_checkpoint=args.keep_checkpoint, keep_best=keep_best)


def _keep_best(text: str | None) -> BestCriterion | None:
    try:
        return BestCriterion.parse(text) if text is not None else None
    except ValueError as e:
        raise CliError(str(e)) from None


def _check_checkpoint_args(args: argparse.Namespace) -> None:
    if args.checkpoint_every is None:
        return
    if not args.artifacts:
        raise CliError("--checkpoint-every needs --artifacts DIR (the checkpoint, or Ray's launch record, goes there)")


def _launch_and_report(
    spec: ExperimentSpec, args: argparse.Namespace, launcher: Launcher, store: LocalArtifactStore | None,
    *, attempt: int, **launch: Any,
) -> int:
    """Launch, write report.md, warn about unverified pins, print the summary. Shared by run/resume."""
    try:
        result = launcher.launch(spec, artifacts=store, attempt=attempt, device=args.device, **launch)
    except (TrackerError, DataError, RegistryError, RunnerError) as e:
        raise CliError(str(e)) from e
    except Exception as e:  # training itself blew up; the FAILED bundle (if any) is already written
        if store is not None:
            run_id = compute_run_id(compute_experiment_id(spec), spec.resources, attempt)
            try:
                write_report(store, run_id)
            except ArtifactError:  # failed before the run started: no bundle to report on
                pass
        print(f"run FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if store is not None:
        write_report(store, result.run_id)
    # Warn about where *this* process ran: after a resume that is the latest segment.
    env = result.resumes[-1].environment if result.resumes else result.environment
    checks = dict(env.checks) if env is not None else {}
    if result.data_snapshot is not None:
        checks["dataset.snapshot"] = result.data_snapshot
    for problem in _unverified(checks):
        print(f"warning: not verified: {problem}", file=sys.stderr)

    summary: dict[str, Any] = {
        "status": "FINISHED",
        "experiment_id": result.experiment_id,
        "run_id": result.run_id,
        "attempt": attempt,
        "device": result.device,
        "launcher": getattr(args, "launcher", "local"),
        "resumed_from_epoch": result.resumes[-1].epoch if result.resumes else None,
        "history": result.history,
        "artifacts": str(Path(args.artifacts) / result.run_id) if args.artifacts else None,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_summary(spec, summary)
    return 0


def _print_summary(spec: ExperimentSpec, s: dict[str, Any]) -> None:
    print(f"{spec.meta.name}: {s['status']}")
    print(f"  experiment_id  {s['experiment_id']}")
    print(f"  run_id         {s['run_id']}  (attempt {s['attempt']}, {s['device']}, launcher {s['launcher']})")
    if s["resumed_from_epoch"] is not None:
        print(f"  resumed after epoch {s['resumed_from_epoch']} from checkpoint.pt")
    for epoch, metrics in enumerate(s["history"], start=1):
        cells = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  epoch {epoch:>3}  {cells}")
    if s["artifacts"]:
        print(f"  artifacts      {s['artifacts']}")


def cmd_resume(args: argparse.Namespace) -> int:
    store = LocalArtifactStore(args.artifacts)
    try:
        spec = ExperimentSpec.model_validate_json(store.get(args.run_id, SPEC))
    except ArtifactError as e:
        raise CliError(str(e)) from e
    launcher: Launcher
    try:
        finished = RunManifest.model_validate_json(store.get(args.run_id, MANIFEST)).status == "FINISHED"
    except ArtifactError:
        finished = False
    if finished:  # checked first: a finished run has usually dropped its checkpoint
        raise CliError(f"run {args.run_id} already FINISHED; nothing to resume")
    if LAUNCH in store.list(args.run_id):
        # A Ray run: its checkpoints live in Ray storage; resume through Ray with the same storage.
        launch = load_launch(store, args.run_id)
        attempt = launch.attempt
        try:
            from mlops.adapters.ray_launcher import RayLauncher
        except ImportError as e:
            raise CliError("this run was launched on Ray; resuming it needs the 'distributed' extra") from e
        launcher = RayLauncher(address=args.ray_address, storage_path=launch.storage_path,
                               experiments=args.experiments)
        args.launcher = "ray"
    else:
        try:
            attempt = load_checkpoint(store, args.run_id).attempt
        except ArtifactError:
            raise CliError(
                f"no checkpoint for run {args.run_id}: nothing to resume "
                "(only runs started with --checkpoint-every can be resumed)"
            ) from None
        launcher = LocalLauncher()
    _check_checkpoint_args(args)
    _discover(args.experiments)
    data = _data_provider(spec, args.data_root)
    tracker = _tracker(spec, args.tracker, args.mlflow_uri)
    return _launch_and_report(spec, args, launcher, store, attempt=attempt, probe=_probe(), data=data,
                              tracker=tracker, checkpoint_every=args.checkpoint_every, resume=True,
                              keep_checkpoint=args.keep_checkpoint)


def cmd_rayjob(args: argparse.Namespace) -> int:
    spec = _load(args.experiment)
    if spec.dataset.name not in _DATA_PROVIDERS and not args.data_pvc:
        raise CliError(
            f"dataset {spec.dataset.name!r} reads files: pass --data-pvc PVC (a claim holding them, "
            "mounted read-only at --data-path) so the in-cluster run can find its data"
        )
    if args.checkpoint_every is not None or args.resume:
        # Checkpoints and launch.json must outlive the RayJob's cluster (shutdownAfterJobFinishes).
        for flag, value, what in (("--artifacts-pvc", args.artifacts_pvc, "spec.json + launch.json"),
                                  ("--ray-storage-pvc", args.ray_storage_pvc, "Ray Train's checkpoints")):
            if not value:
                raise CliError(f"a resumable RayJob needs {flag} PVC ({what} must outlive the job's cluster)")
    try:
        items = render_rayjob(
            spec,
            namespace=args.namespace,
            attempt=args.attempt,
            artifacts_pvc=args.artifacts_pvc,
            artifacts_path=args.artifacts_path,
            mlflow_uri=args.mlflow_uri or spec.tracking.uri or None,
            data_pvc=args.data_pvc,
            data_path=args.data_path,
            checkpoint_every=args.checkpoint_every,
            ray_storage_pvc=args.ray_storage_pvc,
            ray_storage_path=args.ray_storage_path,
            resume=args.resume,
            keep_best=str(_keep_best(args.keep_best)) if args.keep_best else None,
        )
    except (UnpinnedInputError, ValueError) as e:
        raise CliError(str(e)) from e
    if not args.submit:
        print(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=2))
        return 0
    kubectl = shlex.split(args.kubectl)
    try:
        if args.resume:
            # Two jobs of one run would write the same Ray checkpoints: only resume a stopped run.
            run_id = items[-1]["metadata"]["annotations"]["mlops.io/run-id"]
            busy = active_rayjobs(run_id, namespace=args.namespace, kubectl=kubectl, run=subprocess.run)
            if busy:
                raise CliError("this run still has a RayJob that has not finished; wait for it or stop it first:\n  "
                               + "\n  ".join(f"{j.name}: {j.job_status or 'no status yet'}" for j in busy))
        name = submit_rayjob(items, kubectl=kubectl, run=subprocess.run)
    except (KubeRayError, OSError) as e:
        raise CliError(str(e)) from e
    print(f"submitted RayJob {name} in namespace {args.namespace}")
    follow = f"--namespace {args.namespace}" + (f" --kubectl '{args.kubectl}'" if args.kubectl != "kubectl" else "")
    print(f"  status: mlops rayjob-status {name} {follow}")
    print(f"  logs:   mlops rayjob-logs {name} {follow} -f")
    if args.checkpoint_every is not None and not args.resume:
        print(f"  resume: mlops rayjob {args.experiment} --resume {_rayjob_volume_flags(args)} --submit")
    return 0


def _rayjob_volume_flags(args: argparse.Namespace) -> str:
    """The flags a resume job must repeat: same run (attempt), same claims at the same paths."""
    flags = [f"--namespace {args.namespace}", f"--attempt {args.attempt}",
             f"--artifacts-pvc {args.artifacts_pvc}", f"--ray-storage-pvc {args.ray_storage_pvc}"]
    if args.artifacts_path != "/artifacts":
        flags.append(f"--artifacts-path {args.artifacts_path}")
    if args.ray_storage_path != "/ray-storage":
        flags.append(f"--ray-storage-path {args.ray_storage_path}")
    if args.data_pvc:
        flags.append(f"--data-pvc {args.data_pvc}" + (f" --data-path {args.data_path}" if args.data_path != "/data" else ""))
    if args.kubectl != "kubectl":
        flags.append(f"--kubectl '{args.kubectl}'")
    return " ".join(flags)


def cmd_rayjob_status(args: argparse.Namespace) -> int:
    try:
        st = rayjob_status(args.name, namespace=args.namespace, kubectl=shlex.split(args.kubectl), run=subprocess.run)
    except (KubeRayError, OSError) as e:
        raise CliError(str(e)) from e
    if args.json:
        print(json.dumps(st.to_dict(), indent=2))
    else:
        print(f"{st.name}: {st.job_status or 'no status yet'} (deployment {st.deployment_status or '-'})")
        for label, value in (("run_id", st.run_id), ("experiment_id", st.experiment_id), ("message", st.message),
                             ("dashboard", st.dashboard_url), ("started", st.start_time), ("ended", st.end_time)):
            if value:
                print(f"  {label:<14} {value}")
    return 1 if st.finished and not st.succeeded else 0


def cmd_rayjob_logs(args: argparse.Namespace) -> int:
    try:
        return follow_rayjob_logs(args.name, namespace=args.namespace, follow=args.follow,
                                  kubectl=shlex.split(args.kubectl), run=subprocess.run)
    except OSError as e:
        raise CliError(str(e)) from e


def cmd_reproduce(args: argparse.Namespace) -> int:
    store = LocalArtifactStore(args.artifacts)
    try:
        spec = ExperimentSpec.model_validate_json(store.get(args.run_id, SPEC))
    except ArtifactError as e:
        raise CliError(str(e)) from e
    _discover(args.experiments)
    data = _data_provider(spec, args.data_root)
    tracker = _tracker(spec, args.tracker, args.mlflow_uri)

    try:
        report = reproduce_run(
            store, args.run_id, data=data, tracker=tracker, scratch=InMemoryArtifactStore(),
            device=args.device, atol=args.atol, probe=_probe(),
        )
    except (ReproduceError, ArtifactError, TrackerError, DataError, RegistryError, RunnerError) as e:
        raise CliError(str(e)) from e
    except Exception as e:  # the replay itself blew up: environment, not evidence
        print(f"replay FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(spec, report)
    return 0 if report.ok else 1


def _print_report(spec: ExperimentSpec, r: ReproduceReport) -> None:
    print(f"{spec.meta.name}: {'REPRODUCED' if r.ok else 'NOT REPRODUCED'}")
    print(f"  experiment_id  {r.experiment_id}")
    print(f"  run_id         {r.run_id}")
    for check, passed in r.checks.items():
        print(f"  {check:<14} {'ok' if passed else 'MISMATCH'}")
    for m in r.mismatches:
        print(f"    {m.check}: {m.detail}")
    if r.environment:
        heading = "environment differences (likely cause):" if not r.ok else "environment differences (did not matter):"
        _print_changes(f"  {heading}", r.environment)


def cmd_report(args: argparse.Namespace) -> int:
    try:
        md = render_report(load_bundle(LocalArtifactStore(args.artifacts), args.run_id))
    except ArtifactError as e:
        raise CliError(str(e)) from e
    if args.out:
        Path(args.out).write_text(md)
    else:
        sys.stdout.write(md)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    store_a = LocalArtifactStore(args.artifacts)
    store_b = LocalArtifactStore(args.artifacts_b) if args.artifacts_b else store_a
    try:
        d = diff_runs(store_a, args.run_a, store_b, args.run_b)
    except ArtifactError as e:
        raise CliError(str(e)) from e
    if args.json:
        print(json.dumps(d.to_dict(), indent=2))
    else:
        _print_diff(d)
    return 0 if d.same_experiment else 1


def _fmt(v: Any) -> str:
    if v is ABSENT:
        return "(absent)"
    if isinstance(v, float):
        return f"{v:.6g}"
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True)


def _print_changes(title: str, changes: list[Change]) -> None:
    if not changes:
        return
    print(title)
    width = max(len(c.path) for c in changes)
    for c in changes:
        print(f"  {c.path:<{width}}  {_fmt(c.a)} -> {_fmt(c.b)}")


def _print_diff(d: RunDiff) -> None:
    def side(run: str, attempt: int | None, status: str) -> str:
        return f"{run[:12]} (attempt {attempt if attempt is not None else '?'}, {status})"

    print(f"a: {side(d.run_a, d.attempt_a, d.status_a)}")
    print(f"b: {side(d.run_b, d.attempt_b, d.status_b)}")
    if d.same_experiment:
        print(f"same experiment: fingerprints identical ({d.experiment_id_a[:12]})")
    else:
        print(f"different experiments: {len(d.experiment)} fingerprint change(s)")
    _print_changes("experiment (in the ExperimentID):", d.experiment)
    _print_changes("run (RunID only):", d.run)
    _print_changes("labels (in no id):", d.labels)
    _print_changes("environment (what each run ran on):", d.environment)
    if d.metrics:
        print("final metrics:")
        width = max(len(m.name) for m in d.metrics)
        for m in d.metrics:
            delta = f"  ({m.delta:+.4g})" if m.delta is not None and not math.isnan(m.delta) else ""
            print(f"  {m.name:<{width}}  {_fmt(m.a)} -> {_fmt(m.b)}{delta}")


def cmd_list(args: argparse.Namespace) -> int:
    if not Path(args.artifacts).is_dir():
        raise CliError(f"no such directory: {args.artifacts}")
    groups = group_by_experiment(list_runs(LocalArtifactStore(args.artifacts), experiment=args.experiment))
    if args.json:
        print(json.dumps([g.to_dict() for g in groups], indent=2))
    elif not groups:
        print("no runs" + (f" for experiment {args.experiment}" if args.experiment else ""))
    else:
        for i, g in enumerate(groups):
            if i:
                print()
            _print_group(g)
    return 0


def _print_group(g: ExperimentGroup) -> None:
    if g.metrics_agree is None:
        verdict = f"{g.finished} finished"
    elif g.metrics_agree:
        verdict = f"{g.finished} finished, final metrics identical"
    else:
        ranges = ", ".join(f"{k} {_fmt(lo)}..{_fmt(hi)}" for k, (lo, hi) in g.spread.items())
        verdict = f"{g.finished} finished, final metrics DIFFER: {ranges}"
    print(f"{g.name}  {g.experiment_id[:12]}  ({len(g.runs)} run(s); {verdict})")
    for r in g.runs:
        attempt = f"a{r.attempt}" if r.attempt is not None else "a?"
        epochs = f"{r.epochs_done if r.epochs_done is not None else '?'}/{r.epochs_total or '?'} ep"
        notes = []
        if r.resumes:
            notes.append(f"resumed x{r.resumes}")
        if r.resumable:
            notes.append(f"resumable: mlops resume {r.run_id[:12]}...")
        if r.error:
            notes.append(r.error)
        metrics = "  ".join(f"{k}={_fmt(v)}" for k, v in r.final_metrics.items())
        tail = "  ".join(x for x in (metrics, "; ".join(notes)) if x)
        print(f"  {r.run_id[:12]}  {attempt:<3}  {r.status:<11}  {epochs:<8}  {tail}".rstrip())


def cmd_snapshot(args: argparse.Namespace) -> int:
    spec = _load(args.experiment)
    _discover(args.experiments)
    data = _data_provider(spec, args.data_root)
    try:
        served = data.snapshot(spec.dataset)
    except DataError as e:
        raise CliError(str(e)) from e
    if served is None:
        raise CliError(f"the provider for {spec.dataset.name!r} does not hash its data; nothing to pin")
    print(served)
    return 0


# -- entry point ---------------------------------------------------------------


def _add_data_root(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", metavar="DIR", default=os.environ.get("MLOPS_DATA_ROOT"),
                   help="where dataset files live on this machine, e.g. DIR/Task09_Spleen (default: $MLOPS_DATA_ROOT)")


def _add_keep_checkpoint(p: argparse.ArgumentParser) -> None:
    p.add_argument("--keep-checkpoint", action="store_true",
                   help="keep checkpoint.pt (and Ray's checkpoint storage) after the run FINISHED; "
                        "by default they are deleted, model.pt holds the final weights")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlops", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    id_ = sub.add_parser("id", help="print the ExperimentID of a spec without running it")
    id_.add_argument("experiment", help="path to experiment.yaml")
    id_.set_defaults(func=cmd_id)

    run = sub.add_parser("run", help="train one run of a spec")
    run.add_argument("experiment", help="path to experiment.yaml")
    run.add_argument("--artifacts", metavar="DIR", help="write the run bundle under DIR/<run_id>/")
    _add_data_root(run)
    run.add_argument("--tracker", choices=["mlflow", "inmemory"], help="override tracking.backend")
    run.add_argument("--mlflow-uri", metavar="URI", help="override tracking.uri (e.g. sqlite:///mlflow.db)")
    run.add_argument("--launcher", choices=["local", "ray"], default="local",
                     help="local: this process (workers must be 1); ray: Ray Train, one process per worker")
    run.add_argument("--ray-address", metavar="ADDR", help="Ray cluster address (default: RAY_ADDRESS or a local cluster)")
    run.add_argument("--ray-storage", metavar="DIR", help="Ray Train storage_path for run state/checkpoints")
    run.add_argument("--device", help="torch device override, e.g. cpu")
    run.add_argument("--attempt", type=int, default=1, help="retry counter; part of RunID")
    run.add_argument("--checkpoint-every", type=int, metavar="N",
                     help="store the training state as checkpoint.pt every N epochs (needs --artifacts); "
                          "makes the run resumable with `mlops resume`")
    _add_keep_checkpoint(run)
    run.add_argument("--keep-best", metavar="METRIC:min|max",
                     help="also keep the best epoch's weights as best.pt, e.g. val/dice/mean:max "
                          "(not in any id; a resume continues with the same criterion)")
    run.add_argument("--strict-env", action="store_true",
                     help="refuse to run unless the code and image verifiably match code.commit / image_digest")
    run.add_argument("--experiments", default="experiments", help="package to discover components from")
    run.add_argument("--json", action="store_true", help="machine-readable summary on stdout")
    run.set_defaults(func=cmd_run)

    rs = sub.add_parser("resume", help="continue an interrupted run from its checkpoint.pt (same run_id)")
    rs.add_argument("run_id", help="run to continue; its checkpoint must be under --artifacts")
    rs.add_argument("--artifacts", metavar="DIR", required=True, help="artifact root holding DIR/<run_id>/")
    _add_data_root(rs)
    rs.add_argument("--tracker", choices=["mlflow", "inmemory"], help="override tracking.backend")
    rs.add_argument("--mlflow-uri", metavar="URI", help="override tracking.uri")
    rs.add_argument("--device", help="torch device override, e.g. cpu")
    rs.add_argument("--ray-address", metavar="ADDR",
                    help="Ray cluster for a run launched on Ray (storage path comes from its launch.json)")
    rs.add_argument("--checkpoint-every", type=int, metavar="N",
                    help="checkpoint interval from here on (default: the one the run was started with)")
    _add_keep_checkpoint(rs)
    rs.add_argument("--experiments", default="experiments", help="package to discover components from")
    rs.add_argument("--json", action="store_true", help="machine-readable summary on stdout")
    rs.set_defaults(func=cmd_resume)

    rj = sub.add_parser("rayjob", help="render (or submit) a KubeRay RayJob that runs the spec on Kubernetes")
    rj.add_argument("experiment", help="path to experiment.yaml")
    rj.add_argument("--namespace", default="default")
    rj.add_argument("--attempt", type=int, default=1, help="retry counter; part of RunID and the job name")
    rj.add_argument("--artifacts-pvc", metavar="PVC", help="PersistentVolumeClaim to mount for the run bundle")
    rj.add_argument("--artifacts-path", default="/artifacts", help="mount path of that PVC inside the pods")
    rj.add_argument("--data-pvc", metavar="PVC",
                    help="PersistentVolumeClaim holding the dataset files (e.g. Task09_Spleen/), mounted read-only")
    rj.add_argument("--data-path", default="/data", help="mount path of that PVC; passed as --data-root")
    rj.add_argument("--mlflow-uri", metavar="URI", help="override tracking.uri for the in-cluster run")
    rj.add_argument("--checkpoint-every", type=int, metavar="N",
                    help="checkpoint every N epochs so the job can be resumed with --resume "
                         "(needs --artifacts-pvc and --ray-storage-pvc)")
    rj.add_argument("--ray-storage-pvc", metavar="PVC",
                    help="PersistentVolumeClaim for Ray Train's checkpoints, writable on every pod "
                         "(ReadWriteMany across nodes); passed as --ray-storage")
    rj.add_argument("--ray-storage-path", default="/ray-storage", help="mount path of that PVC inside the pods")
    rj.add_argument("--keep-best", metavar="METRIC:min|max", help="passed to the in-cluster `mlops run`")
    rj.add_argument("--resume", action="store_true",
                    help="render a RayJob (<name>-resume-<generated>) that runs `mlops resume` for this spec + --attempt "
                         "from the checkpoints on the PVCs; pass the same PVCs and paths as the first job")
    rj.add_argument("--submit", action="store_true", help="kubectl apply the manifest instead of printing it")
    rj.add_argument("--kubectl", default="kubectl", help='kubectl command, e.g. "kubectl --context prod"')
    rj.set_defaults(func=cmd_rayjob)

    rs_ = sub.add_parser("rayjob-status", help="state of a submitted RayJob (kubectl get); "
                                              "exit 0 succeeded or still going, 1 failed/stopped")
    rs_.add_argument("name", help="RayJob name, as printed by `mlops rayjob --submit`")
    rs_.add_argument("--namespace", default="default")
    rs_.add_argument("--kubectl", default="kubectl", help='kubectl command, e.g. "kubectl --context prod"')
    rs_.add_argument("--json", action="store_true", help="machine-readable status on stdout")
    rs_.set_defaults(func=cmd_rayjob_status)

    rl = sub.add_parser("rayjob-logs", help="driver log of a submitted RayJob (kubectl logs job/NAME)")
    rl.add_argument("name", help="RayJob name, as printed by `mlops rayjob --submit`")
    rl.add_argument("--namespace", default="default")
    rl.add_argument("--kubectl", default="kubectl", help='kubectl command, e.g. "kubectl --context prod"')
    rl.add_argument("-f", "--follow", action="store_true", help="keep streaming until the job ends")
    rl.set_defaults(func=cmd_rayjob_logs)

    rp = sub.add_parser("reproduce", help="re-run a bundle and check run_id, history and weights match")
    rp.add_argument("run_id", help="run to verify; its bundle must be under --artifacts")
    rp.add_argument("--artifacts", metavar="DIR", required=True, help="artifact root holding DIR/<run_id>/")
    _add_data_root(rp)
    rp.add_argument("--tracker", choices=["mlflow", "inmemory"], help="where the replay is recorded")
    rp.add_argument("--mlflow-uri", metavar="URI", help="override tracking.uri")
    rp.add_argument("--device", help="torch device override, e.g. cpu")
    rp.add_argument("--atol", type=float, default=0.0,
                    help="weight tolerance; 0 (default) = bit-exact, >0 = allclose for cross-launcher checks")
    rp.add_argument("--experiments", default="experiments", help="package to discover components from")
    rp.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    rp.set_defaults(func=cmd_reproduce)

    rr = sub.add_parser("report", help="render the human-readable Markdown report of a run from its bundle")
    rr.add_argument("run_id", help="run to report on; its bundle must be under --artifacts")
    rr.add_argument("--artifacts", metavar="DIR", required=True, help="artifact root holding DIR/<run_id>/")
    rr.add_argument("--out", metavar="FILE", help="write the report to FILE instead of stdout")
    rr.set_defaults(func=cmd_report)

    df = sub.add_parser("diff", help="explain why two runs differ (fingerprint, resources, labels, env, metrics); "
                                     "exit 0 same experiment, 1 different")
    df.add_argument("run_a")
    df.add_argument("run_b")
    df.add_argument("--artifacts", metavar="DIR", required=True, help="artifact root holding run_a (and run_b)")
    df.add_argument("--artifacts-b", metavar="DIR", help="artifact root holding run_b, if it is elsewhere")
    df.add_argument("--json", action="store_true", help="machine-readable diff on stdout")
    df.set_defaults(func=cmd_diff)

    ls = sub.add_parser("list", help="runs under an artifact root, grouped by experiment, and whether "
                                     "each experiment's finished attempts agree")
    ls.add_argument("--artifacts", metavar="DIR", required=True, help="artifact root to scan")
    ls.add_argument("--experiment", metavar="ID", help="only this ExperimentID (a prefix is enough)")
    ls.add_argument("--json", action="store_true", help="machine-readable listing on stdout")
    ls.set_defaults(func=cmd_list)

    sn = sub.add_parser("snapshot", help="print the content hash of the data a spec would load, to pin as dataset.snapshot")
    sn.add_argument("experiment", help="path to experiment.yaml")
    _add_data_root(sn)
    sn.add_argument("--experiments", default="experiments", help="package to discover components from")
    sn.set_defaults(func=cmd_snapshot)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
