"""``mlops`` command line. The composition root: the one place that picks
concrete adapters (tracker, data provider, artifact store) and hands them to
``run_experiment``. Nothing under ``mlops/application`` knows this exists.

    mlops id     experiment.yaml
    mlops run    experiment.yaml [--artifacts DIR] [--data-root DIR] [--tracker mlflow|inmemory]
                                 [--mlflow-uri URI] [--launcher local|ray] [--ray-address ADDR]
                                 [--device cpu] [--attempt N] [--strict-env] [--json]
    mlops rayjob experiment.yaml [--namespace NS] [--artifacts-pvc PVC] [--attempt N]
                                 [--submit [--kubectl "kubectl --context X"]]
    mlops reproduce RUN_ID --artifacts DIR [--data-root DIR] [--tracker mlflow|inmemory] [--mlflow-uri URI]
                                 [--device cpu] [--atol X] [--json]
    mlops report RUN_ID --artifacts DIR [--out FILE]

Exit codes: 0 ok, 1 training failed / bundle not reproduced, 2 bad input (yaml, pins, wiring).

Environment: every run records what it actually ran on (code vs code.commit, image digest,
library versions, GPUs) in manifest.json and warns when that does not match the pins;
``--strict-env`` refuses to start instead.

Datasets: built-in ``synthetic-blobs``, plus whatever ``experiments/`` registers in
``mlops.registry.DATASETS``; those read files under ``--data-root`` (default: $MLOPS_DATA_ROOT).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlops.adapters.inmemory_artifacts import InMemoryArtifactStore
from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.kuberay import KubeRayError, render_rayjob, submit_rayjob
from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.adapters.local_environment import LocalEnvironmentProbe
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import RunnerError
from mlops.application.bundle import SPEC, load_bundle
from mlops.application.launch import Launcher, LocalLauncher
from mlops.application.report import render_report, write_report
from mlops.application.reproduce import ReproduceError, ReproduceReport, reproduce_run
from mlops.config import ConfigError, load_experiment_yaml
from mlops.domain import ExperimentSpec, UnpinnedInputError, compute_experiment_id, compute_run_id
from mlops.domain.environment import EnvironmentRecord, record_environment
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


def _unverified(record: EnvironmentRecord) -> list[str]:
    return [f"{name}: {c.status}: {c.detail}" for name, c in record.checks.items() if c.status != "ok"]


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
    _discover(args.experiments)
    data = _data_provider(spec, args.data_root)
    tracker = _tracker(spec, args.tracker, args.mlflow_uri)
    launcher = _launcher(spec, args)
    probe = _probe()
    if args.strict_env:
        # Pre-flight on this machine; the record in the manifest is captured where training runs.
        problems = _unverified(record_environment(spec, probe.capture(spec)))
        if problems:
            raise CliError("--strict-env: environment does not match the pins:\n  " + "\n  ".join(problems))
    store = LocalArtifactStore(args.artifacts) if args.artifacts else None

    try:
        result = launcher.launch(
            spec, data=data, tracker=tracker, artifacts=store, attempt=args.attempt, device=args.device,
            probe=probe,
        )
    except (TrackerError, DataError, RegistryError, RunnerError) as e:
        raise CliError(str(e)) from e
    except Exception as e:  # training itself blew up; the FAILED bundle (if any) is already written
        if store is not None:
            run_id = compute_run_id(compute_experiment_id(spec), spec.resources, args.attempt)
            try:
                write_report(store, run_id)
            except ArtifactError:  # failed before the run started: no bundle to report on
                pass
        print(f"run FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if store is not None:
        write_report(store, result.run_id)
    if result.environment is not None:
        for problem in _unverified(result.environment):
            print(f"warning: environment not verified: {problem}", file=sys.stderr)

    summary: dict[str, Any] = {
        "status": "FINISHED",
        "experiment_id": result.experiment_id,
        "run_id": result.run_id,
        "attempt": args.attempt,
        "device": result.device,
        "launcher": args.launcher,
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
    for epoch, metrics in enumerate(s["history"], start=1):
        cells = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  epoch {epoch:>3}  {cells}")
    if s["artifacts"]:
        print(f"  artifacts      {s['artifacts']}")


def cmd_rayjob(args: argparse.Namespace) -> int:
    spec = _load(args.experiment)
    try:
        items = render_rayjob(
            spec,
            namespace=args.namespace,
            attempt=args.attempt,
            artifacts_pvc=args.artifacts_pvc,
            artifacts_path=args.artifacts_path,
            mlflow_uri=args.mlflow_uri or spec.tracking.uri or None,
        )
    except UnpinnedInputError as e:
        raise CliError(str(e)) from e
    if not args.submit:
        print(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=2))
        return 0
    try:
        name = submit_rayjob(items, kubectl=shlex.split(args.kubectl), run=subprocess.run)
    except (KubeRayError, OSError) as e:
        raise CliError(str(e)) from e
    print(f"submitted RayJob {name} in namespace {args.namespace}")
    return 0


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
            device=args.device, atol=args.atol,
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


# -- entry point ---------------------------------------------------------------


def _add_data_root(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", metavar="DIR", default=os.environ.get("MLOPS_DATA_ROOT"),
                   help="where dataset files live on this machine, e.g. DIR/Task09_Spleen (default: $MLOPS_DATA_ROOT)")


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
    run.add_argument("--strict-env", action="store_true",
                     help="refuse to run unless the code and image verifiably match code.commit / image_digest")
    run.add_argument("--experiments", default="experiments", help="package to discover components from")
    run.add_argument("--json", action="store_true", help="machine-readable summary on stdout")
    run.set_defaults(func=cmd_run)

    rj = sub.add_parser("rayjob", help="render (or submit) a KubeRay RayJob that runs the spec on Kubernetes")
    rj.add_argument("experiment", help="path to experiment.yaml")
    rj.add_argument("--namespace", default="default")
    rj.add_argument("--attempt", type=int, default=1, help="retry counter; part of RunID and the job name")
    rj.add_argument("--artifacts-pvc", metavar="PVC", help="PersistentVolumeClaim to mount for the run bundle")
    rj.add_argument("--artifacts-path", default="/artifacts", help="mount path of that PVC inside the pods")
    rj.add_argument("--mlflow-uri", metavar="URI", help="override tracking.uri for the in-cluster run")
    rj.add_argument("--submit", action="store_true", help="kubectl apply the manifest instead of printing it")
    rj.add_argument("--kubectl", default="kubectl", help='kubectl command, e.g. "kubectl --context prod"')
    rj.set_defaults(func=cmd_rayjob)

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
