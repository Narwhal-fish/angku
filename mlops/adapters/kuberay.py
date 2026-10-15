"""KubeRay: turn an ExperimentSpec into a ``RayJob`` custom resource.

The job's entrypoint is ``mlops run ... --launcher ray`` inside the pinned
training image, so the same code path as a laptop run executes on the
cluster; ``RayLauncher`` then spreads ``resources.workers`` ranks over the
worker pods. The spec itself rides along in a ConfigMap as JSON (a YAML
subset, so ``mlops run`` loads it unchanged). Everything that matters for
reproducibility is already pinned in the spec: the image by digest, the code
by commit (baked into that image), the data by snapshot.

Rendering is pure (dicts in, dicts out) and needs no cluster; ``submit_rayjob``
shells out to ``kubectl apply`` with an injectable runner so it is testable. Following a
job afterwards (``rayjob_status``, ``follow_rayjob_logs``) is likewise kubectl's own
``get`` / ``logs --follow``: no polling loop or retry logic of ours.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from mlops.domain.errors import UnpinnedInputError
from mlops.domain.experiment import ExperimentSpec
from mlops.domain.identity import compute_experiment_id, compute_run_id

SPEC_MOUNT = "/etc/mlops"
SPEC_FILE = "experiment.json"
RAY_VERSION = "2.58.0"  # must match the ray pinned in the training image


class KubeRayError(Exception):
    """kubectl failed or refused the manifest."""


def image_ref(image: str, digest: str) -> str:
    """``repo:tag`` -> ``repo@sha256:...``. A tag is mutable; the digest is the pin."""
    head, _, last = image.rpartition("/")
    last = last.split(":", 1)[0]  # only the final path segment can carry a tag
    repo = f"{head}/{last}" if head else last
    return f"{repo}@{digest}"


def _label(value: str) -> str:
    # Kubernetes label values: <= 63 chars of [A-Za-z0-9_.-], start/end alphanumeric.
    v = re.sub(r"[^A-Za-z0-9_.-]", "-", value)[:63].strip("-_.")
    return v or "unnamed"


def render_rayjob(
    spec: ExperimentSpec,
    *,
    namespace: str = "default",
    attempt: int = 1,
    artifacts_pvc: str | None = None,
    artifacts_path: str = "/artifacts",
    mlflow_uri: str | None = None,
    ttl_seconds: int = 3600,
    data_pvc: str | None = None,
    data_path: str = "/data",
    checkpoint_every: int | None = None,
    keep_best: str | None = None,
    ray_storage_pvc: str | None = None,
    ray_storage_path: str = "/ray-storage",
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Return ``[ConfigMap, RayJob]`` as plain JSON-able dicts (``[RayJob]`` with ``resume``).

    ``data_pvc`` mounts the dataset read-only at ``data_path`` on the head *and* every worker
    (the DataProvider is built on the driver and pickled to the ranks with its root) and passes
    ``--data-root data_path`` to ``mlops run``.

    ``checkpoint_every`` makes the job resumable after ``shutdownAfterJobFinishes`` has deleted
    its cluster: Ray Train's checkpoints go to ``ray_storage_pvc`` (writable, mounted on every
    pod, so ReadWriteMany when pods span nodes) and ``spec.json`` + ``launch.json`` to
    ``artifacts_pvc``; both are required. ``resume=True`` renders a RayJob with
    ``generateName: <name>-resume-`` (Kubernetes adds a suffix, so earlier finished jobs of the run
    never have to be deleted; submit with ``kubectl create``) whose entrypoint is ``mlops resume RUN_ID``: spec, attempt, interval and storage path come
    from those PVCs (so mount them at the same paths as the first job), no ConfigMap needed.
    """
    if not spec.is_pinned():
        raise UnpinnedInputError(spec.unpinned_inputs())
    if keep_best and resume:
        raise ValueError("keep_best is fixed when a run starts; a resume continues with the run's own")
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
    if checkpoint_every or resume:
        # Without both claims the state dies with the cluster: refuse rather than look resumable.
        if not artifacts_pvc:
            raise ValueError("a resumable RayJob needs artifacts_pvc (spec.json + launch.json live there)")
        if not ray_storage_pvc:
            raise ValueError("a resumable RayJob needs ray_storage_pvc (Ray Train's checkpoints live there)")
    experiment_id = compute_experiment_id(spec)
    run_id = compute_run_id(experiment_id, spec.resources, attempt)
    name = rayjob_name(run_id, attempt)
    res = spec.resources
    image = image_ref(spec.environment.image, spec.environment.image_digest)

    if resume:
        # Not `--attempt`/`--ray-storage`: launch.json on the artifacts PVC records both.
        entrypoint = ["mlops", "resume", run_id, "--artifacts", artifacts_path]
    else:
        entrypoint = [
            "mlops", "run", f"{SPEC_MOUNT}/{SPEC_FILE}",
            "--launcher", "ray",
            "--attempt", str(attempt),
            "--artifacts", artifacts_path,
        ]
        if ray_storage_pvc:
            entrypoint += ["--ray-storage", ray_storage_path]
        if keep_best:
            entrypoint += ["--keep-best", keep_best]
    if checkpoint_every:
        entrypoint += ["--checkpoint-every", str(checkpoint_every)]
    if mlflow_uri:
        entrypoint += ["--mlflow-uri", mlflow_uri]
    if data_pvc:
        entrypoint += ["--data-root", data_path]

    volumes: list[dict[str, Any]] = []
    mounts: list[dict[str, Any]] = []
    if not resume:
        volumes.append({"name": "experiment", "configMap": {"name": name}})
        mounts.append({"name": "experiment", "mountPath": SPEC_MOUNT, "readOnly": True})
    if artifacts_pvc:
        volumes.append({"name": "artifacts", "persistentVolumeClaim": {"claimName": artifacts_pvc}})
        mounts.append({"name": "artifacts", "mountPath": artifacts_path})
    if ray_storage_pvc:
        volumes.append({"name": "ray-storage", "persistentVolumeClaim": {"claimName": ray_storage_pvc}})
        mounts.append({"name": "ray-storage", "mountPath": ray_storage_path})
    if data_pvc:
        # Read-only twice over: the claim and the mount. A run must never alter its own inputs.
        volumes.append({"name": "data", "persistentVolumeClaim": {"claimName": data_pvc, "readOnly": True}})
        mounts.append({"name": "data", "mountPath": data_path, "readOnly": True})

    def container(cname: str, limits: dict[str, str]) -> dict[str, Any]:
        return {
            "name": cname,
            "image": image,
            "resources": {"limits": dict(limits), "requests": dict(limits)},
            "volumeMounts": list(mounts),
            # Kubernetes pulls by digest, so the runtime enforces it; this lets the run record it.
            "env": [{"name": "MLOPS_IMAGE_DIGEST", "value": spec.environment.image_digest}],
        }

    worker_limits = {"cpu": str(res.cpu), "memory": res.memory}
    if res.gpu > 0:
        worker_limits["nvidia.com/gpu"] = str(res.gpu)

    metadata = {
        "name": name,
        "namespace": namespace,
        "labels": {
            "app.kubernetes.io/managed-by": "mlops",
            "mlops.io/experiment": _label(spec.meta.name),
            # Selectable (annotations are not): every job of a run, first and resumes, by `-l`.
            RUN_LABEL: run_label(run_id),
        },
        "annotations": {"mlops.io/experiment-id": experiment_id, "mlops.io/run-id": run_id},
    }
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": dict(metadata),
        "data": {SPEC_FILE: spec.model_dump_json(indent=2)},
    }
    job_metadata = dict(metadata)
    if resume:
        del job_metadata["name"]
        job_metadata["generateName"] = f"{name}-resume-"
    rayjob = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": job_metadata,
        "spec": {
            "entrypoint": " ".join(entrypoint),
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": ttl_seconds,
            "rayClusterSpec": {
                "rayVersion": RAY_VERSION,
                "headGroupSpec": {
                    # The head only hosts the driver; every training rank lands on a worker pod.
                    "rayStartParams": {"num-cpus": "0", "dashboard-host": "0.0.0.0"},
                    "template": {
                        "spec": {
                            "containers": [container("ray-head", {"cpu": "2", "memory": "4Gi"})],
                            "volumes": list(volumes),
                        }
                    },
                },
                "workerGroupSpecs": [
                    {
                        "groupName": "train",
                        "replicas": res.workers,
                        "minReplicas": res.workers,
                        "maxReplicas": res.workers,
                        "rayStartParams": {},
                        "template": {
                            "spec": {
                                "containers": [container("ray-worker", worker_limits)],
                                "volumes": list(volumes),
                            }
                        },
                    }
                ],
            },
        },
    }
    return [rayjob] if resume else [configmap, rayjob]


def rayjob_name(run_id: str, attempt: int) -> str:
    return f"mlops-{run_id[:12]}-a{attempt}"


RUN_LABEL = "mlops.io/run"


def run_label(run_id: str) -> str:
    # Label values are limited to 63 chars; 32 hex chars of a sha256 are still unique enough.
    return run_id[:32]


Runner = Callable[..., Any]  # subprocess.run-compatible


def submit_rayjob(
    items: Sequence[dict[str, Any]],
    *,
    kubectl: Sequence[str] = ("kubectl",),
    run: Runner = subprocess.run,
) -> str:
    """``kubectl apply`` the rendered objects; returns the RayJob name.

    A resume job has ``generateName``, which ``apply`` cannot handle: it goes through
    ``kubectl create -o name`` and the name Kubernetes generated is read back from its output.
    """
    (job,) = [i for i in items if i["kind"] == "RayJob"]
    namespace = job["metadata"]["namespace"]
    generated = "name" not in job["metadata"]
    body = json.dumps({"apiVersion": "v1", "kind": "List", "items": list(items)})
    verb = ["create", "-n", namespace, "-o", "name"] if generated else ["apply", "-n", namespace]
    cmd = [*kubectl, *verb, "-f", "-"]
    r = run(cmd, input=body, capture_output=True, text=True)
    if r.returncode != 0:
        raise KubeRayError(f"{' '.join(cmd)} failed ({r.returncode}): {r.stderr.strip()}")
    if not generated:
        return job["metadata"]["name"]
    for line in r.stdout.splitlines():  # e.g. rayjob.ray.io/mlops-...-resume-x8k2p
        kind, _, created = line.strip().partition("/")
        if kind.split(".")[0] == "rayjob":
            return created
    raise KubeRayError(f"{' '.join(cmd)}: no RayJob name in its output: {r.stdout.strip()!r}")


# -- following a submitted job: kubectl's own status and log machinery, nothing rebuilt ----------

_TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}


@dataclass(frozen=True)
class RayJobStatus:
    """What the KubeRay operator reports in ``.status`` (empty strings until it has reported)."""

    name: str
    job_status: str  # Ray job: PENDING / RUNNING / SUCCEEDED / FAILED / STOPPED
    deployment_status: str  # the RayJob CR: Initializing / Running / Complete / Failed / ...
    message: str = ""
    dashboard_url: str = ""
    start_time: str = ""
    end_time: str = ""
    run_id: str | None = None  # from our annotations: which mlops run this job is
    experiment_id: str | None = None

    @property
    def finished(self) -> bool:
        return self.job_status in _TERMINAL

    @property
    def succeeded(self) -> bool:
        return self.job_status == "SUCCEEDED"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "finished": self.finished, "succeeded": self.succeeded}


def rayjob_status(
    name: str, *, namespace: str = "default", kubectl: Sequence[str] = ("kubectl",), run: Runner = subprocess.run
) -> RayJobStatus:
    """``kubectl get rayjob NAME -o json``, parsed. Raise ``KubeRayError`` if kubectl fails."""
    cmd = [*kubectl, "get", "rayjob", name, "-n", namespace, "-o", "json"]
    r = run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise KubeRayError(f"{' '.join(cmd)} failed ({r.returncode}): {r.stderr.strip()}")
    return _parse_status(name, json.loads(r.stdout))


def _parse_status(name: str, obj: dict[str, Any]) -> RayJobStatus:
    st = obj.get("status") or {}
    ann = (obj.get("metadata") or {}).get("annotations") or {}
    return RayJobStatus(
        name=name,
        job_status=st.get("jobStatus", ""),
        deployment_status=st.get("jobDeploymentStatus", ""),
        message=st.get("message", ""),
        dashboard_url=st.get("dashboardURL", ""),
        start_time=st.get("startTime", ""),
        end_time=st.get("endTime", ""),
        run_id=ann.get("mlops.io/run-id"),
        experiment_id=ann.get("mlops.io/experiment-id"),
    )


def active_rayjobs(
    run_id: str, *, namespace: str = "default", kubectl: Sequence[str] = ("kubectl",), run: Runner = subprocess.run
) -> list[RayJobStatus]:
    """RayJobs of this run (by ``mlops.io/run`` label) that have not finished: pending, running,
    or just created with no status yet. Two of them would write the same Ray checkpoints.
    Jobs rendered before the label existed are not found."""
    cmd = [*kubectl, "get", "rayjob", "-n", namespace, "-l", f"{RUN_LABEL}={run_label(run_id)}", "-o", "json"]
    r = run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise KubeRayError(f"{' '.join(cmd)} failed ({r.returncode}): {r.stderr.strip()}")
    jobs = [_parse_status(obj["metadata"]["name"], obj) for obj in json.loads(r.stdout).get("items", [])]
    return [j for j in jobs if not j.finished]


def follow_rayjob_logs(
    name: str,
    *,
    namespace: str = "default",
    follow: bool = False,
    kubectl: Sequence[str] = ("kubectl",),
    run: Runner = subprocess.run,
) -> int:
    """Stream the driver log to this terminal; returns kubectl's exit code.

    KubeRay submits the entrypoint through a Kubernetes Job named like the RayJob, whose pod
    runs ``ray job submit`` and tails the driver output, so ``kubectl logs job/NAME`` is the
    run's stdout/stderr (``mlops run`` summary and warnings included).
    """
    cmd = [*kubectl, "logs", "-n", namespace, f"job/{name}"]
    if follow:
        cmd.append("--follow")
    return run(cmd).returncode
