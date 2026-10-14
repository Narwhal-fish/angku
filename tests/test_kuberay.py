"""KubeRay: render a RayJob (+ ConfigMap carrying the spec) from an ExperimentSpec, submit via kubectl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlops.adapters.kuberay import KubeRayError, image_ref, render_rayjob, submit_rayjob
from mlops.config import load_experiment_dict, load_experiment_yaml
from mlops.domain import ExperimentSpec, UnpinnedInputError, compute_experiment_id, compute_run_id

CONFIGS = Path(__file__).parent.parent / "experiments/configs"
DIGEST = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


@pytest.fixture
def spec():
    return load_experiment_yaml(CONFIGS / "swinunetr_baseline.yaml")


def _by_kind(items, kind):
    (m,) = [i for i in items if i["kind"] == kind]
    return m


def _containers(rayjob):
    cluster = rayjob["spec"]["rayClusterSpec"]
    head = cluster["headGroupSpec"]["template"]["spec"]["containers"][0]
    (group,) = cluster["workerGroupSpecs"]
    worker = group["template"]["spec"]["containers"][0]
    return head, group, worker


# --- image pinning ---


@pytest.mark.parametrize(
    "image",
    ["ghcr.io/example/mlops-train:0.1.0", "ghcr.io/example/mlops-train", "localhost:5000/train:v2", "train:latest"],
)
def test_image_ref_replaces_tag_with_digest(image):
    ref = image_ref(image, DIGEST)
    assert ref.endswith("@" + DIGEST)
    assert ":0.1.0" not in ref and ":v2" not in ref and ":latest" not in ref
    if image.startswith("localhost:5000"):
        assert ref.startswith("localhost:5000/train@")


# --- rendering ---


def test_render_produces_configmap_and_rayjob(spec):
    items = render_rayjob(spec)
    assert [i["kind"] for i in items] == ["ConfigMap", "RayJob"]
    cm, job = items

    eid = compute_experiment_id(spec)
    rid = compute_run_id(eid, spec.resources, attempt=1)
    assert job["apiVersion"] == "ray.io/v1"
    assert job["metadata"]["name"] == f"mlops-{rid[:12]}-a1"
    assert job["metadata"]["namespace"] == "default"
    assert job["metadata"]["annotations"] == {"mlops.io/experiment-id": eid, "mlops.io/run-id": rid}
    assert job["metadata"]["labels"]["mlops.io/experiment"] == "swinunetr-baseline"
    assert cm["metadata"]["name"] == job["metadata"]["name"]

    # the spec rides along verbatim as JSON (a YAML subset, so `mlops run` reads it as-is)
    assert ExperimentSpec.model_validate_json(cm["data"]["experiment.json"]) == spec

    entry = job["spec"]["entrypoint"]
    assert entry.startswith("mlops run /etc/mlops/experiment.json")
    assert "--launcher ray" in entry and "--attempt 1" in entry and "--artifacts /artifacts" in entry


def test_render_pins_image_and_sizes_cluster_from_resources(spec):
    _, job = render_rayjob(spec)
    head, group, worker = _containers(job)
    pinned = image_ref(spec.environment.image, spec.environment.image_digest)
    assert head["image"] == pinned and worker["image"] == pinned
    assert group["replicas"] == group["minReplicas"] == group["maxReplicas"] == spec.resources.workers
    assert worker["resources"]["limits"] == {"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "1"}
    assert worker["resources"]["requests"] == worker["resources"]["limits"]
    # training never lands on the head pod
    assert job["spec"]["rayClusterSpec"]["headGroupSpec"]["rayStartParams"]["num-cpus"] == "0"
    assert "nvidia.com/gpu" not in head["resources"]["limits"]


def test_render_without_gpu_has_no_gpu_limit(spec):
    d = spec.model_dump()
    d["resources"] = {**d["resources"], "gpu": 0, "workers": 2}
    _, job = render_rayjob(load_experiment_dict(d))
    _, group, worker = _containers(job)
    assert group["replicas"] == 2
    assert "nvidia.com/gpu" not in worker["resources"]["limits"]


def test_render_options(spec):
    cm, job = render_rayjob(
        spec, namespace="ml", attempt=2, artifacts_pvc="runs-pvc", artifacts_path="/mnt/runs", mlflow_uri="http://mlflow:5000"
    )
    assert cm["metadata"]["namespace"] == job["metadata"]["namespace"] == "ml"
    assert job["metadata"]["name"].endswith("-a2")
    entry = job["spec"]["entrypoint"]
    assert "--attempt 2" in entry and "--artifacts /mnt/runs" in entry and "--mlflow-uri http://mlflow:5000" in entry
    head, _, worker = _containers(job)
    for pod in (job["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"],
                job["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"]["spec"]):
        names = {v["name"]: v for v in pod["volumes"]}
        assert names["artifacts"]["persistentVolumeClaim"]["claimName"] == "runs-pvc"
        assert names["experiment"]["configMap"]["name"] == cm["metadata"]["name"]
    mounts = {m["name"]: m["mountPath"] for m in head["volumeMounts"]}
    assert mounts == {"artifacts": "/mnt/runs", "experiment": "/etc/mlops"}


def test_render_refuses_unpinned_spec(spec):
    d = spec.model_dump()
    d["environment"] = {**d["environment"], "image_digest": None}
    with pytest.raises(UnpinnedInputError):
        render_rayjob(load_experiment_dict(d))


def test_rendered_manifest_is_plain_json(spec):
    json.dumps(render_rayjob(spec))  # no Path/tuple/pydantic objects leaking


# --- submission ---


class _FakeKubectl:
    def __init__(self, returncode=0, stderr=""):
        self.calls = []
        self.returncode, self.stderr = returncode, stderr

    def __call__(self, cmd, *, input, capture_output, text):
        self.calls.append((cmd, input))

        class R:
            returncode = self.returncode
            stdout = "rayjob.ray.io/x created\n"
            stderr = self.stderr

        return R()


def test_submit_applies_a_list_through_kubectl(spec):
    items = render_rayjob(spec, namespace="ml")
    kubectl = _FakeKubectl()
    name = submit_rayjob(items, run=kubectl)
    assert name == items[1]["metadata"]["name"]
    ((cmd, stdin),) = kubectl.calls
    assert cmd == ["kubectl", "apply", "-n", "ml", "-f", "-"]
    body = json.loads(stdin)
    assert body["kind"] == "List" and [i["kind"] for i in body["items"]] == ["ConfigMap", "RayJob"]


def test_submit_reports_kubectl_failure(spec):
    kubectl = _FakeKubectl(returncode=1, stderr="error: the server doesn't have a resource type RayJob")
    with pytest.raises(KubeRayError, match="resource type RayJob"):
        submit_rayjob(render_rayjob(spec), run=kubectl)


def test_submit_uses_custom_kubectl_and_context(spec):
    kubectl = _FakeKubectl()
    submit_rayjob(render_rayjob(spec), run=kubectl, kubectl=["kubectl", "--context", "gke-prod"])
    ((cmd, _),) = kubectl.calls
    assert cmd[:3] == ["kubectl", "--context", "gke-prod"]


def test_pods_know_which_image_they_run_in(spec):
    # Kubernetes pulls by digest, so the runtime enforces it; the env var lets the run record it.
    _, rayjob = render_rayjob(spec)
    head, _, worker = _containers(rayjob)
    for c in (head, worker):
        env = {e["name"]: e["value"] for e in c.get("env", [])}
        assert env["MLOPS_IMAGE_DIGEST"] == spec.environment.image_digest


# --- following a submitted RayJob: status via `kubectl get`, logs via `kubectl logs` ---


def _rayjob_json(job_status="RUNNING", deployment="Running", message="", **status):
    return json.dumps({
        "kind": "RayJob",
        "metadata": {
            "name": "mlops-abc-a1", "namespace": "ml",
            "annotations": {"mlops.io/run-id": "r" * 64, "mlops.io/experiment-id": "e" * 64},
        },
        "status": {"jobStatus": job_status, "jobDeploymentStatus": deployment, "message": message,
                   "dashboardURL": "head-svc:8265", "startTime": "2026-10-13T10:00:00Z", **status},
    })


class _FakeGet:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.calls = []
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        out = self

        class R:
            returncode, stdout, stderr = out.returncode, out.stdout, out.stderr

        return R()


def test_status_reads_the_rayjob_through_kubectl():
    from mlops.adapters.kuberay import rayjob_status

    kubectl = _FakeGet(_rayjob_json(endTime="2026-10-13T11:00:00Z", job_status="SUCCEEDED", deployment="Complete"))
    s = rayjob_status("mlops-abc-a1", namespace="ml", kubectl=["kubectl", "--context", "x"], run=kubectl)
    ((cmd, kw),) = kubectl.calls
    assert cmd == ["kubectl", "--context", "x", "get", "rayjob", "mlops-abc-a1", "-n", "ml", "-o", "json"]
    assert kw["capture_output"] is True
    assert s.job_status == "SUCCEEDED" and s.deployment_status == "Complete"
    assert s.run_id == "r" * 64 and s.experiment_id == "e" * 64
    assert s.finished and s.succeeded
    assert s.dashboard_url == "head-svc:8265" and s.end_time == "2026-10-13T11:00:00Z"


@pytest.mark.parametrize("job,finished,succeeded", [
    ("PENDING", False, False), ("RUNNING", False, False), ("SUCCEEDED", True, True),
    ("FAILED", True, False), ("STOPPED", True, False), ("", False, False),
])
def test_status_terminal_states(job, finished, succeeded):
    from mlops.adapters.kuberay import rayjob_status

    s = rayjob_status("x", run=_FakeGet(_rayjob_json(job_status=job)))
    assert (s.finished, s.succeeded) == (finished, succeeded)


def test_status_of_a_fresh_job_without_status_block():
    from mlops.adapters.kuberay import rayjob_status

    s = rayjob_status("x", run=_FakeGet(json.dumps({"kind": "RayJob", "metadata": {"name": "x"}})))
    assert s.job_status == "" and not s.finished and s.run_id is None


def test_status_reports_kubectl_failure():
    from mlops.adapters.kuberay import rayjob_status

    with pytest.raises(KubeRayError, match="NotFound"):
        rayjob_status("x", run=_FakeGet(returncode=1, stderr='Error from server (NotFound): rayjobs "x" not found'))


def test_logs_stream_the_submitter_job_without_capturing():
    from mlops.adapters.kuberay import follow_rayjob_logs

    kubectl = _FakeGet()
    assert follow_rayjob_logs("mlops-abc-a1", namespace="ml", follow=True, run=kubectl) == 0
    ((cmd, kw),) = kubectl.calls
    # KubeRay submits through a Kubernetes Job named like the RayJob; its pod tails the driver log.
    assert cmd == ["kubectl", "logs", "-n", "ml", "job/mlops-abc-a1", "--follow"]
    assert not kw.get("capture_output")


def test_logs_without_follow_and_exit_code_passthrough():
    from mlops.adapters.kuberay import follow_rayjob_logs

    kubectl = _FakeGet(returncode=1)
    assert follow_rayjob_logs("x", run=kubectl) == 1
    assert "--follow" not in kubectl.calls[0][0]


# --- data volume: providers that read files get them from a PVC, read-only, at --data-root ---


def _pods(job):
    rc = job["spec"]["rayClusterSpec"]
    return [rc["headGroupSpec"]["template"]["spec"], rc["workerGroupSpecs"][0]["template"]["spec"]]


def test_data_pvc_is_mounted_read_only_on_every_pod(spec):
    _, job = render_rayjob(spec, data_pvc="msd-data")
    # The provider is built on the driver (head) and pickled to the workers with its root:
    # both must see the files at the same path.
    for pod in _pods(job):
        vols = {v["name"]: v for v in pod["volumes"]}
        assert vols["data"]["persistentVolumeClaim"] == {"claimName": "msd-data", "readOnly": True}
        (container,) = pod["containers"]
        mounts = {m["name"]: m for m in container["volumeMounts"]}
        assert mounts["data"] == {"name": "data", "mountPath": "/data", "readOnly": True}
    assert "--data-root /data" in job["spec"]["entrypoint"]


def test_data_path_is_configurable(spec):
    _, job = render_rayjob(spec, data_pvc="msd-data", data_path="/mnt/msd")
    assert "--data-root /mnt/msd" in job["spec"]["entrypoint"]
    head, _, _ = _containers(job)
    assert {m["name"]: m["mountPath"] for m in head["volumeMounts"]}["data"] == "/mnt/msd"


def test_no_data_pvc_no_data_volume(spec):
    _, job = render_rayjob(spec)
    for pod in _pods(job):
        assert "data" not in {v["name"] for v in pod["volumes"]}
    assert "--data-root" not in job["spec"]["entrypoint"]
