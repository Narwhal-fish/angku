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

    # the spec rides along verbatim as JSON (a YAML subset, so `angku run` reads it as-is)
    assert ExperimentSpec.model_validate_json(cm["data"]["experiment.json"]) == spec

    entry = job["spec"]["entrypoint"]
    assert entry.startswith("angku run /etc/mlops/experiment.json")
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
        spec,
        namespace="ml",
        attempt=2,
        artifacts_pvc="runs-pvc",
        artifacts_path="/mnt/runs",
        mlflow_uri="http://mlflow:5000",
    )
    assert cm["metadata"]["namespace"] == job["metadata"]["namespace"] == "ml"
    assert job["metadata"]["name"].endswith("-a2")
    entry = job["spec"]["entrypoint"]
    assert "--attempt 2" in entry and "--artifacts /mnt/runs" in entry and "--mlflow-uri http://mlflow:5000" in entry
    head, _, worker = _containers(job)
    for pod in (
        job["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"],
        job["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"]["spec"],
    ):
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
    def __init__(self, returncode=0, stderr="", stdout="rayjob.ray.io/x created\n"):
        self.calls = []
        self.returncode, self.stderr, self.stdout = returncode, stderr, stdout

    def __call__(self, cmd, *, input, capture_output, text):
        self.calls.append((cmd, input))

        class R:
            returncode = self.returncode
            stdout = self.stdout
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
    return json.dumps(
        {
            "kind": "RayJob",
            "metadata": {
                "name": "mlops-abc-a1",
                "namespace": "ml",
                "annotations": {"mlops.io/run-id": "r" * 64, "mlops.io/experiment-id": "e" * 64},
            },
            "status": {
                "jobStatus": job_status,
                "jobDeploymentStatus": deployment,
                "message": message,
                "dashboardURL": "head-svc:8265",
                "startTime": "2026-10-13T10:00:00Z",
                **status,
            },
        }
    )


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


@pytest.mark.parametrize(
    "job,finished,succeeded",
    [
        ("PENDING", False, False),
        ("RUNNING", False, False),
        ("SUCCEEDED", True, True),
        ("FAILED", True, False),
        ("STOPPED", True, False),
        ("", False, False),
    ],
)
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


# --- resumable jobs: Ray checkpoints + launch.json on PVCs that outlive the RayJob's cluster ---


def _mounts(pod):
    (container,) = pod["containers"]
    return {m["name"]: m for m in container["volumeMounts"]}


def _resumable(spec, **kw):
    return render_rayjob(spec, artifacts_pvc="runs", ray_storage_pvc="ray-ckpt", data_pvc="msd", **kw)


def test_checkpointing_job_stores_ray_checkpoints_on_a_shared_pvc(spec):
    _, job = _resumable(spec, checkpoint_every=5)
    entry = job["spec"]["entrypoint"]
    assert "--checkpoint-every 5" in entry and "--ray-storage /ray-storage" in entry
    # Ray workers write their checkpoint shards, the driver (head) looks them up on resume:
    # every pod sees the same claim, writable, at the same path.
    for pod in _pods(job):
        vols = {v["name"]: v for v in pod["volumes"]}
        assert vols["ray-storage"]["persistentVolumeClaim"] == {"claimName": "ray-ckpt"}
        assert _mounts(pod)["ray-storage"] == {"name": "ray-storage", "mountPath": "/ray-storage"}


def test_ray_storage_path_is_configurable(spec):
    _, job = _resumable(spec, checkpoint_every=1, ray_storage_path="/mnt/ray")
    assert "--ray-storage /mnt/ray" in job["spec"]["entrypoint"]
    head, _, _ = _containers(job)
    assert {m["name"]: m["mountPath"] for m in head["volumeMounts"]}["ray-storage"] == "/mnt/ray"


def test_no_checkpointing_no_ray_storage(spec):
    _, job = render_rayjob(spec, artifacts_pvc="runs")
    entry = job["spec"]["entrypoint"]
    assert "--checkpoint-every" not in entry and "--ray-storage" not in entry
    for pod in _pods(job):
        assert "ray-storage" not in {v["name"] for v in pod["volumes"]}


@pytest.mark.parametrize(
    "kw,missing",
    [
        ({"ray_storage_pvc": "ray-ckpt"}, "artifacts_pvc"),
        ({"artifacts_pvc": "runs"}, "ray_storage_pvc"),
    ],
)
@pytest.mark.parametrize("mode", [{"checkpoint_every": 5}, {"resume": True}])
def test_resumability_needs_both_pvcs(spec, kw, missing, mode):
    # Without them the checkpoints / launch.json die with the cluster: refuse instead of
    # rendering a job that only looks resumable.
    with pytest.raises(ValueError, match=missing):
        render_rayjob(spec, **kw, **mode)


def test_checkpoint_every_must_be_positive(spec):
    with pytest.raises(ValueError, match="checkpoint_every"):
        _resumable(spec, checkpoint_every=0)


def test_resume_job_runs_angku_resume_for_the_same_run(spec):
    (_, fresh), (job,) = (
        _resumable(spec, attempt=2, checkpoint_every=5),
        _resumable(spec, attempt=2, resume=True),
    )
    rid = compute_run_id(compute_experiment_id(spec), spec.resources, attempt=2)
    assert job["kind"] == "RayJob"
    assert job["metadata"]["annotations"] == fresh["metadata"]["annotations"]
    assert job["metadata"]["annotations"]["mlops.io/run-id"] == rid
    # Kubernetes picks a fresh suffix per resume: the first RayJob (and earlier resumes) may
    # still exist, finished, and no one has to delete them first.
    assert "name" not in job["metadata"]
    assert job["metadata"]["generateName"] == fresh["metadata"]["name"] + "-resume-"
    # one selectable label ties every job of a run together (annotations cannot be selected)
    assert job["metadata"]["labels"]["mlops.io/run"] == fresh["metadata"]["labels"]["mlops.io/run"] == rid[:32]
    entry = job["spec"]["entrypoint"]
    assert entry.startswith(f"angku resume {rid} --artifacts /artifacts")
    assert "--data-root /data" in entry and "--mlflow-uri" not in entry
    # spec.json and launch.json (storage path, interval, attempt) come from the artifacts PVC
    assert "--checkpoint-every" not in entry and "--ray-storage" not in entry
    # ... so the spec ConfigMap is not needed; the storage must sit where the first job put it
    for pod in _pods(job):
        assert "experiment" not in {v["name"] for v in pod["volumes"]}
        assert {m["name"]: m["mountPath"] for m in _mounts(pod).values()} == {
            "artifacts": "/artifacts",
            "ray-storage": "/ray-storage",
            "data": "/data",
        }
    assert _containers(job)[0]["image"] == _containers(fresh)[0]["image"]


def test_resume_job_can_change_the_checkpoint_interval(spec):
    (job,) = _resumable(spec, resume=True, checkpoint_every=2, mlflow_uri="http://mlflow:5000")
    entry = job["spec"]["entrypoint"]
    assert "--checkpoint-every 2" in entry and "--mlflow-uri http://mlflow:5000" in entry


@pytest.mark.parametrize("mode", [{"checkpoint_every": 3}, {"resume": True, "checkpoint_every": 3}])
def test_entrypoint_is_accepted_by_the_angku_cli(spec, mode):
    # The in-cluster command must parse with the very CLI it calls: no flag drift.
    from mlops.cli import build_parser

    job = _by_kind(_resumable(spec, mlflow_uri="http://mlflow:5000", **mode), "RayJob")
    argv = job["spec"]["entrypoint"].split()
    assert argv[0] == "angku"
    args = build_parser().parse_args(argv[1:])
    assert args.checkpoint_every == 3 and args.data_root == "/data" and args.artifacts == "/artifacts"


def test_resume_job_is_created_and_its_generated_name_returned(spec):
    # `apply` cannot handle generateName; `create -o name` prints the name Kubernetes chose.
    kubectl = _FakeKubectl(stdout="rayjob.ray.io/mlops-abc-a1-resume-x8k2p\n")
    name = submit_rayjob(_resumable(spec, resume=True), run=kubectl)
    ((cmd, _),) = kubectl.calls
    assert cmd == ["kubectl", "create", "-n", "default", "-o", "name", "-f", "-"]
    assert name == "mlops-abc-a1-resume-x8k2p"


def test_active_rayjobs_of_a_run_are_found_by_label(spec):
    from mlops.adapters.kuberay import active_rayjobs

    listing = json.dumps(
        {
            "kind": "List",
            "items": [
                json.loads(_rayjob_json(job_status="FAILED")) | {"metadata": {"name": "mlops-abc-a1"}},
                json.loads(_rayjob_json(job_status="RUNNING")) | {"metadata": {"name": "mlops-abc-a1-resume-x8k2p"}},
                json.loads(_rayjob_json(job_status=""))
                | {"metadata": {"name": "mlops-abc-a1-resume-q4z7d"}},  # just created
            ],
        }
    )
    kubectl = _FakeGet(listing)
    rid = compute_run_id(compute_experiment_id(spec), spec.resources, attempt=1)
    active = active_rayjobs(rid, namespace="ml", run=kubectl)
    assert [a.name for a in active] == ["mlops-abc-a1-resume-x8k2p", "mlops-abc-a1-resume-q4z7d"]
    ((cmd, _),) = kubectl.calls
    assert cmd == ["kubectl", "get", "rayjob", "-n", "ml", "-l", f"mlops.io/run={rid[:32]}", "-o", "json"]


def test_active_rayjobs_reports_kubectl_failure():
    from mlops.adapters.kuberay import active_rayjobs

    with pytest.raises(KubeRayError, match="forbidden"):
        active_rayjobs("r" * 64, run=_FakeGet(returncode=1, stderr="Error from server (Forbidden): forbidden"))


# --- for `angku ui` (roadmap 13 phase D) ---


def test_list_rayjobs_finds_every_job_we_rendered():
    from mlops.adapters.kuberay import list_rayjobs

    listing = json.dumps(
        {
            "kind": "List",
            "items": [
                json.loads(_rayjob_json(job_status="SUCCEEDED", deployment="Complete"))
                | {"metadata": {"name": "mlops-a-a1"}},
                json.loads(_rayjob_json(job_status="RUNNING")) | {"metadata": {"name": "mlops-b-a1"}},
            ],
        }
    )
    kubectl = _FakeGet(listing)
    jobs = list_rayjobs(namespace="ml", kubectl=["kubectl", "--context", "x"], run=kubectl)
    assert [(j.name, j.job_status, j.finished) for j in jobs] == [
        ("mlops-a-a1", "SUCCEEDED", True),
        ("mlops-b-a1", "RUNNING", False),
    ]
    ((cmd, _),) = kubectl.calls
    assert cmd == ["kubectl", "--context", "x", "get", "rayjob", "-n", "ml", "-l", "mlops.io/run", "-o", "json"]
    with pytest.raises(KubeRayError, match="forbidden"):
        list_rayjobs(run=_FakeGet(returncode=1, stderr="forbidden"))


def test_rayjob_logs_are_captured_tail():
    from mlops.adapters.kuberay import rayjob_logs

    kubectl = _FakeGet("epoch 1\nepoch 2\n")
    assert rayjob_logs("mlops-a-a1", namespace="ml", tail=50, run=kubectl) == "epoch 1\nepoch 2\n"
    ((cmd, kw),) = kubectl.calls
    assert cmd == ["kubectl", "logs", "-n", "ml", "job/mlops-a-a1", "--tail=50"]
    assert kw["capture_output"] is True
    with pytest.raises(KubeRayError, match="NotFound"):
        rayjob_logs("x", run=_FakeGet(returncode=1, stderr="Error from server (NotFound)"))
