#!/usr/bin/env bash
# Check the KubeRay path for real on the kind cluster of scripts/kind-up.sh (CPU only):
#
#   1. build the CPU image from HEAD and push it to the kind registry (digest pinned)
#   2. run A (attempt 1, 2 workers, checkpoint every 2 epochs); kill its RayJob mid-run
#   3. wait until the run lists INTERRUPTED (its live progress went stale), rayjob --resume it
#   4. run B (attempt 2) uninterrupted, same spec
#   5. checks: A resumed == B bit for bit (history + every tensor of model.pt); code.commit and
#      image_digest verified inside the pods; all epochs reached the driver (metrics relay with a
#      num-cpus=0 head); rayjob-logs reads the submitter Job; the UI's RayJobs list sees both runs
#
#   scripts/kind-verify.sh [--skip-build]       (--skip-build: reuse the last pushed image)
#
# Needs a clean checkout: the image bakes MLOPS_GIT_COMMIT=HEAD (build-image.sh refuses otherwise).
# Writes the spec it runs to .kind/tiny_cnn_kind.yaml, bundles to .kind/artifacts. Each finished
# RayJob is deleted after its checks: KubeRay keeps the cluster for ttlSecondsAfterFinished, and
# two clusters do not fit on one laptop. Exit 0 = every check passed.
set -euo pipefail

CLUSTER="${MLOPS_KIND_CLUSTER:-mlops}"
KUBECTL="kubectl --context kind-$CLUSTER"
IMAGE=localhost:5001/mlops-train:kind
EPOCHS="${EPOCHS:-40}"
KILL_AT_EPOCH="${KILL_AT_EPOCH:-8}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
PY="$root/.venv/bin/python"
SPEC=.kind/tiny_cnn_kind.yaml
ART=.kind/artifacts
PVCS=(--artifacts-pvc mlops-artifacts --ray-storage-pvc mlops-ray-storage)
STATUSES=.kind/job-statuses.txt
mlops() { "$PY" -m mlops "$@"; }
step() { echo; echo "== $*"; }
submitted() { sed -n 's/^submitted RayJob \([^ ]*\) .*/\1/p'; }

wait_job() {  # NAME -> waits until finished; records every jobStatus seen; prints the final one
    local name="$1" deadline=$((SECONDS + 1800)) st
    while [ $SECONDS -lt $deadline ]; do
        st="$(mlops rayjob-status "$name" --kubectl "$KUBECTL" --json 2>/dev/null || true)"
        if [ -n "$st" ]; then
            "$PY" -c 'import json,sys; s=json.loads(sys.argv[1]); print(s["job_status"] or "(none)")' "$st" >> "$STATUSES"
            if "$PY" -c 'import json,sys; sys.exit(0 if json.loads(sys.argv[1])["finished"] else 1)' "$st"; then
                "$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["job_status"])' "$st"
                return 0
            fi
        fi
        sleep 5
    done
    echo "TIMEOUT"
}

epoch_of() {  # RUN_ID -> epoch in its live progress.json (0 if none yet)
    "$PY" -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["epoch"])
except (OSError, ValueError, KeyError): print(0)' "$ART/$1/progress.json"
}

status_of() {  # RUN_ID -> status as `mlops list` sees it
    mlops list --artifacts "$ART" --json | "$PY" -c 'import json,sys
runs = [r for g in json.load(sys.stdin) for r in g["runs"]]
print(next((r["status"] for r in runs if r["run_id"] == sys.argv[1]), "ABSENT"))' "$1"
}

run_id() {  # ATTEMPT -> run_id of $SPEC
    "$PY" -c 'import sys
from mlops.config import load_experiment_yaml
from mlops.domain import compute_experiment_id, compute_run_id
s = load_experiment_yaml(sys.argv[1]); print(compute_run_id(compute_experiment_id(s), s.resources, int(sys.argv[2])))' "$SPEC" "$1"
}

: > "$STATUSES"

step "image"
if [ "${1:-}" = --skip-build ]; then
    digest="$(docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"; digest="${digest##*@}"
else
    out="$(scripts/build-image.sh "$IMAGE" --cpu --push)"
    echo "$out" | tail -4
    digest="$(echo "$out" | sed -n 's/^image_digest: //p')"
fi
commit="$(git rev-parse HEAD)"
echo "commit $commit  digest $digest"

step "spec $SPEC ($EPOCHS epochs, 2 workers)"
"$PY" - "$commit" "$digest" "$EPOCHS" "$SPEC" <<'EOF'
import sys
import yaml
commit, digest, epochs, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = yaml.safe_load(open("experiments/configs/tiny_cnn_smoke.yaml"))
d["meta"]["name"] = "tiny-cnn-kind"
d["code"]["commit"] = commit
d["environment"] = {"image": "localhost:5001/mlops-train", "image_digest": digest}
d["training"]["epochs"] = epochs
d["resources"] = {"gpu": 0, "cpu": 1, "memory": "2Gi", "workers": 2}
d["tracking"] = {"backend": "mlflow", "experiment_name": "kind", "uri": "sqlite:////artifacts/mlflow.db"}
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
EOF
RUN_A="$(run_id 1)"; RUN_B="$(run_id 2)"
echo "run A $RUN_A"; echo "run B $RUN_B"

step "run A: submit, kill at epoch $KILL_AT_EPOCH"
JOB_A="$(mlops rayjob "$SPEC" --attempt 1 "${PVCS[@]}" --checkpoint-every 2 --submit --kubectl "$KUBECTL" | submitted)"
echo "RayJob $JOB_A"
deadline=$((SECONDS + 1200))
until [ "$(epoch_of "$RUN_A")" -ge "$KILL_AT_EPOCH" ]; do
    [ $SECONDS -lt $deadline ] || { echo "run A never reached epoch $KILL_AT_EPOCH" >&2; exit 1; }
    sleep 2
done
echo "epoch $(epoch_of "$RUN_A") reached; deleting $JOB_A"
$KUBECTL delete rayjob "$JOB_A" --wait=true
KILLED_AT="$(epoch_of "$RUN_A")"

step "run A: wait for INTERRUPTED (heartbeat goes stale after 60 s)"
deadline=$((SECONDS + 300))
until [ "$(status_of "$RUN_A")" = INTERRUPTED ]; do
    [ $SECONDS -lt $deadline ] || { echo "run A did not become INTERRUPTED: $(status_of "$RUN_A")" >&2; exit 1; }
    sleep 5
done

step "run A: rayjob --resume"
JOB_A2="$(mlops rayjob "$SPEC" --attempt 1 "${PVCS[@]}" --resume --submit --kubectl "$KUBECTL" | submitted)"
echo "RayJob $JOB_A2"
FINAL_A="$(wait_job "$JOB_A2")"
LOGS_A="$(mlops rayjob-logs "$JOB_A2" --kubectl "$KUBECTL" 2>&1 || true)"
$KUBECTL delete rayjob "$JOB_A2" --wait=true

step "run B: uninterrupted"
JOB_B="$(mlops rayjob "$SPEC" --attempt 2 "${PVCS[@]}" --checkpoint-every 2 --submit --kubectl "$KUBECTL" | submitted)"
echo "RayJob $JOB_B"
FINAL_B="$(wait_job "$JOB_B")"
LOGS_B="$(mlops rayjob-logs "$JOB_B" --kubectl "$KUBECTL" 2>&1 || true)"
UI_JOBS="$("$PY" - "$ART" "$KUBECTL" <<'EOF'
import sys
from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.cli import _ui_actions, build_parser
args = build_parser().parse_args(["ui", "--artifacts", sys.argv[1], "--kubectl", sys.argv[2]])
print(" ".join(j.run_id or "-" for j in _ui_actions(args, LocalArtifactStore(sys.argv[1])).rayjobs()))
EOF
)"
$KUBECTL delete rayjob "$JOB_B" --wait=true

step "checks"
rc=0
"$PY" - "$ART" "$RUN_A" "$RUN_B" "$commit" "$digest" "$EPOCHS" "$KILLED_AT" "$FINAL_A" "$FINAL_B" \
    "$LOGS_A" "$LOGS_B" "$UI_JOBS" <<'EOF' || rc=$?
import json
import sys
from pathlib import Path

import torch

art, run_a, run_b, commit, digest, epochs, killed_at, final_a, final_b, logs_a, logs_b, ui_jobs = sys.argv[1:]
epochs, killed_at = int(epochs), int(killed_at)
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


def manifest(run: str) -> dict:
    return json.loads((Path(art) / run / "manifest.json").read_text())


a, b = manifest(run_a), manifest(run_b)
check("RayJob A (resumed) SUCCEEDED", final_a == "SUCCEEDED", final_a)
check("RayJob B SUCCEEDED", final_b == "SUCCEEDED", final_b)
check("both runs FINISHED", a["status"] == b["status"] == "FINISHED", f"{a['status']} / {b['status']}")
check("A was killed mid-run and resumed once", 0 < killed_at < epochs and len(a["resumes"]) == 1,
      f"killed at epoch {killed_at}, resumes {[r['epoch'] for r in a['resumes']]}")
check("every epoch reached the driver (metrics relay, num-cpus=0 head)",
      len(a["history"]) == len(b["history"]) == epochs, f"{len(a['history'])} / {len(b['history'])} of {epochs}")
check("history: resumed A == uninterrupted B", a["history"] == b["history"])
wa = torch.load(Path(art) / run_a / "model.pt", map_location="cpu", weights_only=True)
wb = torch.load(Path(art) / run_b / "model.pt", map_location="cpu", weights_only=True)
same = wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)
check("weights: resumed A == uninterrupted B (torch.equal)", same, f"{len(wa)} tensors")
for label, m in (("A", a), ("B", b)):
    env = m["environment"]
    checks = {k: v["status"] for k, v in env["checks"].items()}
    check(f"{label}: code.commit + image_digest verified in the pod",
          checks.get("code.commit") == "ok" and checks.get("environment.image_digest") == "ok", str(checks))
    check(f"{label}: MLOPS_IMAGE_DIGEST reached the pod", env["actual"]["image_digest"] == digest,
          str(env["actual"]["image_digest"]))
check("no progress.json left behind",
      not (Path(art) / run_a / "progress.json").exists() and not (Path(art) / run_b / "progress.json").exists())
check("rayjob-logs reads the submitter Job (A resume)", "FINISHED" in logs_a)
check("rayjob-logs reads the submitter Job (B)", "FINISHED" in logs_b)
check("UI RayJobs list sees run B", run_b in ui_jobs.split(), ui_jobs)

width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
EOF
echo
echo "jobStatus values seen: $(sort -u "$STATUSES" | tr '\n' ' ')"
echo "browse: mlops ui --artifacts $ART --kubectl \"$KUBECTL\""
exit $rc
