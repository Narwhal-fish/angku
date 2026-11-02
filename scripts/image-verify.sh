#!/usr/bin/env bash
# Check the GPU training image on this machine's GPU (roadmap 21, phase 4):
#
#   1. the image sees the GPU (`docker run --gpus all`): torch.cuda, its CUDA + cuDNN
#   2. run C: the baseline spec cut to EPOCHS epochs, pinned to this image (image + digest, and
#      code.commit = the MLOPS_GIT_COMMIT baked into it), run inside the container on the real
#      spleen data; the pod-style MLOPS_IMAGE_DIGEST is passed like KubeRay does
#   3. every pin verifies inside the container: code.commit, environment.image_digest,
#      dataset.snapshot all ok
#   4. `mlops reproduce C` inside the container: bit-exact
#   5. (info, not a check) the same spec run on the host: same run_id; whether history and
#      weights are identical to the container's (bit-exactness is only promised within one env)
#
#   scripts/image-verify.sh REPO@sha256:DIGEST DATA_ROOT
#
# REPO@sha256:... = what build-image.sh --push printed (e.g. localhost:5001/mlops-train@sha256:...).
# DATA_ROOT holds Task09_Spleen/ (or $MLOPS_DATA_ROOT). Needs docker with GPU support, torch with
# CUDA on the host, the medical extra; exit 2 otherwise. Spec -> .gpu/image/swinunetr_image.yaml,
# bundles -> .gpu/image/container (mounted at /work) and .gpu/image/host (both cleared first).
# Exit 0 = every check passed.
set -euo pipefail

EPOCHS="${EPOCHS:-2}"
IMAGE_REF="${1:-}"
DATA_ROOT="${2:-${MLOPS_DATA_ROOT:-}}"
export MLFLOW_DISABLE_AGENT_HINT=1

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
PY="$root/.venv/bin/python"
BASELINE=experiments/configs/swinunetr_baseline.yaml
OUT=.gpu/image
SPEC=$OUT/swinunetr_image.yaml
mlops() { "$PY" -m mlops "$@"; }
step() { echo; echo "== $*"; }
usage() { echo "usage: $0 REPO@sha256:DIGEST DATA_ROOT  (DATA_ROOT holds Task09_Spleen/; or set MLOPS_DATA_ROOT)" >&2; exit 2; }

step "image + data"
[[ "$IMAGE_REF" =~ ^([^@]+)@(sha256:[0-9a-f]{64})$ ]] || usage
REPO="${BASH_REMATCH[1]}" DIGEST="${BASH_REMATCH[2]}"
{ [ -n "$DATA_ROOT" ] && [ -f "$DATA_ROOT/Task09_Spleen/dataset.json" ]; } || usage
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
if ! timeout 15 docker info >/dev/null 2>&1; then
    echo "docker does not answer (Docker Desktop running? this shell in the docker group?)" >&2
    exit 2
fi
if ! docker image inspect "$IMAGE_REF" >/dev/null 2>&1 && ! docker pull -q "$IMAGE_REF" >/dev/null; then
    echo "no image $IMAGE_REF" >&2
    exit 2
fi
COMMIT="$(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$IMAGE_REF" \
    | sed -n 's/^MLOPS_GIT_COMMIT=//p')"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "$IMAGE_REF has no MLOPS_GIT_COMMIT: not built by build-image.sh" >&2; exit 2; }
echo "$REPO @ ${DIGEST:0:19}... built from ${COMMIT:0:12} | data root $DATA_ROOT"
rm -rf "$OUT/container" "$OUT/host" && mkdir -p "$OUT/container" "$OUT/host"

in_image() {
    docker run --rm --gpus all -e "MLOPS_IMAGE_DIGEST=$DIGEST" -e MLFLOW_DISABLE_AGENT_HINT=1 \
        -v "$root/$OUT:/work" -v "$DATA_ROOT:/data:ro" -w /work "$IMAGE_REF" "$@"
}
# The base image's entrypoint (kept: KubeRay pods run it too) prints NVIDIA's CUDA banner on stdout;
# the JSON a command prints starts at the first line that opens an object.
json_only() { sed -n '/^{/,$p'; }

step "GPU in the container"
in_image python -c 'import json, torch; print(json.dumps({"cuda_available": torch.cuda.is_available(),
"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))' | json_only > $OUT/container_gpu.json
cat $OUT/container_gpu.json

step "spec"
"$PY" - "$BASELINE" "$SPEC" "$EPOCHS" "$COMMIT" "$REPO" "$DIGEST" <<'EOF'
import sys
import yaml
baseline, path, epochs, commit, repo, digest = sys.argv[1], sys.argv[2], int(sys.argv[3]), *sys.argv[4:7]
raw = yaml.safe_load(open(baseline))
raw["meta"]["name"] = "swinunetr-image"
raw["code"]["commit"] = commit
raw["environment"] = {"image": repo, "image_digest": digest}
raw["training"]["epochs"] = epochs
raw["tracking"] = {"backend": "inmemory", "experiment_name": "image-verify"}
yaml.safe_dump(raw, open(path, "w"), sort_keys=False)
print(f"{path}: {baseline} with {epochs} epochs, image {repo}@{digest[:19]}..., commit {commit[:12]}")
EOF

step "run C in the container"
in_image mlops run /work/swinunetr_image.yaml --artifacts /work/container --data-root /data --tracker inmemory \
    --json | json_only > $OUT/run_c.json
RUN_C="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' $OUT/run_c.json)"
echo "run C = $RUN_C"

step "reproduce C in the container"
set +e
in_image mlops reproduce "$RUN_C" --artifacts /work/container --data-root /data --tracker inmemory \
    --json | json_only > $OUT/reproduce_c.json
REPRODUCE_EXIT=$?
set -e
echo "exit $REPRODUCE_EXIT"

step "run H on the host (same spec)"
mlops run "$SPEC" --artifacts $OUT/host --data-root "$DATA_ROOT" --tracker inmemory --json > $OUT/run_h.json
RUN_H="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' $OUT/run_h.json)"
echo "run H = $RUN_H"

step "checks"
"$PY" - "$OUT" "$RUN_C" "$RUN_H" "$REPRODUCE_EXIT" "$DIGEST" "$BASELINE" <<'EOF'
import io
import json
import sys
from pathlib import Path

import torch
import yaml

out, run_c, run_h, reproduce_exit, digest, baseline = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), \
    sys.argv[5], sys.argv[6]
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def bundle(where, run):
    d = out / where / run
    return json.loads((d / "manifest.json").read_text()), torch.load(
        io.BytesIO((d / "model.pt").read_bytes()), map_location="cpu", weights_only=True)


gpu = json.loads((out / "container_gpu.json").read_text())
check("container sees the GPU", gpu["cuda_available"], f"{gpu['gpu']} | torch {gpu['torch']} | CUDA {gpu['cuda']}"
      f" | cuDNN {gpu['cudnn']}")
c, wc = bundle("container", run_c)
env = c["environment"]
check("run C FINISHED on cuda in the container", c["status"] == "FINISHED" and c["device"].startswith("cuda"),
      c["device"])
for key in ("code.commit", "environment.image_digest"):
    chk = env["checks"].get(key, {})
    check(f"{key} verified in the container", chk.get("status") == "ok", chk.get("detail", ""))
check("the probe recorded this image's digest", env["actual"]["image_digest"] == digest, str(env["actual"]["image_digest"]))
ds = c.get("data_snapshot") or {}
check("dataset.snapshot verified in the container", ds.get("status") == "ok", ds.get("detail", "")[:60])
pinned = yaml.safe_load(open(baseline))["dataset"]["snapshot"]
check("... against the baseline's pin", pinned in ds.get("detail", ""), pinned[:19] + "...")
rep = json.loads((out / "reproduce_c.json").read_text())
check("reproduce C in the container: REPRODUCED (torch.equal)", reproduce_exit == 0 and all(rep["checks"].values()),
      ", ".join(f"{k}={v}" for k, v in rep["checks"].items()))

h, wh = bundle("host", run_h)
ha = h["environment"]["actual"]
print(f"(info) host: torch {ha['packages'].get('torch')} | CUDA {ha['cuda']} | cuDNN {ha['cudnn']}"
      f"   container: torch {env['actual']['packages'].get('torch')} | CUDA {env['actual']['cuda']}"
      f" | cuDNN {env['actual']['cudnn']}")
print(f"(info) host vs container: same run_id {run_h == run_c}, history identical {h['history'] == c['history']}")
same = wc.keys() == wh.keys() and all(torch.equal(wc[k], wh[k]) for k in wc)
worst = max((wc[k].float() - wh[k].float()).abs().max().item() for k in wc) if wc.keys() == wh.keys() else float("nan")
print(f"(info) host vs container weights: torch.equal {same}, max |diff| {worst:.3g} over {len(wc)} tensors")
for (ec, mc), mh in zip(enumerate(c["history"], 1), h["history"], strict=False):
    print(f"(info)   epoch {ec}: " + "  ".join(f"{k} {mc[k]:.6f}/{mh.get(k, float('nan')):.6f}" for k in mc))

width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
failed = sum(not ok for _, ok, _ in results)
print(f"\n{len(results) - failed}/{len(results)} PASS")
sys.exit(1 if failed else 0)
EOF
