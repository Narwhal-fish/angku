#!/usr/bin/env bash
# Check the CUDA claims on this machine's GPU with tiny_cnn (roadmap 21, phase 1):
#
#   1. run A (attempt 1, gpu: 1) through `angku run`
#   2. `angku reproduce A`: the replay on the same GPU must be bit-exact (torch.equal)
#   3. run B (attempt 2, checkpoint every 2 epochs) SIGKILLed from inside at epoch KILL_AT_EPOCH,
#      after its epoch metrics were logged and before that epoch's checkpoint: a hard kill, no
#      manifest; then `angku resume B` continues from the last checkpoint (CUDA RNG restored)
#   4. checks: B resumed == A bit for bit (history + every tensor of model.pt); the runs ran on
#      cuda; the probe recorded GPU / CUDA / cuDNN; deterministic algorithms on; the cross_entropy
#      loss runs on CUDA under deterministic algorithms (nll_loss2d has no such kernel), gives the
#      same loss and gradient twice, and agrees with nn.CrossEntropyLoss on CPU to rounding
#
#   scripts/gpu-verify.sh
#
# Needs torch with CUDA (`uv pip install -e ".[dev]" --torch-backend=auto`) and a GPU; exit 2
# otherwise. The spec pins code.commit to HEAD: its verdict is printed (a dirty tree says
# mismatch) but is not a check. Writes the spec to .gpu/tiny_cnn_gpu.yaml, bundles to
# .gpu/artifacts (cleared first). Exit 0 = every check passed.
set -euo pipefail

EPOCHS="${EPOCHS:-20}"
KILL_AT_EPOCH="${KILL_AT_EPOCH:-9}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
PY="$root/.venv/bin/python"
SPEC=.gpu/tiny_cnn_gpu.yaml
ART=.gpu/artifacts
angku() { "$PY" -m mlops "$@"; }
step() { echo; echo "== $*"; }

step "GPU"
if ! "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "torch.cuda is not available (torch $("$PY" -c 'import torch; print(torch.__version__)')):" \
        'install a CUDA build: uv pip install -e ".[dev]" --torch-backend=auto' >&2
    exit 2
fi
"$PY" -c 'import torch; print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| CUDA", torch.version.cuda,
"| cuDNN", torch.backends.cudnn.version())'

step "spec"
rm -rf "$ART" && mkdir -p "$ART"
"$PY" - "$SPEC" "$EPOCHS" "$(git rev-parse HEAD)" <<'EOF'
import sys
import yaml
path, epochs, commit = sys.argv[1], int(sys.argv[2]), sys.argv[3]
raw = yaml.safe_load(open("experiments/configs/tiny_cnn_smoke.yaml"))
raw["meta"]["name"] = "tiny-cnn-gpu"
raw["code"]["commit"] = commit
raw["model"]["params"]["hidden"] = 16
raw["training"]["epochs"] = epochs
raw["resources"]["gpu"] = 1
raw["tracking"] = {"backend": "inmemory", "experiment_name": "gpu-verify"}
yaml.safe_dump(raw, open(path, "w"), sort_keys=False)
print(f"{path}: {epochs} epochs, gpu 1, commit {commit[:12]}")
EOF

step "run A (attempt 1)"
angku run "$SPEC" --artifacts "$ART" --tracker inmemory --attempt 1 --json > .gpu/run_a.json
RUN_A="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' .gpu/run_a.json)"
echo "run A = $RUN_A"

step "reproduce A"
set +e
angku reproduce "$RUN_A" --artifacts "$ART" --tracker inmemory --json > .gpu/reproduce_a.json
REPRODUCE_EXIT=$?
set -e
echo "exit $REPRODUCE_EXIT"

step "run B (attempt 2): SIGKILL at epoch $KILL_AT_EPOCH"
# A file, not a heredoc: bash echoes the whole command when it reports the SIGKILL.
cat > .gpu/kill_run.py <<'EOF'
import os
import signal
import sys

from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.adapters.local_environment import LocalEnvironmentProbe
from mlops.adapters.synthetic_data import SyntheticBlobs
from mlops.application import run_experiment
from mlops.config import load_experiment_yaml
from mlops.registry import discover

spec_path, art, kill_at = sys.argv[1], sys.argv[2], int(sys.argv[3])


class KillingTracker(InMemoryTracker):
    def log_metrics(self, metrics, *, step):
        super().log_metrics(metrics, step=step)
        if step == kill_at:  # before this epoch's checkpoint: the resume must replay it
            print(f"killing at epoch {step}", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)


discover("experiments")
run_experiment(
    load_experiment_yaml(spec_path),
    data=SyntheticBlobs(),
    tracker=KillingTracker(),
    artifacts=LocalArtifactStore(art),
    probe=LocalEnvironmentProbe(),
    attempt=2,
    checkpoint_every=2,
)
EOF
set +e
"$PY" .gpu/kill_run.py "$SPEC" "$ART" "$KILL_AT_EPOCH" 2>/dev/null
KILL_EXIT=$?
set -e
echo "exit $KILL_EXIT (137 = SIGKILL)"
RUN_B="$("$PY" -c 'import sys
from mlops.config import load_experiment_yaml
from mlops.domain import compute_experiment_id, compute_run_id
s = load_experiment_yaml(sys.argv[1]); print(compute_run_id(compute_experiment_id(s), s.resources, 2))' "$SPEC")"
echo "run B = $RUN_B"
angku list --artifacts "$ART" --json > .gpu/list_killed.json

step "resume B"
angku resume "$RUN_B" --artifacts "$ART" --tracker inmemory --json > .gpu/resume_b.json

step "checks"
"$PY" - "$ART" "$RUN_A" "$RUN_B" "$REPRODUCE_EXIT" "$KILL_EXIT" "$KILL_AT_EPOCH" "$EPOCHS" <<'EOF'
import io
import json
import sys
from pathlib import Path

import torch

art, run_a, run_b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
reproduce_exit, kill_exit, kill_at, epochs = (int(x) for x in sys.argv[4:8])
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def manifest(run):
    return json.loads((art / run / "manifest.json").read_text())


def weights(run):
    return torch.load(io.BytesIO((art / run / "model.pt").read_bytes()), map_location="cpu", weights_only=True)


a, b = manifest(run_a), manifest(run_b)
rep = json.loads(Path(".gpu/reproduce_a.json").read_text())
killed = {r["run_id"]: r for g in json.loads(Path(".gpu/list_killed.json").read_text()) for r in g["runs"]}
resume = json.loads(Path(".gpu/resume_b.json").read_text())
spec = json.loads((art / run_a / "spec.json").read_text())
actual = a["environment"]["actual"]

check("run A FINISHED on cuda", a["status"] == "FINISHED" and a["device"].startswith("cuda"), a["device"])
check("probe recorded GPU, CUDA, cuDNN", bool(actual["gpus"]) and actual["cuda"] and actual["cudnn"],
      f"{actual['gpus']} cuda {actual['cuda']} cudnn {actual['cudnn']}")
check("deterministic algorithms on", spec["training"]["deterministic"] is True)
check("reproduce A: REPRODUCED (torch.equal)", reproduce_exit == 0 and all(rep["checks"].values()),
      ", ".join(f"{k}={v}" for k, v in rep["checks"].items()))
check("run B was SIGKILLed", kill_exit == 137, f"exit {kill_exit}")
kb = killed.get(run_b, {})
check("killed B lists INTERRUPTED + resumable", kb.get("status") == "INTERRUPTED" and kb.get("resumable"),
      f"{kb.get('status')} resumable={kb.get('resumable')}")
check("resume continued from the last checkpoint before the kill",
      resume.get("resumed_from_epoch") == kill_at - kill_at % 2, f"from epoch {resume.get('resumed_from_epoch')}")
check("run B FINISHED on cuda, resumed once", b["status"] == "FINISHED" and b["device"].startswith("cuda")
      and len(b["resumes"]) == 1, f"{b['status']} {b['device']} resumes={len(b['resumes'])}")
check("history: resumed B == uninterrupted A", a["history"] == b["history"], f"{len(b['history'])}/{epochs} epochs")
wa, wb = weights(run_a), weights(run_b)
same = wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)
check("weights: resumed B == uninterrupted A (torch.equal)", same, f"{len(wa)} tensors")
check("no checkpoint.pt left behind", not (art / run_b / "checkpoint.pt").exists())

from mlops.application.runner import seed_everything  # noqa: E402
from mlops.registry import LOSSES, discover  # noqa: E402

discover("experiments.losses")
seed_everything(0, deterministic=True)
ce = LOSSES.build("cross_entropy", {"weight": [0.4, 1.6], "label_smoothing": 0.1})
logits = torch.randn(2, 2, 16, 16, 16)
target = torch.randint(0, 2, (2, 16, 16, 16))


def ce_on(device):
    x = logits.to(device).requires_grad_()
    loss = ce.to(device)(x, target.to(device))
    loss.backward()
    return loss.detach().cpu(), x.grad.cpu()


(l1, g1), (l2, g2), (lc, gc) = ce_on("cuda"), ce_on("cuda"), ce_on("cpu")
check("cross_entropy on CUDA, deterministic: same loss + grad twice", torch.equal(l1, l2) and torch.equal(g1, g2))
check("cross_entropy on CUDA == CPU nn.CrossEntropyLoss (allclose)",
      torch.allclose(l1, lc, atol=1e-6) and torch.allclose(g1, gc, atol=1e-6), f"|loss diff| {(l1 - lc).abs():.1e}")

code = a["environment"]["checks"].get("code.commit", {})
print(f"(info) code.commit: {code.get('status')}  {code.get('detail', '')}")
width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
failed = sum(not ok for _, ok, _ in results)
print(f"\n{len(results) - failed}/{len(results)} PASS")
sys.exit(1 if failed else 0)
EOF
