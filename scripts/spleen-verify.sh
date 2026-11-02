#!/usr/bin/env bash
# Check SwinUNETR on the real MSD Task09 spleen data on this machine's GPU (roadmap 21, phase 3):
#
#   1. `mlops snapshot`: the data under DATA_ROOT hashes to the snapshot pinned in
#      experiments/configs/swinunetr_baseline.yaml
#   2. measure: one train epoch (batch 4 x 96^3 crops) and sliding-window validation of the 9
#      full val volumes, time + peak VRAM; both must fit in the GPU's memory
#   3. run A (attempt 1, the baseline spec cut to EPOCHS epochs, keep-best val/dice/spleen)
#   4. `mlops reproduce A`: bit-exact on the same GPU (history, model.pt, best.pt)
#   5. run B (attempt 2, checkpoint every 2 epochs) SIGKILLed from inside at epoch KILL_AT_EPOCH,
#      then `mlops resume B`: B == A bit for bit (history, model.pt, best.pt)
#
#   scripts/spleen-verify.sh DATA_ROOT        (DATA_ROOT holds Task09_Spleen/; or $MLOPS_DATA_ROOT)
#
# Needs torch with CUDA, the medical extra and the data; exit 2 otherwise. Sets
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless already set (the mlops CLI does too; the
# measure step does not go through it): without it a batch of 4
# reserves more than a 16 GB card and spills into shared memory (~7x slower on WSL). The setting
# changes allocation only, not results. Spec -> .gpu/spleen/swinunetr_gpu.yaml, bundles ->
# .gpu/spleen/artifacts (cleared first). ~11 min on an RTX 5070 Ti. Exit 0 = every check passed.
set -euo pipefail

EPOCHS="${EPOCHS:-4}"
KILL_AT_EPOCH="${KILL_AT_EPOCH:-3}"
DATA_ROOT="${1:-${MLOPS_DATA_ROOT:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MLFLOW_DISABLE_AGENT_HINT=1

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
PY="$root/.venv/bin/python"
BASELINE=experiments/configs/swinunetr_baseline.yaml
OUT=.gpu/spleen
SPEC=$OUT/swinunetr_gpu.yaml
ART=$OUT/artifacts
mlops() { "$PY" -m mlops "$@"; }
step() { echo; echo "== $*"; }

step "GPU + data"
if [ -z "$DATA_ROOT" ] || [ ! -f "$DATA_ROOT/Task09_Spleen/dataset.json" ]; then
    echo "usage: $0 DATA_ROOT  (the directory holding Task09_Spleen/dataset.json; or set MLOPS_DATA_ROOT)" >&2
    exit 2
fi
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
if ! "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo 'torch.cuda is not available: uv pip install -e ".[dev,medical]" --torch-backend=auto' >&2
    exit 2
fi
if ! "$PY" -c 'import monai, nibabel' 2>/dev/null; then
    echo 'needs the medical extra: uv pip install -e ".[dev,medical]" --torch-backend=auto' >&2
    exit 2
fi
"$PY" -c 'import torch; p = torch.cuda.get_device_properties(0); print(p.name, f"{p.total_memory / 2**30:.1f} GiB",
"| torch", torch.__version__, "| CUDA", torch.version.cuda)'
echo "data root $DATA_ROOT | PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

step "spec"
rm -rf "$ART" && mkdir -p "$ART"
"$PY" - "$BASELINE" "$SPEC" "$EPOCHS" "$(git rev-parse HEAD)" <<'EOF'
import sys
import yaml
baseline, path, epochs, commit = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
raw = yaml.safe_load(open(baseline))
raw["meta"]["name"] = "swinunetr-gpu"
raw["code"]["commit"] = commit
raw["training"]["epochs"] = epochs
raw["tracking"] = {"backend": "inmemory", "experiment_name": "spleen-verify"}
yaml.safe_dump(raw, open(path, "w"), sort_keys=False)
print(f"{path}: {baseline} with {epochs} epochs, commit {commit[:12]}")
EOF

step "snapshot"
mlops snapshot "$SPEC" --data-root "$DATA_ROOT" > $OUT/snapshot.txt
cat $OUT/snapshot.txt

step "measure: one train epoch + sliding-window validation"
"$PY" - "$SPEC" "$DATA_ROOT" $OUT/measure.json <<'EOF'
import json
import sys
import time

import torch

from mlops.application.runner import seed_everything
from mlops.config import load_experiment_yaml
from mlops.registry import DATASETS, INFERERS, LOSSES, MODELS, discover

spec_path, data_root, out = sys.argv[1:4]
discover("experiments")
spec = load_experiment_yaml(spec_path)
t = time.perf_counter()
splits = DATASETS.build(spec.dataset.name, {"root": data_root}).load(spec.dataset, spec.preprocessing)
result = {"preprocess_s": time.perf_counter() - t, "total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}
seed_everything(spec.training.seed, deterministic=True)
dev = torch.device("cuda")
model = MODELS.build(spec.model.name, spec.model.params).to(dev)
loss_fn = LOSSES.build(spec.loss.name, spec.loss.params).to(dev)
inferer = INFERERS.build(spec.inference.name, spec.inference.params)
opt = torch.optim.AdamW(model.parameters(), lr=spec.training.learning_rate)


def measured(name, t):
    torch.cuda.synchronize()
    result[name] = {"s": time.perf_counter() - t, "alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


model.train()
t = time.perf_counter()
for x, y in torch.utils.data.DataLoader(splits.train, batch_size=spec.training.batch_size):
    opt.zero_grad()
    loss_fn(model(x.to(dev)), y.to(dev)).backward()
    opt.step()
measured("train_epoch", t)
model.eval()
t = time.perf_counter()
with torch.no_grad():
    for i in range(len(splits.val)):
        inferer(model, splits.val[i][0][None].to(dev))
measured("sliding_window_val", t)
result["val_cases"] = len(splits.val)
json.dump(result, open(out, "w"), indent=2)
print(f"preprocess {result['preprocess_s']:.0f}s")
for k in ("train_epoch", "sliding_window_val"):
    r = result[k]
    print(f"{k}: {r['s']:.1f}s  peak allocated {r['alloc_gib']:.2f} GiB  peak reserved {r['reserved_gib']:.2f} GiB"
          f" (of {result['total_gib']:.1f})")
EOF

step "run A (attempt 1)"
mlops run "$SPEC" --artifacts "$ART" --data-root "$DATA_ROOT" --tracker inmemory --attempt 1 \
    --keep-best val/dice/spleen:max --json > $OUT/run_a.json
RUN_A="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' $OUT/run_a.json)"
echo "run A = $RUN_A"

step "reproduce A"
set +e
mlops reproduce "$RUN_A" --artifacts "$ART" --data-root "$DATA_ROOT" --tracker inmemory --json > $OUT/reproduce_a.json
REPRODUCE_EXIT=$?
set -e
echo "exit $REPRODUCE_EXIT"

step "run B (attempt 2): SIGKILL at epoch $KILL_AT_EPOCH"
# A file, not a heredoc: bash echoes the whole command when it reports the SIGKILL.
cat > $OUT/kill_run.py <<'EOF'
import os
import signal
import sys

from mlops.adapters.inmemory_tracker import InMemoryTracker
from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.adapters.local_environment import LocalEnvironmentProbe
from mlops.application import run_experiment
from mlops.config import load_experiment_yaml
from mlops.domain.best import BestCriterion
from mlops.registry import DATASETS, discover

spec_path, art, data_root, kill_at = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])


class KillingTracker(InMemoryTracker):
    def log_metrics(self, metrics, *, step):
        super().log_metrics(metrics, step=step)
        if step == kill_at:  # before this epoch's checkpoint: the resume must replay it
            print(f"killing at epoch {step}", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)


discover("experiments")
spec = load_experiment_yaml(spec_path)
run_experiment(
    spec,
    data=DATASETS.build(spec.dataset.name, {"root": data_root}),
    tracker=KillingTracker(),
    artifacts=LocalArtifactStore(art),
    probe=LocalEnvironmentProbe(),
    attempt=2,
    checkpoint_every=2,
    keep_best=BestCriterion(metric="val/dice/spleen", mode="max"),
)
EOF
set +e
"$PY" $OUT/kill_run.py "$SPEC" "$ART" "$DATA_ROOT" "$KILL_AT_EPOCH" 2>/dev/null
KILL_EXIT=$?
set -e
echo "exit $KILL_EXIT (137 = SIGKILL)"
RUN_B="$("$PY" -c 'import sys
from mlops.config import load_experiment_yaml
from mlops.domain import compute_experiment_id, compute_run_id
s = load_experiment_yaml(sys.argv[1]); print(compute_run_id(compute_experiment_id(s), s.resources, 2))' "$SPEC")"
echo "run B = $RUN_B"
mlops list --artifacts "$ART" --json > $OUT/list_killed.json

step "resume B"
mlops resume "$RUN_B" --artifacts "$ART" --data-root "$DATA_ROOT" --tracker inmemory --json > $OUT/resume_b.json

step "checks"
"$PY" - "$ART" "$RUN_A" "$RUN_B" "$REPRODUCE_EXIT" "$KILL_EXIT" "$KILL_AT_EPOCH" "$EPOCHS" "$BASELINE" "$OUT" <<'EOF'
import io
import json
import sys
from pathlib import Path

import torch
import yaml

art, run_a, run_b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
reproduce_exit, kill_exit, kill_at, epochs = (int(x) for x in sys.argv[4:8])
baseline, out = sys.argv[8], Path(sys.argv[9])
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def manifest(run):
    return json.loads((art / run / "manifest.json").read_text())


def weights(run, name):
    return torch.load(io.BytesIO((art / run / name).read_bytes()), map_location="cpu", weights_only=True)


def same_weights(name):
    wa, wb = weights(run_a, name), weights(run_b, name)
    return wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa), f"{len(wa)} tensors"


pinned = yaml.safe_load(open(baseline))["dataset"]["snapshot"]
served = (out / "snapshot.txt").read_text().strip()
check("data hashes to the snapshot pinned in the baseline", served == pinned, served[:19] + "...")
m = json.loads((out / "measure.json").read_text())
for k in ("train_epoch", "sliding_window_val"):
    r = m[k]
    check(f"{k} fits in GPU memory", r["reserved_gib"] < m["total_gib"],
          f"{r['s']:.1f}s, peak reserved {r['reserved_gib']:.2f} of {m['total_gib']:.1f} GiB")

a, b = manifest(run_a), manifest(run_b)
rep = json.loads((out / "reproduce_a.json").read_text())
killed = {r["run_id"]: r for g in json.loads((out / "list_killed.json").read_text()) for r in g["runs"]}
resume = json.loads((out / "resume_b.json").read_text())
check("run A FINISHED on cuda", a["status"] == "FINISHED" and a["device"].startswith("cuda"), a["device"])
check("run A verified the data snapshot", (a.get("data_snapshot") or {}).get("status") == "ok",
      str((a.get("data_snapshot") or {}).get("status")))
best = a.get("best") or {}
check("run A kept a best epoch", best.get("epoch") is not None,
      f"{best.get('metric')} {best.get('value')} @ epoch {best.get('epoch')}")
check("reproduce A: REPRODUCED (torch.equal, model.pt + best.pt)", reproduce_exit == 0 and all(rep["checks"].values()),
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
ok, detail = same_weights("model.pt")
check("model.pt: resumed B == uninterrupted A (torch.equal)", ok, detail)
ok, detail = same_weights("best.pt")
check("best.pt: resumed B == uninterrupted A (torch.equal)", ok and a["best"] == b["best"], detail)
check("no checkpoint.pt left behind", not (art / run_b / "checkpoint.pt").exists())

code = a["environment"]["checks"].get("code.commit", {})
print(f"(info) code.commit: {code.get('status')}  {code.get('detail', '')}")
print("(info) final " + "  ".join(f"{k}={v:.4f}" for k, v in a["history"][-1].items() if isinstance(v, float)))
width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
failed = sum(not ok for _, ok, _ in results)
print(f"\n{len(results) - failed}/{len(results)} PASS")
sys.exit(1 if failed else 0)
EOF
