"""scripts/kind-up.sh, kind-down.sh, kind-verify.sh: the KubeRay path checked on a local kind cluster.

Running them needs Docker and pulls GBs, so tests only check them statically: they parse, the
pins they share agree, every `mlops` command line they run is accepted by the real parser,
and the state they write stays out of git and out of the image.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from mlops.cli import build_parser

ROOT = Path(__file__).parent.parent
SCRIPTS = {name: (ROOT / "scripts" / name).read_text() for name in ("kind-up.sh", "kind-down.sh", "kind-verify.sh")}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_script_parses_and_is_strict(name):
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)
    assert "set -euo pipefail" in SCRIPTS[name]


def test_registry_and_cluster_names_agree():
    up, down, verify = SCRIPTS["kind-up.sh"], SCRIPTS["kind-down.sh"], SCRIPTS["kind-verify.sh"]
    port = re.search(r"^REGISTRY_PORT=(\d+)", up, re.M).group(1)
    assert f"IMAGE=localhost:{port}/" in verify
    assert f'"image": "localhost:{port}/mlops-train"' in verify
    for script in (up, down, verify):
        assert 'CLUSTER="${MLOPS_KIND_CLUSTER:-mlops}"' in script
    assert re.search(r"^REGISTRY=mlops-registry$", down, re.M) and re.search(r"^REGISTRY=mlops-registry$", up, re.M)


def test_kuberay_is_pinned():
    assert re.search(r"^KUBERAY_VERSION=v\d+\.\d+\.\d+$", SCRIPTS["kind-up.sh"], re.M)


def test_pvcs_the_verify_script_mounts_are_created():
    up, verify = SCRIPTS["kind-up.sh"], SCRIPTS["kind-verify.sh"]
    assert "for name in artifacts ray-storage" in up and "name: mlops-$name" in up
    assert "--artifacts-pvc mlops-artifacts --ray-storage-pvc mlops-ray-storage" in verify


def test_image_is_built_through_the_clean_tree_check():
    assert 'scripts/build-image.sh "$IMAGE" --cpu --push' in SCRIPTS["kind-verify.sh"]


@pytest.mark.parametrize("argv", [
    ["rayjob", "s.yaml", "--attempt", "1", "--artifacts-pvc", "a", "--ray-storage-pvc", "r",
     "--checkpoint-every", "2", "--submit", "--kubectl", "kubectl --context kind-mlops"],
    ["rayjob", "s.yaml", "--attempt", "1", "--artifacts-pvc", "a", "--ray-storage-pvc", "r",
     "--resume", "--submit", "--kubectl", "k"],
    ["rayjob-status", "job", "--kubectl", "k", "--json"],
    ["rayjob-logs", "job", "--kubectl", "k"],
    ["list", "--artifacts", ".kind/artifacts", "--json"],
    ["ui", "--artifacts", ".kind/artifacts", "--kubectl", "k"],
])
def test_mlops_command_lines_of_the_verify_script_parse(argv):
    build_parser().parse_args(argv)


def test_every_mlops_subcommand_used_is_one_of_those():
    used = set(re.findall(r"\bmlops (rayjob-status|rayjob-logs|rayjob|list|ui)\b", SCRIPTS["kind-verify.sh"]))
    assert used <= {"rayjob", "rayjob-status", "rayjob-logs", "list", "ui"} and "rayjob" in used


def test_kind_state_stays_out_of_git_and_the_image():
    assert ".kind/" in (ROOT / ".gitignore").read_text().split()
    assert ".kind" in (ROOT / ".dockerignore").read_text().split()
