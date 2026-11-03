"""scripts/gpu-verify.sh: the CUDA claims checked on a real GPU (roadmap 21).

CI has no GPU, and the full job allows no skipped test, so the GPU checks live in a script
(like kind-verify.sh) and the tests only check it statically: it parses, every `angku` command
it runs is accepted by the real parser, and what it writes stays out of git and the image.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mlops.cli import build_parser

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "gpu-verify.sh"


def test_script_parses_and_is_strict():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert "set -euo pipefail" in SCRIPT.read_text()


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "s.yaml", "--artifacts", "a", "--tracker", "inmemory", "--attempt", "1", "--json"],
        ["reproduce", "R", "--artifacts", "a", "--tracker", "inmemory", "--json"],
        ["resume", "R", "--artifacts", "a", "--tracker", "inmemory", "--json"],
        ["list", "--artifacts", "a", "--json"],
    ],
)
def test_angku_command_lines_parse(argv):
    build_parser().parse_args(argv)


def test_every_angku_command_in_the_script_is_covered():
    lines = [ln.strip() for ln in SCRIPT.read_text().splitlines() if ln.strip().startswith("angku ")]
    assert sorted({ln.split()[1] for ln in lines}) == ["list", "reproduce", "resume", "run"]


def test_gpu_state_stays_out_of_git_and_the_image():
    assert ".gpu/" in (ROOT / ".gitignore").read_text().split()
    assert ".gpu" in (ROOT / ".dockerignore").read_text().split()
