"""scripts/image-verify.sh: the GPU training image on a real GPU (roadmap 21).

CI builds no image and has no GPU, and the full job allows no skipped test, so the checks live in
a script and the tests only check it statically: it parses, every `angku` command line (host and
in-container) is accepted by the real parser, the spec pins what the image itself carries, and
what it writes stays out of git and the image.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from angku.adapters.local_environment import COMMIT_ENV, IMAGE_DIGEST_ENV
from angku.cli import build_parser

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "image-verify.sh"
TEXT = SCRIPT.read_text()


def test_script_parses_and_is_strict():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert "set -euo pipefail" in TEXT


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "/work/swinunetr_image.yaml", "--artifacts", "/work/container", "--data-root", "/data",
         "--tracker", "inmemory", "--json"],
        ["reproduce", "R", "--artifacts", "/work/container", "--data-root", "/data", "--tracker", "inmemory",
         "--json"],
        ["run", "s.yaml", "--artifacts", "h", "--data-root", "d", "--tracker", "inmemory", "--json"],
    ],
)  # fmt: skip
def test_angku_command_lines_parse(argv):
    build_parser().parse_args(argv)


def test_every_angku_command_in_the_script_is_covered():
    lines = [ln.strip() for ln in TEXT.splitlines() if re.match(r"\s*(in_image )?angku ", ln)]
    assert sorted({ln.removeprefix("in_image ").split()[1] for ln in lines}) == ["reproduce", "run"]


def test_spec_pins_the_commit_baked_into_the_image_and_its_digest():
    # no .git in the container: code.commit is checked against the baked env var, so the spec must pin that
    assert f"sed -n 's/^{COMMIT_ENV}=//p'" in TEXT
    assert '"$PY" - "$BASELINE" "$SPEC" "$EPOCHS" "$COMMIT" "$REPO" "$DIGEST"' in TEXT
    assert f'-e "{IMAGE_DIGEST_ENV}=$DIGEST"' in TEXT  # what KubeRay pods get


def test_runs_with_the_gpu_and_the_data_read_only():
    assert "docker run --rm --gpus all" in TEXT
    assert '-v "$DATA_ROOT:/data:ro"' in TEXT


def test_image_state_stays_out_of_git_and_the_image():
    assert "OUT=.gpu/image" in TEXT
    assert ".gpu/" in (ROOT / ".gitignore").read_text().split()
    assert ".gpu" in (ROOT / ".dockerignore").read_text().split()
