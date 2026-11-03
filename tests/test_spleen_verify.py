"""scripts/spleen-verify.sh: SwinUNETR on the real MSD Task09 data on a real GPU (roadmap 21).

CI has neither the GPU nor the 1.5 GB dataset, and the full job allows no skipped test, so the
checks live in a script and the tests only check it statically: it parses, every `angku` command
it runs is accepted by the real parser, it checks the snapshot the baseline pins, and what it
writes stays out of git and the image.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from mlops.cli import build_parser

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "spleen-verify.sh"


def test_script_parses_and_is_strict():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert "set -euo pipefail" in SCRIPT.read_text()


@pytest.mark.parametrize(
    "argv",
    [
        ["snapshot", "s.yaml", "--data-root", "d"],
        ["run", "s.yaml", "--artifacts", "a", "--data-root", "d", "--tracker", "inmemory", "--attempt", "1",
         "--keep-best", "val/dice/spleen:max", "--json"],
        ["reproduce", "R", "--artifacts", "a", "--data-root", "d", "--tracker", "inmemory", "--json"],
        ["resume", "R", "--artifacts", "a", "--data-root", "d", "--tracker", "inmemory", "--json"],
        ["list", "--artifacts", "a", "--json"],
    ],
)  # fmt: skip
def test_angku_command_lines_parse(argv):
    build_parser().parse_args(argv)


def test_every_angku_command_in_the_script_is_covered():
    lines = [ln.strip() for ln in SCRIPT.read_text().splitlines() if ln.strip().startswith("angku ")]
    assert sorted({ln.split()[1] for ln in lines}) == ["list", "reproduce", "resume", "run", "snapshot"]


def test_starts_from_the_baseline_that_pins_a_content_hash():
    text = SCRIPT.read_text()
    assert "BASELINE=experiments/configs/swinunetr_baseline.yaml" in text
    baseline = (ROOT / "experiments/configs/swinunetr_baseline.yaml").read_text()
    assert re.search(r"^\s+snapshot: sha256:[0-9a-f]{64}$", baseline, re.M)


def test_keep_best_metric_is_one_the_baseline_logs():
    # dice with class_names [background, spleen] and no background -> the only key is dice/spleen
    baseline = (ROOT / "experiments/configs/swinunetr_baseline.yaml").read_text()
    assert "include_background: false, class_names: [background, spleen]" in baseline
    text = SCRIPT.read_text()
    assert "--keep-best val/dice/spleen:max" in text  # run A (CLI)
    assert 'BestCriterion(metric="val/dice/spleen", mode="max")' in text  # run B (kill_run.py): same criterion


def test_spleen_state_stays_out_of_git_and_the_image():
    assert "OUT=.gpu/spleen" in SCRIPT.read_text()
    assert ".gpu/" in (ROOT / ".gitignore").read_text().split()
    assert ".gpu" in (ROOT / ".dockerignore").read_text().split()
