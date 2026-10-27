""".github/workflows/ci.yml: checked statically (it only runs on GitHub).

The lint job must run the ruff version pinned in the dev extra, or CI and the dev box disagree
on what passes; the formatting commit listed in .git-blame-ignore-revs must be a full SHA.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _dev_ruff_pin() -> str:
    (pin,) = [d for d in PYPROJECT["project"]["optional-dependencies"]["dev"] if d.startswith("ruff")]
    return re.fullmatch(r"ruff==([\d.]+)", pin).group(1)


def test_runs_on_pull_requests_only():
    # PyYAML reads the bare key `on` as True.
    assert set(WORKFLOW[True]) == {"pull_request"}


def test_lint_job_runs_the_pinned_ruff_check_and_format():
    script = "\n".join(step.get("run", "") for step in WORKFLOW["jobs"]["lint"]["steps"])
    versions = set(re.findall(r"ruff@([\d.]+)", script))
    assert versions == {_dev_ruff_pin()}
    assert "ruff@" in script and " check" in script and "format --check" in script


def test_blame_ignore_revs_are_full_shas():
    revs = [
        line
        for line in (ROOT / ".git-blame-ignore-revs").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert revs and all(re.fullmatch(r"[0-9a-f]{40}", r) for r in revs)
