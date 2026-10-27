""".github/workflows/ci.yml: checked statically (it only runs on GitHub).

The lint job must run the ruff version pinned in the dev extra, or CI and the dev box disagree
on what passes; the formatting commits listed in .git-blame-ignore-revs must be full SHAs on the current history
(a squash merge replaces the branch's commits, leaving the listed one unreachable).
"""

from __future__ import annotations

import re
import subprocess
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


def _blame_ignore_revs() -> list[str]:
    lines = (ROOT / ".git-blame-ignore-revs").read_text().splitlines()
    return [line for line in lines if line.strip() and not line.startswith("#")]


def test_blame_ignore_revs_are_full_shas():
    revs = _blame_ignore_revs()
    assert revs and all(re.fullmatch(r"[0-9a-f]{40}", r) for r in revs)


def test_blame_ignore_revs_are_on_the_current_history():
    for rev in _blame_ignore_revs():
        git = ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", rev, "HEAD"]
        assert subprocess.run(git, capture_output=True).returncode == 0, f"{rev} is not an ancestor of HEAD"
