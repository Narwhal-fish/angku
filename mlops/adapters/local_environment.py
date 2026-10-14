"""``EnvironmentProbe`` for the current process: git, importlib.metadata, torch.

Code state: the working tree of ``repo`` is diffed against ``code.commit`` over
``CODE_PATHS`` (untracked files included), excluding ``experiments/configs/``. Comparing
HEAD itself cannot work: committing the yaml that pins a commit moves HEAD. Without a repo
(a container), ``MLOPS_GIT_COMMIT`` baked into the image is used. The image digest comes
from ``MLOPS_IMAGE_DIGEST`` (set in KubeRay pods; a laptop has none).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

import torch

from mlops.domain.environment import CodeState, RunEnvironment
from mlops.domain.experiment import ExperimentSpec

CODE_PATHS = ("mlops", "experiments", "pyproject.toml")
EXCLUDED_PATHS = ("experiments/configs",)
PACKAGES = ("mlops", "torch", "numpy", "pydantic", "monai", "nibabel", "einops", "ray", "mlflow")

COMMIT_ENV = "MLOPS_GIT_COMMIT"
IMAGE_DIGEST_ENV = "MLOPS_IMAGE_DIGEST"


class LocalEnvironmentProbe:
    def __init__(self, repo: Path | str | None = None, environ: Mapping[str, str] | None = None) -> None:
        # Absolute so a Ray worker (different cwd, same machine) looks at the same repo.
        self.repo = Path(repo).resolve() if repo is not None else Path.cwd()
        self.environ = dict(os.environ if environ is None else environ)

    def capture(self, spec: ExperimentSpec) -> RunEnvironment:
        return RunEnvironment(
            python=".".join(map(str, sys.version_info[:3])),
            platform=platform.platform(),
            packages=_packages(),
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
            torch_threads=torch.get_num_threads(),
            code=self._code_state(spec.code.commit),
            image_digest=self.environ.get(IMAGE_DIGEST_ENV) or None,
        )

    def _code_state(self, pinned: str | None) -> CodeState:
        top = self._git("rev-parse", "--show-toplevel")
        if top is None:
            baked = self.environ.get(COMMIT_ENV) or None
            return CodeState(source="env" if baked else "none", head=baked)
        head = self._git("rev-parse", "HEAD")
        if pinned is None or self._git("cat-file", "-e", f"{pinned}^{{commit}}") is None:
            return CodeState(source="git", head=head, pinned_found=False)
        spec = ["--", *CODE_PATHS, *(f":(exclude){p}" for p in EXCLUDED_PATHS)]
        diffed = self._git("diff", "--name-only", pinned, *spec, cwd=top)
        untracked = self._git("ls-files", "--others", "--exclude-standard", *spec, cwd=top)
        if diffed is None or untracked is None:
            return CodeState(source="git", head=head, pinned_found=True, changed=None)
        changed = sorted({*diffed.splitlines(), *untracked.splitlines()} - {""})
        return CodeState(source="git", head=head, pinned_found=True, changed=changed)

    def _git(self, *args: str, cwd: str | None = None) -> str | None:
        """stdout, or None if git is missing or the command fails."""
        try:
            r = subprocess.run(
                ["git", *args], cwd=cwd or self.repo, capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None


def _packages() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out
