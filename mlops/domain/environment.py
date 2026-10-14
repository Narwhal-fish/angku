"""What a run actually ran on, and whether that matches what the spec pins. Pure data + logic.

``RunEnvironment`` is captured by an ``EnvironmentProbe`` adapter (git, importlib, torch);
``verify_environment`` compares it with the spec's pins. Statuses:

    ok        verified equal
    mismatch  verified different (the run did not run what the spec says)
    unknown   could not be verified here (e.g. no git repo, not in a container)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlops.domain.experiment import ExperimentSpec

CheckStatus = Literal["ok", "mismatch", "unknown"]

# Changed files listed by name in a mismatch detail; the rest are counted.
_MAX_LISTED = 10


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CodeState(_Frozen):
    """The running code relative to ``code.commit``.

    ``source="git"``: a repo was found. ``pinned_found`` says whether the pinned commit exists
    in it; ``changed`` lists code files that differ from it (None when it could not be diffed).
    ``source="env"``: no repo; ``head`` is ``MLOPS_GIT_COMMIT`` baked into the image.
    ``source="none"``: neither.
    """

    source: Literal["git", "env", "none"]
    head: str | None = None
    pinned_found: bool | None = None
    changed: list[str] | None = None


class RunEnvironment(_Frozen):
    python: str
    platform: str
    packages: dict[str, str] = Field(default_factory=dict)  # installed ones only
    cuda: str | None = None  # torch.version.cuda
    cudnn: int | None = None
    gpus: list[str] = Field(default_factory=list)
    torch_threads: int
    code: CodeState
    image_digest: str | None = None  # of the container this ran in, if known


class EnvCheck(_Frozen):
    status: CheckStatus
    detail: str


class EnvironmentRecord(_Frozen):
    """Stored in the manifest: the facts plus the verdict at capture time."""

    actual: RunEnvironment
    checks: dict[str, EnvCheck]

    @property
    def verified(self) -> bool:
        return all(c.status == "ok" for c in self.checks.values())


def verify_environment(spec: ExperimentSpec, env: RunEnvironment) -> dict[str, EnvCheck]:
    return {
        "code.commit": _check_code(spec.code.commit, env.code),
        "environment.image_digest": _check_image(spec.environment.image_digest, env.image_digest),
    }


def record_environment(spec: ExperimentSpec, env: RunEnvironment) -> EnvironmentRecord:
    return EnvironmentRecord(actual=env, checks=verify_environment(spec, env))


def _check_code(pinned: str | None, code: CodeState) -> EnvCheck:
    if pinned is None:
        return EnvCheck(status="unknown", detail="code.commit is not pinned")
    short = pinned[:12]
    if code.source == "git":
        if not code.pinned_found or code.changed is None:
            return EnvCheck(status="unknown", detail=f"commit {pinned} is not in this repo (fetch it?)")
        if code.changed:
            listed = ", ".join(code.changed[:_MAX_LISTED])
            more = f" and {len(code.changed) - _MAX_LISTED} more" if len(code.changed) > _MAX_LISTED else ""
            return EnvCheck(
                status="mismatch",
                detail=f"{len(code.changed)} file(s) differ from {short}: {listed}{more}",
            )
        at = "" if code.head == pinned else f" (HEAD {code.head[:12] if code.head else '?'}; only non-code files differ)"
        return EnvCheck(status="ok", detail=f"code matches {short}{at}")
    if code.source == "env":
        if code.head == pinned:
            return EnvCheck(status="ok", detail=f"image was built from {short} (MLOPS_GIT_COMMIT)")
        return EnvCheck(status="mismatch", detail=f"image was built from {code.head}, spec pins {pinned}")
    return EnvCheck(status="unknown", detail="no git repo and no MLOPS_GIT_COMMIT")


def _check_image(pinned: str | None, actual: str | None) -> EnvCheck:
    if pinned is None:
        return EnvCheck(status="unknown", detail="environment.image_digest is not pinned")
    if actual is None:
        return EnvCheck(status="unknown", detail="not running in a known image (MLOPS_IMAGE_DIGEST unset)")
    if actual == pinned:
        return EnvCheck(status="ok", detail=f"running in {pinned}")
    return EnvCheck(status="mismatch", detail=f"running in {actual}, spec pins {pinned}")
