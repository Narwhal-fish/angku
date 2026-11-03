"""Verify the environment instead of trusting the yaml.

Two parts:

* ``LocalEnvironmentProbe.capture(spec)`` (adapter) records what the run *actually* ran on:
  python/platform, package versions, CUDA/cuDNN, GPU models, torch threads, the state of the
  code relative to the pinned commit, and the container image digest.
* ``verify_environment(spec, env)`` (domain, pure) turns that into checks
  ``{"code.commit": EnvCheck, "environment.image_digest": EnvCheck}`` with status
  ``ok`` / ``mismatch`` / ``unknown``.

Code check: HEAD == code.commit cannot work, because committing the yaml that pins a commit
moves HEAD. So the probe compares the *code paths* (``angku/``, ``experiments/`` minus
``experiments/configs/``, ``pyproject.toml``) in the working tree against the pinned commit,
untracked files included. No git repo (a container) -> ``ANGKU_GIT_COMMIT`` baked into the
image. Image digest -> ``ANGKU_IMAGE_DIGEST`` (KubeRay pods get it; a laptop has none).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from angku.adapters.local_environment import LocalEnvironmentProbe
from angku.config import load_experiment_yaml
from angku.domain.environment import CodeState, EnvCheck, RunEnvironment, verify_environment

CONFIGS = Path(__file__).parent.parent / "experiments/configs"
DIGEST = "sha256:" + "ab" * 32
OTHER_DIGEST = "sha256:" + "cd" * 32


@pytest.fixture
def base_spec():
    return load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")


def _pin(spec, commit: str, digest: str = DIGEST):
    return spec.model_copy(
        update={
            "code": spec.code.model_copy(update={"commit": commit}),
            "environment": spec.environment.model_copy(update={"image_digest": digest}),
        }
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "repo"
    (r / "angku").mkdir(parents=True)
    (r / "experiments/configs").mkdir(parents=True)
    (r / "angku/core.py").write_text("x = 1\n")
    (r / "experiments/model.py").write_text("y = 1\n")
    (r / "experiments/configs/e.yaml").write_text("a: 1\n")
    (r / "pyproject.toml").write_text("[project]\n")
    (r / "README.md").write_text("hi\n")
    _git(r, "init", "-q")
    _git(r, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    return r


def _commit_all(repo: Path, msg: str = "c") -> str:
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD")


def _probe(repo: Path, **env: str) -> LocalEnvironmentProbe:
    return LocalEnvironmentProbe(repo=repo, environ=env)


def _checks(repo, spec, **env):
    return verify_environment(spec, _probe(repo, **env).capture(spec))


# --- code.commit ---


def test_clean_tree_at_pinned_commit_is_ok(repo, base_spec):
    spec = _pin(base_spec, _git(repo, "rev-parse", "HEAD"))
    assert _checks(repo, spec)["code.commit"].status == "ok"


def test_committing_only_the_yaml_after_pinning_is_still_ok(repo, base_spec):
    pinned = _git(repo, "rev-parse", "HEAD")
    (repo / "experiments/configs/e.yaml").write_text("a: 2\n")
    (repo / "README.md").write_text("docs\n")
    assert _commit_all(repo) != pinned
    assert _checks(repo, _pin(base_spec, pinned))["code.commit"].status == "ok"


def test_modified_code_file_is_a_mismatch_naming_the_file(repo, base_spec):
    spec = _pin(base_spec, _git(repo, "rev-parse", "HEAD"))
    (repo / "angku/core.py").write_text("x = 2\n")
    check = _checks(repo, spec)["code.commit"]
    assert check.status == "mismatch" and "angku/core.py" in check.detail


def test_untracked_code_file_is_a_mismatch(repo, base_spec):
    spec = _pin(base_spec, _git(repo, "rev-parse", "HEAD"))
    (repo / "experiments/new_loss.py").write_text("z = 1\n")
    check = _checks(repo, spec)["code.commit"]
    assert check.status == "mismatch" and "experiments/new_loss.py" in check.detail


def test_code_committed_after_the_pin_is_a_mismatch(repo, base_spec):
    pinned = _git(repo, "rev-parse", "HEAD")
    (repo / "experiments/model.py").write_text("y = 2\n")
    _commit_all(repo)
    assert _checks(repo, _pin(base_spec, pinned))["code.commit"].status == "mismatch"


def test_dependency_change_is_a_mismatch(repo, base_spec):
    spec = _pin(base_spec, _git(repo, "rev-parse", "HEAD"))
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert _checks(repo, spec)["code.commit"].status == "mismatch"


def test_commit_not_in_repo_is_unknown(repo, base_spec):
    check = _checks(repo, _pin(base_spec, "f" * 40))["code.commit"]
    assert check.status == "unknown" and "f" * 40 in check.detail


def test_no_repo_falls_back_to_baked_commit_env(tmp_path, base_spec):
    spec = _pin(base_spec, "a" * 40)
    assert _checks(tmp_path, spec, ANGKU_GIT_COMMIT="a" * 40)["code.commit"].status == "ok"
    assert _checks(tmp_path, spec, ANGKU_GIT_COMMIT="b" * 40)["code.commit"].status == "mismatch"
    assert _checks(tmp_path, spec)["code.commit"].status == "unknown"


def test_many_changed_files_are_summarised(repo, base_spec):
    spec = _pin(base_spec, _git(repo, "rev-parse", "HEAD"))
    for i in range(20):
        (repo / f"angku/m{i:02}.py").write_text("")
    detail = _checks(repo, spec)["code.commit"].detail
    assert "20 file" in detail and "m19.py" not in detail


# --- image digest ---


def test_image_digest_from_env(repo, base_spec):
    spec = _pin(base_spec, _git(repo, "rev-parse", "HEAD"))
    assert _checks(repo, spec, ANGKU_IMAGE_DIGEST=DIGEST)["environment.image_digest"].status == "ok"
    bad = _checks(repo, spec, ANGKU_IMAGE_DIGEST=OTHER_DIGEST)["environment.image_digest"]
    assert bad.status == "mismatch" and OTHER_DIGEST in bad.detail
    assert _checks(repo, spec)["environment.image_digest"].status == "unknown"


# --- recorded facts ---


def test_capture_records_versions_and_hardware(repo, base_spec):
    env = _probe(repo).capture(_pin(base_spec, _git(repo, "rev-parse", "HEAD")))
    assert env.python == ".".join(map(str, sys.version_info[:3]))
    assert env.packages["torch"] == torch.__version__.split("+")[0] or env.packages["torch"] == torch.__version__
    assert "pydantic" in env.packages
    assert env.torch_threads == torch.get_num_threads()
    assert env.cuda == torch.version.cuda
    assert env.gpus == (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    )
    assert env.code.head == _git(repo, "rev-parse", "HEAD")


def test_environment_round_trips_through_json(repo, base_spec):
    env = _probe(repo).capture(_pin(base_spec, _git(repo, "rev-parse", "HEAD")))
    assert RunEnvironment.model_validate_json(env.model_dump_json()) == env


def test_probe_is_picklable_for_ray_workers(repo):
    import pickle

    p = pickle.loads(pickle.dumps(_probe(repo, ANGKU_IMAGE_DIGEST=DIGEST)))
    assert p.repo == repo


# --- verify_environment is pure ---


def test_verify_is_pure_domain_logic(base_spec):
    spec = _pin(base_spec, "a" * 40)
    env = RunEnvironment(
        python="3.12.0",
        platform="test",
        packages={},
        cuda=None,
        cudnn=None,
        gpus=[],
        torch_threads=1,
        code=CodeState(source="git", head="b" * 40, pinned_found=True, changed=[]),
        image_digest=DIGEST,
    )
    checks = verify_environment(spec, env)
    assert checks == {
        "code.commit": EnvCheck(status="ok", detail=checks["code.commit"].detail),
        "environment.image_digest": EnvCheck(status="ok", detail=checks["environment.image_digest"].detail),
    }


# --- shared test double ---


class FixedProbe:
    """Picklable probe returning a fixed environment; ``pid`` shows which process captured it."""

    def __init__(self, *, commit_ok: bool = True, digest: str | None = None) -> None:
        self.commit_ok, self.digest = commit_ok, digest

    def capture(self, spec) -> RunEnvironment:
        import os

        return RunEnvironment(
            python="3.12.0",
            platform=f"pid={os.getpid()}",
            packages={"torch": "2.x"},
            torch_threads=torch.get_num_threads(),
            code=CodeState(
                source="git", head=spec.code.commit, pinned_found=True, changed=[] if self.commit_ok else ["angku/x.py"]
            ),
            image_digest=self.digest,
        )
