"""The training image: Dockerfile + scripts/build-image.sh.

In-cluster there is no git repo, so ``LocalEnvironmentProbe`` verifies code.commit against
``MLOPS_GIT_COMMIT`` baked into the image. That only means something if the image is built
from a clean checkout of exactly that commit: the build script refuses otherwise.

The script is exercised in a throwaway git repo with a fake ``docker`` on PATH; nothing is
actually built (the real image is GPU-sized).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mlops.adapters.kuberay import RAY_VERSION

ROOT = Path(__file__).parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text() if (ROOT / "Dockerfile").exists() else ""
SCRIPT = ROOT / "scripts" / "build-image.sh"


# --- Dockerfile ---


def test_ray_in_the_image_matches_the_rayjob():
    # KubeRay's rayVersion and the image's ray must agree, or the cluster will not start.
    assert re.search(rf"^ARG RAY_VERSION={re.escape(RAY_VERSION)}$", DOCKERFILE, re.M)
    assert "rayproject/ray:${RAY_VERSION}" in DOCKERFILE
    assert '"ray[train]==${RAY_VERSION}"' in DOCKERFILE  # extras must not upgrade it


def test_commit_is_required_validated_and_baked_in():
    assert re.search(r"^ARG MLOPS_GIT_COMMIT$", DOCKERFILE, re.M)
    assert "^[0-9a-f]{40}$" in DOCKERFILE
    assert re.search(r"^ENV .*MLOPS_GIT_COMMIT=\$\{MLOPS_GIT_COMMIT\}", DOCKERFILE, re.M)


def test_cuda_allocator_default_is_the_cli_one():
    # Ray workers on other nodes do not inherit the driver's environment: the image sets it for every pod.
    from mlops.cli import CUDA_ALLOC_CONF, CUDA_ALLOC_DEFAULT

    assert re.search(rf"^ENV {CUDA_ALLOC_CONF}={re.escape(CUDA_ALLOC_DEFAULT)}$", DOCKERFILE, re.M)


def test_experiments_are_importable_from_any_working_directory():
    # The Ray job driver does not start in /app; `discover("experiments")` must still work.
    assert re.search(r"^ENV .*PYTHONPATH=/app", DOCKERFILE, re.M)
    assert re.search(r"^COPY .*experiments \./experiments$", DOCKERFILE, re.M)
    assert re.search(r"^COPY .*mlops \./mlops$", DOCKERFILE, re.M)


def test_gpu_torch_wheel_is_chosen_explicitly():
    # `--torch-backend=auto` at build time sees no GPU and would pick the CPU wheel.
    assert "--torch-backend=${TORCH_BACKEND}" in DOCKERFILE
    assert re.search(r"^ARG TORCH_BACKEND=cu\d+$", DOCKERFILE, re.M)


def test_cpu_base_is_selectable():
    # kind on a laptop has no GPU: the CPU flavour of the same Ray release (and the CPU torch wheel).
    assert re.search(r"^ARG RAY_FLAVOR=-gpu$", DOCKERFILE, re.M)
    assert DOCKERFILE.index("ARG RAY_FLAVOR") < DOCKERFILE.index("\nFROM ")  # usable in FROM
    assert "FROM rayproject/ray:${RAY_VERSION}-py312${RAY_FLAVOR}" in DOCKERFILE


def test_all_extras_the_cluster_needs_are_installed():
    assert ".[tracking,distributed,medical]" in DOCKERFILE


def test_dockerignore_keeps_git_and_local_state_out():
    ignored = (ROOT / ".dockerignore").read_text().split()
    # no .git in the image: code is verified via MLOPS_GIT_COMMIT, not a repo that could drift
    for entry in (".git", ".venv", "artifacts", ".kind", "tests", "**/__pycache__"):
        assert entry in ignored, entry


# --- build script ---


@pytest.fixture
def repo(tmp_path) -> Path:
    """A minimal clean checkout with the real script and Dockerfile."""
    r = tmp_path / "repo"
    (r / "mlops").mkdir(parents=True)
    (r / "experiments" / "configs").mkdir(parents=True)
    (r / "scripts").mkdir()
    (r / "mlops" / "x.py").write_text("x = 1\n")
    (r / "experiments" / "configs" / "a.yaml").write_text("a: 1\n")
    (r / "pyproject.toml").write_text("[project]\nname='m'\n")
    shutil.copy(ROOT / "Dockerfile", r / "Dockerfile")
    shutil.copy(ROOT / ".dockerignore", r / ".dockerignore")
    shutil.copy(SCRIPT, r / "scripts" / "build-image.sh")
    git = ["git", "-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    return r


@pytest.fixture
def docker(tmp_path) -> Path:
    """Fake docker: appends its argv to a log; `inspect` prints a repo digest."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "docker.log"
    fake = bin_ / "docker"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> {log}\n'
        'if [ "$1" = inspect ]; then echo "ghcr.io/x/train@sha256:' + "ab" * 32 + '"; fi\n'
    )
    fake.chmod(0o755)
    return log


def _build(repo: Path, docker_log: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{docker_log.parent / 'bin'}:{os.environ['PATH']}"}
    return subprocess.run(
        ["bash", str(repo / "scripts" / "build-image.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo / "experiments",
    )


def _head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def _calls(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def test_clean_checkout_builds_with_the_commit_baked_in(repo, docker):
    r = _build(repo, docker, "ghcr.io/x/train:0.2")
    assert r.returncode == 0, r.stderr
    (call,) = _calls(docker)
    assert call.startswith("build ")
    assert f"--build-arg MLOPS_GIT_COMMIT={_head(repo)}" in call
    assert "-t ghcr.io/x/train:0.2" in call and call.endswith(str(repo))  # context = repo root, any cwd
    assert f"code.commit: {_head(repo)}" in r.stdout
    assert "--push" in r.stdout  # explains that the digest only exists after a push
    assert "RAY_FLAVOR" not in call and "TORCH_BACKEND" not in call  # the Dockerfile's GPU defaults


def test_cpu_build_picks_the_cpu_base_and_wheel(repo, docker):
    r = _build(repo, docker, "localhost:5001/train:kind", "--cpu", "--push")
    assert r.returncode == 0, r.stderr
    build, _, _ = _calls(docker)
    assert "--build-arg RAY_FLAVOR= " in build and "--build-arg TORCH_BACKEND=cpu" in build
    assert f"--build-arg MLOPS_GIT_COMMIT={_head(repo)}" in build


@pytest.mark.parametrize("path", ["mlops/x.py", "pyproject.toml", "Dockerfile"])
def test_uncommitted_code_is_refused(repo, docker, path):
    (repo / path).write_text((repo / path).read_text() + "\n# local edit\n")
    r = _build(repo, docker, "ghcr.io/x/train:0.2")
    assert r.returncode == 2
    assert path in r.stderr and "commit" in r.stderr
    assert _calls(docker) == []


def test_untracked_code_is_refused(repo, docker):
    (repo / "experiments" / "new_model.py").write_text("y = 2\n")
    r = _build(repo, docker, "ghcr.io/x/train:0.2")
    assert r.returncode == 2 and "experiments/new_model.py" in r.stderr
    assert _calls(docker) == []


def test_editing_experiment_configs_does_not_block_a_build(repo, docker):
    # Same rule as the environment probe: yamls are not code (committing one moves HEAD anyway).
    (repo / "experiments" / "configs" / "a.yaml").write_text("a: 2\n")
    assert _build(repo, docker, "ghcr.io/x/train:0.2").returncode == 0


def test_push_prints_the_values_to_pin(repo, docker):
    r = _build(repo, docker, "ghcr.io/x/train:0.2", "--push")
    assert r.returncode == 0, r.stderr
    build, push, inspect = _calls(docker)
    assert push == "push ghcr.io/x/train:0.2"
    assert inspect.startswith("inspect")
    assert f"image_digest: sha256:{'ab' * 32}" in r.stdout
    assert "image: ghcr.io/x/train:0.2" in r.stdout


@pytest.mark.parametrize("args", [[], ["--push"], ["--cpu"], ["a", "b"], ["a", "--gpu"]])
def test_usage_errors(repo, docker, args):
    r = _build(repo, docker, *args)
    assert r.returncode == 2 and "usage" in r.stderr
    assert _calls(docker) == []
