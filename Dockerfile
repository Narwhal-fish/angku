# Training image for `angku run` locally in a container and for KubeRay RayJobs.
#
# Build it with scripts/build-image.sh, never by hand: the script refuses a dirty checkout
# and passes MLOPS_GIT_COMMIT, which is how a run inside this image verifies code.commit
# (there is no .git in the image). Pin the pushed digest as environment.image_digest.
#
# RAY_VERSION must equal mlops.adapters.kuberay.RAY_VERSION (tests/test_image.py checks it).
# TORCH_BACKEND = the PyTorch wheel index. The wheels bring their own CUDA runtime, so what has to
# fit is the host driver (cu130 needs >= 580) and the GPU: it must be an index that has the torch
# pyproject.toml asks for (cu128 stops at 2.11) and kernels for Blackwell (cu126 has none).
# `build-image.sh --cpu` sets RAY_FLAVOR="" and TORCH_BACKEND=cpu: the same Ray release without
# CUDA, for kind on a machine with no GPU.
ARG RAY_VERSION=2.58.0
ARG RAY_FLAVOR=-gpu
FROM rayproject/ray:${RAY_VERSION}-py312${RAY_FLAVOR}

ARG RAY_VERSION
ARG TORCH_BACKEND=cu130
ARG MLOPS_GIT_COMMIT
RUN echo "${MLOPS_GIT_COMMIT}" | grep -Eq '^[0-9a-f]{40}$' \
    || { echo "MLOPS_GIT_COMMIT must be the full 40-hex commit this image is built from; use scripts/build-image.sh" >&2; exit 1; }

WORKDIR /app
COPY --chown=ray:users pyproject.toml ./
COPY --chown=ray:users mlops ./mlops
COPY --chown=ray:users experiments ./experiments

# ray stays at the base image's version (the extras would otherwise be free to upgrade it);
# the GPU torch wheel is chosen explicitly because the build has no GPU for `auto` to detect.
RUN pip install --no-cache-dir uv \
    && uv pip install --python "$(command -v python)" --no-cache --torch-backend=${TORCH_BACKEND} \
        ".[tracking,distributed,medical]" "ray[train]==${RAY_VERSION}"

# PYTHONPATH: the Ray job driver does not start in /app, and `angku run` must still find
# the `experiments` package to discover models/losses/datasets.
ENV PYTHONPATH=/app
ENV MLOPS_GIT_COMMIT=${MLOPS_GIT_COMMIT}
# Same default as mlops.cli.CUDA_ALLOC_DEFAULT, for every process in the pod (Ray workers on other
# nodes do not inherit the driver's environment). Allocation layout only, results are unchanged.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
