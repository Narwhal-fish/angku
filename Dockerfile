# Training image for `mlops run` locally in a container and for KubeRay RayJobs.
#
# Build it with scripts/build-image.sh, never by hand: the script refuses a dirty checkout
# and passes MLOPS_GIT_COMMIT, which is how a run inside this image verifies code.commit
# (there is no .git in the image). Pin the pushed digest as environment.image_digest.
#
# RAY_VERSION must equal mlops.adapters.kuberay.RAY_VERSION (tests/test_image.py checks it).
# TORCH_BACKEND must match the CUDA of the base image.
ARG RAY_VERSION=2.58.0
FROM rayproject/ray:${RAY_VERSION}-py312-gpu

ARG RAY_VERSION
ARG TORCH_BACKEND=cu128
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

# PYTHONPATH: the Ray job driver does not start in /app, and `mlops run` must still find
# the `experiments` package to discover models/losses/datasets.
ENV PYTHONPATH=/app
ENV MLOPS_GIT_COMMIT=${MLOPS_GIT_COMMIT}
