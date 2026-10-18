#!/usr/bin/env bash
# Build (and optionally push) the training image from a clean checkout of HEAD.
#
#   scripts/build-image.sh IMAGE[:TAG] [--push] [--cpu]
#
# Refuses when code differs from HEAD (same scope as the environment probe: mlops/,
# experiments/ except experiments/configs/, pyproject.toml, plus Dockerfile and
# .dockerignore; untracked files count), because MLOPS_GIT_COMMIT=HEAD is baked into the
# image and a run inside it trusts that value. With --push, prints the repo digest to pin
# as environment.image_digest. --cpu builds on the CPU flavour of the Ray base image with the
# CPU torch wheel (kind on a machine without a GPU). Exit 2 on usage errors or a dirty tree.
set -euo pipefail

usage() { echo "usage: $0 IMAGE[:TAG] [--push] [--cpu]" >&2; exit 2; }

image="" push=0 cpu=()
for arg in "$@"; do
    case "$arg" in
        --push) push=1 ;;
        --cpu) cpu=(--build-arg "RAY_FLAVOR=" --build-arg "TORCH_BACKEND=cpu") ;;
        -*) usage ;;
        *) [ -z "$image" ] || usage; image="$arg" ;;
    esac
done
[ -n "$image" ] || usage

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

dirty="$(git status --porcelain --untracked-files=all -- \
    mlops experiments pyproject.toml Dockerfile .dockerignore ':(exclude)experiments/configs')"
if [ -n "$dirty" ]; then
    echo "refusing to build: these files differ from HEAD; commit them first:" >&2
    echo "$dirty" | sed 's/^/  /' >&2
    exit 2
fi

commit="$(git rev-parse HEAD)"
docker build --build-arg "MLOPS_GIT_COMMIT=$commit" ${cpu[@]+"${cpu[@]}"} -t "$image" "$root"
echo "built $image"
echo "code.commit: $commit"

if [ "$push" -eq 0 ]; then
    echo "not pushed: a registry digest only exists after a push; rerun with --push to get image_digest"
    exit 0
fi

docker push "$image"
ref="$(docker inspect --format '{{index .RepoDigests 0}}' "$image")"
echo "pin these in experiment.yaml:"
echo "image: $image"
echo "image_digest: ${ref##*@}"
