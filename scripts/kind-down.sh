#!/usr/bin/env bash
# Remove what scripts/kind-up.sh created: the kind cluster and the registry container.
# .kind/ (the bundles and Ray checkpoints the cluster wrote) stays on disk; delete it yourself.
set -euo pipefail

CLUSTER="${ANGKU_KIND_CLUSTER:-angku}"
REGISTRY=angku-registry

kind delete cluster --name "$CLUSTER"
docker rm -f "$REGISTRY" >/dev/null 2>&1 || true
echo "removed cluster $CLUSTER and registry $REGISTRY; .kind/ kept"
