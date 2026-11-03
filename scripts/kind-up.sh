#!/usr/bin/env bash
# A local Kubernetes for checking the KubeRay path on a machine without a GPU or a cluster:
# kind + a registry the cluster pulls from by digest + the KubeRay operator + PVs for the run
# bundles and Ray Train's checkpoints. Idempotent: rerun it to repair a half-created setup.
#
#   scripts/kind-up.sh                      then scripts/kind-verify.sh
#   scripts/kind-down.sh                    removes cluster + registry, keeps .kind/
#
# The PVs are host directories (.kind/artifacts, .kind/ray-storage) mounted into the kind node,
# so the bundles a RayJob writes are readable on this machine:
#   angku ui --artifacts .kind/artifacts --kubectl "kubectl --context kind-mlops"
# A single kind node stands in for RWX storage; it says nothing about RWX across real nodes.
set -euo pipefail

CLUSTER="${MLOPS_KIND_CLUSTER:-mlops}"
NODE_IMAGE="${MLOPS_KIND_NODE_IMAGE:-kindest/node:v1.35.0}"
REGISTRY=mlops-registry
REGISTRY_PORT=5001  # images are pushed as localhost:5001/...; the node resolves that to the registry
KUBERAY_VERSION=v1.7.1
CONTEXT="kind-$CLUSTER"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state="$root/.kind"
mkdir -p "$state/artifacts" "$state/ray-storage"
chmod 0777 "$state/artifacts" "$state/ray-storage"  # the Ray image runs as uid 1000 (ray)

if ! timeout 15 docker info >/dev/null 2>&1; then
    echo "docker does not answer (Docker Desktop paused or in resource saver mode?); start it and retry" >&2
    exit 2
fi

echo "== registry $REGISTRY on 127.0.0.1:$REGISTRY_PORT"
if [ "$(docker inspect -f '{{.State.Running}}' "$REGISTRY" 2>/dev/null || true)" != true ]; then
    docker rm -f "$REGISTRY" >/dev/null 2>&1 || true
    docker run -d --restart=always -p "127.0.0.1:$REGISTRY_PORT:5000" --name "$REGISTRY" registry:2 >/dev/null
fi

echo "== kind cluster $CLUSTER ($NODE_IMAGE)"
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
    kind create cluster --name "$CLUSTER" --image "$NODE_IMAGE" --wait 120s --config=- <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
- |-
  [plugins."io.containerd.cri.v1.images".registry]
    config_path = "/etc/containerd/certs.d"
nodes:
- role: control-plane
  extraMounts:
  - hostPath: $state/artifacts
    containerPath: /mlops/artifacts
  - hostPath: $state/ray-storage
    containerPath: /mlops/ray-storage
EOF
fi

# localhost:5001 inside the node means the node itself; point it at the registry container.
docker network connect kind "$REGISTRY" 2>/dev/null || true
for node in $(kind get nodes --name "$CLUSTER"); do
    docker exec "$node" mkdir -p "/etc/containerd/certs.d/localhost:$REGISTRY_PORT"
    printf '[host."http://%s:5000"]\n' "$REGISTRY" \
        | docker exec -i "$node" cp /dev/stdin "/etc/containerd/certs.d/localhost:$REGISTRY_PORT/hosts.toml"
done

echo "== KubeRay operator $KUBERAY_VERSION"
kubectl --context "$CONTEXT" apply --server-side \
    -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=$KUBERAY_VERSION"
kubectl --context "$CONTEXT" rollout status deployment/kuberay-operator --timeout=300s

echo "== PVs + PVCs: mlops-artifacts, mlops-ray-storage"
for name in artifacts ray-storage; do
    kubectl --context "$CONTEXT" apply -f - <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mlops-$name
spec:
  capacity: {storage: 20Gi}
  accessModes: [ReadWriteMany]
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ""
  hostPath: {path: /mlops/$name, type: Directory}
  claimRef: {namespace: default, name: mlops-$name}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlops-$name
  namespace: default
spec:
  accessModes: [ReadWriteMany]
  storageClassName: ""
  volumeName: mlops-$name
  resources: {requests: {storage: 20Gi}}
EOF
done

echo
echo "ready: context $CONTEXT, registry localhost:$REGISTRY_PORT, bundles in $state/artifacts"
