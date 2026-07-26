#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="$project_dir/external/foundationpose_plus_plus"
test -f "$upstream_dir/FoundationPose/build_all_conda.sh"
test -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
stage="${1:-all}"

case "$stage" in
  nvdiffrast)
    python3 -m pip install --no-build-isolation \
      "git+https://github.com/NVlabs/nvdiffrast.git"
    ;;
  pytorch3d)
    python3 -m pip install --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git"
    ;;
  requirements)
    python3 -m pip install \
      -r "$project_dir/docker/foundationpose-runtime.requirements"
    # Cutie's published metadata includes its GUI, demo and training stack.
    # The ROS runtime uses only the model and inference modules.
    python3 -m pip install --no-deps -e "$upstream_dir/Cutie"
    ;;
  extensions)
    bash "$upstream_dir/FoundationPose/build_all_conda.sh"
    ;;
  all)
    for item in nvdiffrast pytorch3d requirements extensions; do
      bash "$0" "$item"
    done
    ;;
  *)
    echo "usage: $0 [nvdiffrast|pytorch3d|requirements|extensions|all]" >&2
    exit 2
    ;;
esac
