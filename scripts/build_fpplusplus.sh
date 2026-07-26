#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="$project_dir/external/foundationpose_plus_plus"
test -f "$upstream_dir/FoundationPose/build_all_conda.sh"
test -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;8.9}"
# Build pytorch3d first, with the base setuptools still in place -- this is the
# combination the image was validated with.
python3 -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git"
# nvdiffrast v0.4 declares its name via PEP 621 [project]; ros:humble ships
# setuptools 59 which ignores it under --no-build-isolation and installs the
# package as UNKNOWN (import nvdiffrast then fails). Upgrade the build backend
# before installing nvdiffrast so the name resolves.
python3 -m pip install -U "setuptools>=64" wheel ninja
python3 -m pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast.git"
python3 -m pip install -r "$upstream_dir/FoundationPose/requirements.txt"
python3 -m pip install -e "$upstream_dir/Cutie"
bash "$upstream_dir/FoundationPose/build_all_conda.sh"

