# Setup

## Clone

This repository uses a pinned upstream submodule:

```bash
git clone --recurse-submodules <your-repository-url>
cd perception_plus_plus
git submodule update --init --recursive
python3 scripts/check_upstream.py
```

The required revision is also recorded in
`assets/model_manifests/upstream.json`. Do not develop directly inside the
submodule. Store compatibility patches in `patches/foundationpose_plus_plus/`.

## Models and mesh

Large files are not committed. Before distribution, replace each all-zero
template digest in `assets/model_manifests/models.json` with the SHA-256 of the
exact approved file. Then place files beneath `models/` or copy them from a
private source directory:

```bash
sha256sum /path/to/each/model
python3 scripts/fetch_models.py --source-root /path/to/model-tree
```

Add the cup mesh as `assets/meshes/cup.obj` and set
`mesh_scale_to_meters`. Use `1.0` for metre meshes, `0.001` for millimetres,
and `0.01` for centimetres.

## Local tests

```bash
python3 -m pip install -e '.[test]'
bash scripts/run_tests.sh
colcon build --base-paths ros_ws/src --symlink-install
source install/setup.bash
colcon test --base-paths ros_ws/src
colcon test-result
```

## Containers

Build the distribution you need:

```bash
docker build -f docker/humble/Dockerfile -t perception-plus-plus:humble .
docker build -f docker/jazzy/Dockerfile -t perception-plus-plus:jazzy .
```

The Dockerfiles install the CUDA toolkit matching their locked PyTorch wheel
and build the model-based FoundationPose extensions for
compute capabilities 8.6 (RTX 3090) and 8.9 (RTX 4070 Laptop). The NVIDIA driver remains a host dependency. The build
downloads PyTorch3D and NVDiffRast source, so archive or mirror those sources
before producing an air-gapped release. Validate with:

```bash
bash scripts/container_smoke.sh humble
bash scripts/container_smoke.sh jazzy
```

## Run without a camera

An NPZ replay contains `rgb[N,H,W,3]`, metric `depth[N,H,W]`,
`timestamps_ns[N]`, and `K[3,3]`.

```bash
python3 scripts/replay_rgbd.py recording.npz \
  --mesh assets/meshes/cup.obj \
  --yolo models/yolo/yolo11n.pt
```

Run the readiness report last:

```bash
python3 scripts/pre_camera_check.py
```

The JSON result is written to `reports/pre_camera_readiness.json`.
