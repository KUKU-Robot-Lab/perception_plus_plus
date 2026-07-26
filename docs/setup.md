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

Large files are not committed. Install the official public model set and
create its exact local SHA-256 lock:

```bash
python3 -m pip install -e '.[models]'
python3 scripts/bootstrap_models.py
python3 scripts/bootstrap_models.py --verify-only
```

Move the same model bytes to Humble or Jazzy through an external drive:

```bash
python3 scripts/export_model_bundle.py \
  --output /media/$USER/MODELS/perception-plus-plus-models-v1

python3 scripts/import_model_bundle.py \
  --bundle /media/$USER/MODELS/perception-plus-plus-models-v1
```

Export and import verify every file against `models/models.lock.json`.

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

Humble and Jazzy are built and validated as separate images; each has its own
Dockerfile, CUDA toolkit, and pinned requirements lock. Build only the one that
matches the target host. This workstation runs Jazzy:

```bash
docker build -f docker/jazzy/Dockerfile -t perception-plus-plus:jazzy .   # Ubuntu 24.04 host
docker build -f docker/humble/Dockerfile -t perception-plus-plus:humble . # Ubuntu 22.04 host
```

Each Dockerfile installs the CUDA toolkit matching its locked PyTorch wheel
and builds the model-based FoundationPose extensions with compute capability
8.9 for the RTX 4070. The NVIDIA driver remains a host dependency. The build
downloads PyTorch3D and NVDiffRast source, so archive or mirror those sources
before producing an air-gapped release. ROS sources are copied after the CUDA
and pip stages, so a ROS-only edit rebuilds just the colcon layer. Validate
the image you built:

```bash
bash scripts/container_smoke.sh jazzy
bash scripts/container_smoke.sh humble
```

Both containers mount the host `models/` tree read-only; checkpoint bytes are
not duplicated inside the images.

## Run without a camera

An NPZ replay contains `rgb[N,H,W,3]`, metric `depth[N,H,W]`,
`timestamps_ns[N]`, and `K[3,3]`.

```bash
python3 scripts/replay_rgbd.py recording.npz \
  --mesh assets/meshes/cup.obj \
  --yolo models/yolo/yolo11n-seg.pt
```

Run the readiness report last:

```bash
python3 scripts/pre_camera_check.py
```

The JSON result is written to `reports/pre_camera_readiness.json`.
