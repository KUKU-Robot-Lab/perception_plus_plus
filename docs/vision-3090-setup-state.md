# vision-3090 Setup State

Setup status for `perception_plus_plus` on **vision-3090**
(RTX 3090, Ampere, compute capability 8.6). Updated 2026-07-27.

## Status: validated end to end on a live D435 frame

The Humble container `perception-plus-plus:humble` runs the full
FoundationPose++ pipeline on a real RealSense D435 capture. A one-frame
`fpplusplus_smoke` returns a valid 6D cup pose on the 3090 GPU.

`pre_camera_check.py --smoke-npz <frame> --mesh assets/meshes/cup.obj`
inside the container reports every capability PASS
(models, cpu_tests, ros_colcon, cuda_torch, cup_mesh, **fpplusplus_smoke**).
The overall line stays `NOT_READY` only because `upstream_revision` cannot run
without `.git`, which is intentionally excluded from the image; the pin
(`58aa715`) is verified on the host, where that check PASSes.

### Verified inside the container
- torch 2.4.1+cu121, `cuda.is_available()` True, RTX 3090
- nvdiffrast 0.4.0, pytorch3d 0.7.9 (`_C`), cutie, `mycpp.cluster_poses`
- `ros2 interface show perception_plus_plus_msgs/msg/TrackingStatus`
- `FoundationPose.estimater` import + live `register()` -> 6D pose

## Three build defects were found and fixed (reproducible, in source)
1. **nvdiffrast installed as `UNKNOWN`** (import failed). `ros:humble` ships
   setuptools 59, which ignores nvdiffrast v0.4's PEP 621 `[project].name`
   under `--no-build-isolation`. Fix: `scripts/build_fpplusplus.sh` builds
   pytorch3d first, then upgrades `setuptools>=64 wheel ninja` before nvdiffrast.
2. **`ros2` could not resolve `perception_plus_plus_msgs`**. Fix:
   `docker/{humble,jazzy}/Dockerfile` colcon uses `--merge-install`.
3. **`mycpp` (FoundationPose C++ ext) never built -> `mycpp.cluster_poses`
   was `None`, FP init crashed**. Its CMake needs Boost + pybind11, which the
   apt layer omitted. Fix: add `libboost-system-dev
   libboost-program-options-dev pybind11-dev` to the Dockerfile apt install so
   `build_all_conda.sh` builds `mycpp/build/*.so`.

The current local image was assembled base + thin patch layers to avoid
recompiling pytorch3d each time; the committed sources produce the same image
from a clean `docker build` (nvdiffrast + merge-install verified clean; boost +
mycpp build verified in-container).

## Host-side prerequisites (done)
- Submodule `external/foundationpose_plus_plus` pinned at `58aa715`.
- 7 models under `models/` with real SHA-256 in
  `assets/model_manifests/models.json`.
- Container arch `TORCH_CUDA_ARCH_LIST="8.6;8.9"` (3090 + 4070).
- GPU driver 595.84 (kernel module and userspace match).
- `.venv` for host CPU tests + RealSense capture (pyrealsense2, ultralytics).

## Live capture -> smoke (how the validation frame was made)
1. Host: `python /tmp/capture_frame.py` (pyrealsense2, aligned RGB-D + K)
   -> raw npz.
2. Host: YOLO-seg (`yolo11n-seg`, modern ultralytics in `.venv`) segments the
   cup (COCO class 41) -> mask; assemble `{rgb, depth, mask, K}` npz.
   NOTE: the container's FP++ env pins an old ultralytics with a broken wandb,
   so mask generation is done on the host, not in the container.
3. Container: `scripts/fpplusplus_smoke.py --npz <frame> --mesh
   assets/meshes/cup.obj --mesh-scale 1.0` -> 6D pose.

The manifest's `yolo11n.pt` is detection-only; a production mask source
(seg model or bbox+depth) is still a design decision. Camera extrinsics
`T_base_cam` in the sim2real config remain a PLACEHOLDER (calibrate once the
global camera is mounted).

## Run
Start the container with the repo bind-mounted, then source ROS and the overlay:

    docker run --rm -it --gpus all \
      -v "$PWD":/workspace/perception_plus_plus \
      perception-plus-plus:humble bash

Inside: `source /opt/ros/humble/setup.bash` then
`source /opt/perception_plus_plus/setup.bash`.
