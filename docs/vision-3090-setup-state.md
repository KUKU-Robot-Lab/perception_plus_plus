# vision-3090 Setup State

Setup status for `perception_plus_plus` on **vision-3090**
(RTX 3090, Ampere, compute capability 8.6). Updated 2026-07-26.

## Status: container built and functionally validated

The Humble target container `perception-plus-plus:humble` (33.7 GB) is built
and passes the full dependency + FoundationPose++ smoke on the 3090 GPU.

### Verified inside the container
- torch 2.4.1+cu121, `cuda.is_available()` True, RTX 3090
- nvdiffrast 0.4.0, pytorch3d 0.7.9 (`_C` CUDA ext), cutie
- `ros2 interface show perception_plus_plus_msgs/msg/TrackingStatus`
- `from FoundationPose.estimater import FoundationPose, PoseRefinePredictor,
  ScorePredictor, dr` -> OK (Warp initialised on sm_86)
- `pre_camera_check.py` inside the container: models, cpu_tests, ros_colcon,
  cuda_torch, cup_mesh all PASS.

### Two build defects were found and fixed (reproducible, in source)
1. **nvdiffrast installed as `UNKNOWN`** -> `import nvdiffrast` failed.
   Cause: `ros:humble` ships setuptools 59, which ignores nvdiffrast v0.4's
   PEP 621 `[project].name` under `--no-build-isolation`.
   Fix: `scripts/build_fpplusplus.sh` now upgrades `setuptools>=64 wheel ninja`
   before the nvdiffrast install.
2. **`ros2` could not resolve `perception_plus_plus_msgs`** (only `_ros` on
   `AMENT_PREFIX_PATH`). Fix: `docker/{humble,jazzy}/Dockerfile` colcon build
   now uses `--merge-install`.

A clean rebuild from these fixed sources is correct end to end. The current
image was produced by building the base then applying a thin patch layer
(`/tmp/ppp_patch/Dockerfile`) to avoid recompiling pytorch3d; `:humble` and
`:humble-fix` are the same image.

## Host-side prerequisites (done)
- Submodule `external/foundationpose_plus_plus` pinned at `58aa715`.
- All 7 models under `models/` with real SHA-256 in
  `assets/model_manifests/models.json` (sourced from
  `~/rl_ws/repo/FoundationPose-plus-plus/` + ultralytics `yolo11n.pt`).
- Container arch `TORCH_CUDA_ARCH_LIST="8.6;8.9"` (3090 + 4070).
- GPU driver: 595.84 (kernel module and userspace match after reboot).
- A `.venv` exists for host-side CPU tests; the container is the real runtime.

## Remaining to reach READY (data-gated, not setup)
- `fpplusplus_smoke` needs a recorded initialisation frame: an NPZ with
  `rgb[N,H,W,3]`, metric `depth[N,H,W]`, `timestamps_ns[N]`, `K[3,3]`.
  Capture one from the D435 now attached to vision-3090, or supply a replay
  recording, then run `scripts/fpplusplus_smoke.py --npz <frame.npz>
  --mesh assets/meshes/cup.obj` inside the container.
- `upstream_revision` reports FAIL inside the container only because `.git`
  is excluded from the image; it PASSes on the host where the pin is verified.
- Camera extrinsics `T_base_cam` in the sim2real config remain a PLACEHOLDER
  (calibrate once the global camera is mounted).

## Run
Start the container with the repo bind-mounted, then source ROS and the overlay:

    docker run --rm -it --gpus all \
      -v "$PWD":/workspace/perception_plus_plus \
      perception-plus-plus:humble bash

Inside: `source /opt/ros/humble/setup.bash` then
`source /opt/perception_plus_plus/setup.bash`.
