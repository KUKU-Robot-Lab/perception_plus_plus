# perception_plus_plus

FoundationPose++-based single-cup 6D tracking for ROS 2 Humble and Jazzy.
The system is designed for aligned RealSense D435i RGB-D input and keeps all
tracking logic in a ROS-independent Python core.

See `docs/setup.md` for installation and
`docs/superpowers/specs/2026-07-24-foundationpose-plus-plus-system-design.md`
for the architecture.

## Humble container (validated target)

ROS 2 **Humble** is the validated, supported target. To reproduce the GPU
container on any Humble host with an NVIDIA GPU:

```bash
git clone --recurse-submodules <repo-url> && cd perception_plus_plus
git submodule update --init --recursive
# place the 7 licensed model files under models/ and verify their digests
python3 scripts/fetch_models.py
# build (arch is set for RTX 3090/4070 = compute 8.6;8.9)
docker build -f docker/humble/Dockerfile -t perception-plus-plus:humble .
```

See `docs/vision-3090-setup-state.md` for the exact validated state, the two
build fixes it required, and how to run it. (A Jazzy Dockerfile exists but is
not validated.)

## Quick verification

```bash
git submodule update --init --recursive
bash scripts/run_tests.sh
python3 scripts/pre_camera_check.py
```

The readiness command remains `NOT_READY` until real model digests/files,
CUDA/FP++ imports, replay data, and the required target containers have been
validated. See `docs/validation.md` for the exact contract.
