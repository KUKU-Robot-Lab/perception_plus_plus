# perception_plus_plus

FoundationPose++-based single-cup 6D tracking for ROS 2 Humble and Jazzy.
The system is designed for aligned RealSense D435i RGB-D input and keeps all
tracking logic in a ROS-independent Python core.

See `docs/setup.md` for installation and
`docs/superpowers/specs/2026-07-24-foundationpose-plus-plus-system-design.md`
for the architecture.

## Quick verification

```bash
git submodule update --init --recursive
bash scripts/run_tests.sh
python3 scripts/pre_camera_check.py
```

The readiness command remains `NOT_READY` until real model digests/files,
CUDA/FP++ imports, replay data, and the required target containers have been
validated. See `docs/validation.md` for the exact contract.
