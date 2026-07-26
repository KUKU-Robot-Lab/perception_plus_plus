# perception_plus_plus

FoundationPose++-based single-cup 6D tracking for ROS 2, packaged as separate
Humble and Jazzy images. This workstation runs Jazzy.
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

Install and lock the official public model set once:

```bash
python3 -m pip install -e '.[models]'
python3 scripts/bootstrap_models.py
```

After building and sourcing the ROS workspace, start the validated D435i
profile and tracker together:

```bash
ros2 launch perception_plus_plus_ros realsense_cup_tracking.launch.py \
  project_root:="$PWD"
```

The readiness command remains `NOT_READY` until the model lock, CUDA/FP++
imports, live initialization, and required target containers are validated.
