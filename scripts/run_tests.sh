#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python3 -m pytest -q
python3 -m compileall -q perception_plus_plus_core \
  ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros

