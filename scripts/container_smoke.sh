#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || "$1" != "humble" && "$1" != "jazzy" ]]; then
  echo "usage: $0 humble|jazzy" >&2
  exit 2
fi
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -f "$project_dir/docker/$1/Dockerfile" -t "perception-plus-plus:$1" "$project_dir"
docker run --rm --gpus all "perception-plus-plus:$1" bash -lc \
  "source /opt/ros/$1/setup.bash && source /opt/perception_plus_plus/setup.bash && python3 -c 'import torch; assert torch.cuda.is_available()' && ros2 interface show perception_plus_plus_msgs/msg/TrackingStatus"

