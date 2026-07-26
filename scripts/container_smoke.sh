#!/usr/bin/env bash
set -euo pipefail
distro="${1:-jazzy}"
if [[ $# -gt 1 || "$distro" != "humble" && "$distro" != "jazzy" ]]; then
  echo "usage: $0 [humble|jazzy]   # defaults to jazzy, this host's distribution" >&2
  exit 2
fi
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -f "$project_dir/docker/$distro/Dockerfile" -t "perception-plus-plus:$distro" "$project_dir"
docker run --rm --gpus all \
  -v "$project_dir/models:/workspace/perception_plus_plus/models:ro" \
  "perception-plus-plus:$distro" bash -lc \
  "source /opt/ros/$distro/setup.bash && \
   source /opt/perception_plus_plus/setup.bash && \
   python3 scripts/bootstrap_models.py --verify-only && \
   python3 -c 'import numpy; from cv_bridge import CvBridge; from sensor_msgs.msg import Image; assert CvBridge().imgmsg_to_cv2(Image(height=2, width=2, encoding=\"rgb8\", step=6, data=bytes(12)), \"rgb8\").shape == (2, 2, 3); print(\"cv_bridge works with numpy\", numpy.__version__)' && \
   python3 -c 'import sys, torch; assert torch.cuda.is_available(); sys.path[:0] = [\"external/foundationpose_plus_plus\", \"external/foundationpose_plus_plus/src\", \"external/foundationpose_plus_plus/FoundationPose\", \"external/foundationpose_plus_plus/Cutie\"]; import nvdiffrast.torch; from estimater import FoundationPose; from cutie.utils.get_default_model import get_default_model; print(torch.cuda.get_device_name())' && \
   ros2 interface show perception_plus_plus_msgs/msg/TrackingStatus"
