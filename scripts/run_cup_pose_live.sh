#!/bin/bash
# 라이브 /cup_pose 파이프라인 브링업 (vision-3090)
# 체인: RealSense ROS → FP++ 컨테이너(cup_tracking_node) → cup_pose_relay → /cup_pose(base)
#
# ★선행(사용자, 별도 터미널 — 카메라 점유): RealSense ROS 노드
#   source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=126
#   ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
# (최초 1회 설치: sudo apt install -y ros-humble-realsense2-camera)
#
# 이 스크립트는 FP++ 컨테이너 + relay 를 띄운다. 목은 교정 자세(pan-90/tilt280)로 고정돼 있어야
# global_camera_extrinsics.yaml 이 유효하다.
set -e
DOMAIN=126
PPP=/home/usr/rl_ws/perception_plus_plus
SIM=/home/usr/rl_ws/sim2real

# 0) 세그멘테이션 YOLO 가중치 확보 (detector 가 mask 를 요구 → -seg 필수. det-only 는 실패)
if [ ! -f "$PPP/models/yolo/yolov8m-seg.pt" ]; then
  "$PPP/.venv/bin/python" - <<PY
from ultralytics import YOLO; import shutil,os
m=YOLO("yolov8m-seg.pt"); src=getattr(m,"ckpt_path",None) or "yolov8m-seg.pt"
shutil.copy(src, "$PPP/models/yolo/yolov8m-seg.pt"); print("downloaded yolov8m-seg.pt")
PY
fi

# 1) wandb stub — 컨테이너 wandb(protobuf 깨짐)를 ultralytics 가 import 시 크래시.
#    ImportError 로 바꿔 ultralytics 가 안전하게 스킵하게 한다. FP 는 wandb 미사용.
mkdir -p /tmp/wandb_stub
echo 'raise ImportError("wandb disabled in container (protobuf mismatch)")' > /tmp/wandb_stub/wandb.py

# 2) params — 세그 YOLO(mask) + 컵 class 41. (컨테이너 baked yolo11n 은 구버전 ultralytics 8.0.120 이 못 읽음)
cat > /tmp/cup_tracking_v8.yaml <<YAML
cup_tracking:
  ros__parameters:
    rgb_topic: /camera/camera/color/image_raw
    depth_topic: /camera/camera/aligned_depth_to_color/image_raw
    camera_info_topic: /camera/camera/color/camera_info
    pose_topic: /perception_plus_plus/cup/pose
    status_topic: /perception_plus_plus/cup/tracking_status
    child_frame_id: cup
    mesh_path: assets/meshes/cup.obj
    mesh_scale_to_meters: 1.0
    yolo_weights: models/yolo/yolov8m-seg.pt
    cup_class_id: 41
    yolo_confidence: 0.5
    tracking_config: config/cup_tracking.yaml
    sync_slop_seconds: 0.04
    sync_queue_size: 10
YAML

# 3) FP++ 컨테이너 노드
#    --network host: 호스트 DDS(도메인126) 공유 / --ipc=host: FastDDS 공유메모리 데이터경로(없으면 discovery만 되고 데이터 0)
#    PYTHONPATH: core 는 설치본이 아니라 소스라 경로 추가 필요
docker rm -f fpp_cup >/dev/null 2>&1 || true
docker run -d --name fpp_cup --network host --ipc=host --gpus all -e ROS_DOMAIN_ID=$DOMAIN \
  -v "$PPP/models/yolo":/workspace/perception_plus_plus/models/yolo:ro \
  -v /tmp/cup_tracking_v8.yaml:/tmp/cup_tracking_v8.yaml:ro \
  -v /tmp/wandb_stub:/tmp/wandb_stub:ro \
  perception-plus-plus:humble bash -lc '
    source /opt/ros/humble/setup.bash
    source /opt/perception_plus_plus/setup.bash
    export PYTHONPATH=/tmp/wandb_stub:/workspace/perception_plus_plus:$PYTHONPATH
    cd /workspace/perception_plus_plus
    exec ros2 launch perception_plus_plus_ros cup_tracking.launch.py parameters_file:=/tmp/cup_tracking_v8.yaml'
echo "FP++ container up: $(docker ps -q -f name=fpp_cup)  (로그: docker logs -f fpp_cup)"

# 4) relay: 카메라프레임 컵 6D pose → base 프레임 /cup_pose (detached)
cat > /tmp/run_relay.sh <<SH
#!/bin/bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=$DOMAIN
cd $SIM/scripts
exec python3 cup_pose_relay.py --in-type posestamped \
  --in-topic /perception_plus_plus/cup/pose --out-topic /cup_pose \
  --extrinsics ../config/global_camera_extrinsics.yaml
SH
chmod +x /tmp/run_relay.sh
pkill -f cup_pose_relay.py 2>/dev/null || true; sleep 1
setsid /tmp/run_relay.sh </dev/null >/tmp/relay.log 2>&1 &
echo "relay up (로그: tail -f /tmp/relay.log)"
echo "확인: ROS_DOMAIN_ID=$DOMAIN ros2 topic echo /cup_pose   (~8Hz, base 프레임 6D)"
