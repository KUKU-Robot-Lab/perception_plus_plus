#!/bin/bash
# 라이브 /cup_pose 파이프라인 브링업 (vision-3090) — baked 이미지 perception-plus-plus:humble-cup 사용
# 체인: RealSense ROS → FP++ cup_tracking_node → cup_pose_relay → /cup_pose(base 6D)
#
# 5개 컨테이너 수정은 이미지에 baked: yolov8m-seg(mask) · wandb-stub(protobuf 우회) ·
#   PYTHONPATH(core) · seg params · (베이스: perception-plus-plus:humble). 재빌드는 /tmp/cupbake/Dockerfile.
# 런타임 필수 플래그만 유지: --network host --ipc=host --gpus all -e ROS_DOMAIN_ID.
#
# ★선행(사용자, 별도 터미널 — 카메라 점유): RealSense ROS 노드
#   source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=126
#   ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
#   (최초 1회 설치: sudo apt install -y ros-humble-realsense2-camera)
# 목은 교정 자세(pan-90 / tilt280) 고정 유지 → global_camera_extrinsics.yaml 유효.
set -e
DOMAIN=126
SIM=/home/usr/rl_ws/sim2real

# 1) FP++ 컨테이너 (baked). --ipc=host: FastDDS 공유메모리 데이터경로(없으면 discovery만·데이터0) / --network host: DDS 공유
docker rm -f fpp_cup >/dev/null 2>&1 || true
docker run -d --name fpp_cup --network host --ipc=host --gpus all -e ROS_DOMAIN_ID=$DOMAIN \
  perception-plus-plus:humble-cup bash -lc '
    source /opt/ros/humble/setup.bash
    source /opt/perception_plus_plus/setup.bash
    cd /workspace/perception_plus_plus
    exec ros2 launch perception_plus_plus_ros cup_tracking.launch.py'
echo "FP++ container up: $(docker ps -q -f name=fpp_cup)  (로그: docker logs -f fpp_cup)"

# 2) relay: 카메라프레임 컵 6D pose → base 프레임 /cup_pose (detached, ssh 종료에도 유지)
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

# 종료: docker rm -f fpp_cup ; pkill -f cup_pose_relay.py
