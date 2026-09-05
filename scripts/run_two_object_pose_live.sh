#!/bin/bash
# 2물체(우 cup_big + 좌 shaker) 라이브 파이프라인 — 2026-09-02 (Step 1)
# 체인: RealSense ROS → FP++ ×2 (fpp_cup, fpp_shaker) → relay ×2
#   → /cup_pose (cup_big, base 6D) + /shaker_pose (shaker, base 6D)
#
# ★선행(사용자, 별도 터미널 — 카메라 점유): RealSense ROS 노드
#   source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=126
#   ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
# ★목 자세는 최신 extrinsics 기준(09.01 hand-eye): pan 0 / tilt -20 (head_home.yaml)
# ★shaker 는 COCO 클래스 매핑(bottle=39) 임시 — 인식 실패 시 shaker_tracking.yaml 의
#   cup_class_id 를 75(vase)/41(cup) 로 바꿔 재기동. 그래도 안 되면 파인튜닝 필요.
set -e
DOMAIN=126
SIM=/home/usr/rl_ws/sim2real
PPP=/home/usr/rl_ws/perception_plus_plus

# 1) cup 컨테이너 (기존과 동일 — baked 설정 그대로)
docker rm -f fpp_cup >/dev/null 2>&1 || true
docker run -d --name fpp_cup --network host --ipc=host --gpus all -e ROS_DOMAIN_ID=$DOMAIN \
  -v $PPP/perception_plus_plus_core/detection/yolo.py:/workspace/perception_plus_plus/perception_plus_plus_core/detection/yolo.py:ro \
  -v $PPP/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:/workspace/perception_plus_plus/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:ro \
  -v $PPP/ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/node.py:/opt/perception_plus_plus/lib/python3.10/site-packages/perception_plus_plus_ros/node.py:ro \
  -v $PPP/ros_ws/src/perception_plus_plus_ros/config/cup_tracking_bright.yaml:/opt/perception_plus_plus/share/perception_plus_plus_ros/config/cup_tracking.yaml:ro \
  perception-plus-plus:humble-cup bash -lc '
    source /opt/ros/humble/setup.bash
    source /opt/perception_plus_plus/setup.bash
    cd /workspace/perception_plus_plus
    exec ros2 launch perception_plus_plus_ros cup_tracking.launch.py'
echo "fpp_cup up"

# 2) shaker 컨테이너 — 메시·설정을 호스트에서 마운트, parameters_file 로 주입
#    (launch 수정 0줄: yaml 최상위 키를 노드명 cup_tracking 에 맞춰둠. 노드명 중복은
#     DDS 허용 — 문제되면 launch 의 name 을 인자화한다)
docker rm -f fpp_shaker >/dev/null 2>&1 || true
docker run -d --name fpp_shaker --network host --ipc=host --gpus all -e ROS_DOMAIN_ID=$DOMAIN \
  -v $PPP/perception_plus_plus_core/detection/yolo.py:/workspace/perception_plus_plus/perception_plus_plus_core/detection/yolo.py:ro \
  -v $PPP/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:/workspace/perception_plus_plus/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:ro \
  -v $PPP/ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/node.py:/opt/perception_plus_plus/lib/python3.10/site-packages/perception_plus_plus_ros/node.py:ro \
  -v $PPP/assets/meshes:/workspace/perception_plus_plus/assets/meshes:ro \
  -v $PPP/ros_ws/src/perception_plus_plus_ros/config/shaker_tracking.yaml:/opt/params/shaker_tracking.yaml:ro \
  perception-plus-plus:humble-cup bash -lc '
    source /opt/ros/humble/setup.bash
    source /opt/perception_plus_plus/setup.bash
    cd /workspace/perception_plus_plus
    exec ros2 launch perception_plus_plus_ros cup_tracking.launch.py parameters_file:=/opt/params/shaker_tracking.yaml'
echo "fpp_shaker up"

# 3) relay ×2 (detached)
pkill -f cup_pose_relay.py 2>/dev/null || true; sleep 1
cat > /tmp/run_relay_cup.sh <<SH
#!/bin/bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=$DOMAIN
cd $SIM/scripts
exec python3 cup_pose_relay.py --in-type posestamped \
  --in-topic /perception_plus_plus/cup/pose --out-topic /cup_pose \
  --extrinsics ../config/global_camera_extrinsics.yaml
SH
cat > /tmp/run_relay_shaker.sh <<SH
#!/bin/bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=$DOMAIN
cd $SIM/scripts
exec python3 cup_pose_relay.py --in-type posestamped \
  --in-topic /perception_plus_plus/shaker/pose --out-topic /shaker_pose \
  --extrinsics ../config/global_camera_extrinsics_shaker.yaml
SH
chmod +x /tmp/run_relay_cup.sh /tmp/run_relay_shaker.sh
setsid /tmp/run_relay_cup.sh </dev/null >/tmp/relay_cup.log 2>&1 &
setsid /tmp/run_relay_shaker.sh </dev/null >/tmp/relay_shaker.log 2>&1 &
echo "relay ×2 up (로그: /tmp/relay_cup.log /tmp/relay_shaker.log)"
echo "확인: ros2 topic echo /cup_pose · /shaker_pose"
# 종료: docker rm -f fpp_cup fpp_shaker ; pkill -f cup_pose_relay.py
