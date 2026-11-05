#!/bin/bash
# vision-3090 에서 sim_fpp 스크립트를 라이브 FP++ 컨테이너와 같은 이미지·같은 파일 마운트로 돌린다.
# usage: run_simfpp.sh <tag> <script.py> [args...]      (컨테이너 이름 simfpp_<tag>, --rm)
#   라이브(sim2real scripts/vision/fpp_up.sh)와 같음: 이미지 perception-plus-plus:humble-cup,
#   호스트 yolo.py · foundationpose_plus_plus.py 파일 단위 :ro 마운트, assets/meshes · s2r_meshes, torch/warp 캐시.
#   다름: ROS 를 띄우지 않음(--network none), 노드 대신 /workspace/sim_fpp/scripts/<script.py> 실행.
set -eo pipefail
TAG=${1:?tag}; SCRIPT=${2:?script}; shift 2
PPP="$HOME/rl_ws/perception_plus_plus"
SIM2REAL="$HOME/rl_ws/sim2real"
WORK="$HOME/rl_ws/sim_fpp"
IMAGE=perception-plus-plus:humble-cup
CACHE="$WORK/cache"   # 라이브 캐시(~/.cache/fpp)는 건드리지 않는다 — 처음 한 번 그곳에서 복사해 둠
mkdir -p "$CACHE/torch" "$CACHE/warp"
LIVE_MD5=$(md5sum "$PPP/perception_plus_plus_core/detection/yolo.py" \
  "$PPP/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py" | awk '{print substr($1,1,12)}' | paste -sd,)
COMMIT=$(git -C "$PPP" rev-parse --short HEAD)
exec docker run --rm --name "simfpp_$TAG" --network none --ipc=host --gpus all \
  -e PPP_ROOT=/workspace/perception_plus_plus -e PYTHONUNBUFFERED=1 \
  -e SIMFPP_IMAGE="$IMAGE@$(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)" \
  -e SIMFPP_PPP_COMMIT="$COMMIT" -e SIMFPP_LIVE_MD5="$LIVE_MD5" -e SIMFPP_HOST="$(hostname)" \
  -v "$PPP/perception_plus_plus_core/detection/yolo.py":/workspace/perception_plus_plus/perception_plus_plus_core/detection/yolo.py:ro \
  -v "$PPP/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py":/workspace/perception_plus_plus/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:ro \
  -v "$PPP/assets/meshes":/workspace/perception_plus_plus/assets/meshes:ro \
  -v "$SIM2REAL/assets/meshes":/workspace/perception_plus_plus/assets/s2r_meshes:ro \
  -v "$WORK":/workspace/sim_fpp \
  -v "$CACHE/torch":/home/perception/.cache/torch -v "$CACHE/warp":/home/perception/.cache/warp \
  "$IMAGE" bash -lc 'cd /workspace/perception_plus_plus && exec python3 "/workspace/sim_fpp/scripts/$0" "$@"' \
  "$SCRIPT" "$@"
