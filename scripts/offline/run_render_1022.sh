#!/bin/bash
# 로컬 렌더(1022_sim_fpp/render/*.npz) -> vision-3090 sim_fpp 로 복사 -> FP++ 배치(gt, yolo) + 오차 요약 -> 결과를 로컬 fpp/ 로 회수.
# usage: run_render_1022.sh [params_yaml_name]  비우면 에피소드마다 meta(object_real·fpp_mesh)로 data/params/ 에서 고른다(--params-dir)
set -eo pipefail
EVAL=$HOME/rl_ws/our_source/rh_aglt_runs/eval/1022_sim_fpp
HOST=vision-3090
RDIR=rl_ws/sim_fpp
PARAMS=${1:-}
mkdir -p "$EVAL/fpp"
ssh $HOST "mkdir -p ~/$RDIR/data/render_1022 ~/$RDIR/out/render_1022_gt ~/$RDIR/out/render_1022_yolo"
rsync -a "$EVAL/render/" $HOST:$RDIR/data/render_1022/
PARG="--params-dir /workspace/sim_fpp/data/params"
[ -n "$PARAMS" ] && PARG="--params /workspace/sim_fpp/data/params/$PARAMS"
for MODE in gt yolo; do
  ssh $HOST "cd ~/$RDIR && timeout 1800 scripts/run_simfpp.sh r1022_$MODE sim_fpp_batch.py /workspace/sim_fpp/data/render_1022 --out-dir /workspace/sim_fpp/out/render_1022_$MODE --mask-mode $MODE $PARG 2>&1 | grep -E '^\[simfpp\]|Error|Traceback'" || true
  ssh $HOST "cd ~/$RDIR && timeout 600 scripts/run_simfpp.sh r1022_eval_$MODE sim_fpp_eval.py --gt-dir /workspace/sim_fpp/data/render_1022 --fpp-dir /workspace/sim_fpp/out/render_1022_$MODE 2>&1 | grep -v Warning | tail -40" || true
  mkdir -p "$EVAL/fpp/$MODE"
  rsync -a $HOST:$RDIR/out/render_1022_$MODE/ "$EVAL/fpp/$MODE/"
done
ssh $HOST "docker ps --format '{{.Names}}' | grep simfpp_ || echo 'no simfpp containers'"
