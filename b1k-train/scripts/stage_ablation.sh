#!/usr/bin/env bash
# Stage-conditioning ablation for the RLC checkpoint_2 on task 1: how much does the action model
# depend on the stage input? Runs wait_for_gpu_and_rollout.sh once per condition, sequentially.
#   control  : normal stage tracking
#   fixed0   : model always sees stage 0
#   shift1   : model sees tracked stage + 1
# Results: eval_runs/stage_ablation_<date>/<condition>/picking_up_trash/json/*.json
#   cd b1k-train && setsid nohup scripts/stage_ablation.sh > outputs/logs/stage_ablation.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
DATE=$(date +%Y%m%d)
export TASKS="${TASKS:-1}" MODE="${MODE:-public_test}" INSTANCES="${INSTANCES:-0-1}"
export ENV_WRAPPER="${ENV_WRAPPER:-omnigibson.eval.wrappers.DefaultWrapper}"   # same as the l1_pilot baseline
export RESTART_TRAINING_WATCHER=0
export SERVE_SCRIPT=serve_ilia_logged.py   # asset-id auto-resolve + stage decision log
export EVAL_RUNS="${EVAL_RUNS:-$PWD/../eval_runs/stage_ablation_${DATE}}"
mkdir -p "$EVAL_RUNS"; cp ../eval_runs/eval_rollout.py "$EVAL_RUNS/"   # the sim reads /scratch/eval_rollout.py
for cond in ${CONDITIONS:-control fixed0 shift1}; do
    case "$cond" in
        control) unset B1K_STAGE_OVERRIDE ;;
        fixed0)  export B1K_STAGE_OVERRIDE="fixed:0" ;;
        shift1)  export B1K_STAGE_OVERRIDE="shift:1" ;;
        *) echo "unknown condition $cond"; exit 1 ;;
    esac
    echo "$(date '+%F %T') === condition $cond (B1K_STAGE_OVERRIDE=${B1K_STAGE_OVERRIDE:-unset}) ==="
    RUN_NAME="$cond" TEAM="rlc-ckpt2-stage-$cond" scripts/wait_for_gpu_and_rollout.sh
    echo "$(date '+%F %T') === $cond done: $(find "$EVAL_RUNS/$cond" -path '*/json/*.json' | wc -l) rollouts ==="
done
echo "$(date '+%F %T') all conditions done -> $EVAL_RUNS"
