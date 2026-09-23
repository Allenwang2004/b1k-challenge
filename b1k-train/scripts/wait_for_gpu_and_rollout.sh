#!/usr/bin/env bash
# Wait for GPU memory, then run the RLC (2025 winner) checkpoint on task 1 in OmniGibson:
#   1. start their policy server (serve_ilia.py, eval venv) on the freest GPU
#   2. run eval_rollout.py inside the b1k-sim container against it
#   3. stop the server; optionally restart the training watcher
#
#   cd b1k-train && setsid nohup scripts/wait_for_gpu_and_rollout.sh > outputs/logs/rollout_watcher.log 2>&1 &
#   tail -f outputs/logs/rollout_watcher.log                 # this script
#   tail -f ../eval_runs/<RUN_NAME>/logs/01_picking_up_trash.log   # the sim, once running
#   tail -f outputs/logs/ilia_serve.log                      # the policy server
#   kill "$(cat outputs/rollout_watcher.pid)"                # stop (also stops server + sim)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # b1k-train
EVAL_ROOT="$(cd "$ROOT/.." && pwd)"                # /home/b1k-challenge/evaluation
cd "$ROOT"
mkdir -p outputs/logs
echo $$ > outputs/rollout_watcher.pid

TASKS="${TASKS:-1}"                                # picking_up_trash
MODE="${MODE:-public_test}"
INSTANCES="${INSTANCES:-0-9}"                      # -> instance ids 301..310
RUN_NAME="${RUN_NAME:-rlc_ckpt2_task1_$(date +%Y%m%d)}"
TEAM="${TEAM:-RLC-checkpoint2}"                    # --team written into submission.json
EVAL_RUNS="${EVAL_RUNS:-$EVAL_ROOT/eval_runs}"
CKPT="${CKPT:-$EVAL_ROOT/behavior_checkpoints/ilia/checkpoint_2}"   # ckpt 2 covers task 1
PORT="${PORT:-8010}"
POLICY_VENV="${POLICY_VENV:-$EVAL_ROOT/b1k-evaluation/baselines/openpi/.venv}"
DATA_PATH="${DATA_PATH:-$EVAL_ROOT/BEHAVIOR-1K/datasets}"
SIM_IMAGE="${SIM_IMAGE:-b1k-sim:latest}"
POLICY_MEM_FRACTION="${POLICY_MEM_FRACTION:-0.25}"   # ~24 GB on a 96 GB card (what the Aug run used)
SERVER_MIN_FREE_MIB="${SERVER_MIN_FREE_MIB:-27648}"  # fraction*total + headroom
SIM_MIN_FREE_MIB="${SIM_MIN_FREE_MIB:-16384}"        # Isaac Sim headless, 3 cameras
POLL_SEC="${POLL_SEC:-60}"
MIN_FREE_GB_DISK="${MIN_FREE_GB_DISK:-10}"
# Cameras at data-collection resolution (head 720, wrists 480, RGB only) so videos are full-res and the
# policy sees frames like the demos (it resizes to 224 itself). DefaultWrapper renders at 224px.
ENV_WRAPPER="${ENV_WRAPPER:-omnigibson.eval.wrappers.RGBFullResWrapper}"
VIDEO_CRF="${VIDEO_CRF:-18}"
# The image's /behavior-src copy of omnigibson/eval is byte-identical to the host's; mounting the host dir
# lets edits (wrapper, video compositing) take effect without rebuilding the image.
EVAL_SRC_MOUNT="${EVAL_SRC_MOUNT:-$EVAL_ROOT/BEHAVIOR-1K/OmniGibson/omnigibson/eval:/behavior-src/OmniGibson/omnigibson/eval:ro}"
RESTART_TRAINING_WATCHER="${RESTART_TRAINING_WATCHER:-1}"
# serve_ilia_logged.py = serve_ilia.py + per-decision stage log (JSONL under <run>/stage_logs) + automatic
# asset-id resolution (checkpoint_2 carries assets/IliaLarchenko/behavior_224_rgb, the config names the 2026 id).
SERVE_SCRIPT="${SERVE_SCRIPT:-serve_ilia.py}"

log() { echo "$(date '+%F %T') $*"; }
gpu_free() { nvidia-smi -i "$1" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $2-$1}'; }

SERVER_PID=""
CONTAINER="${CONTAINER:-b1k-eval-${RUN_NAME}}"
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then kill "$SERVER_PID"; log "policy server stopped"; fi
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    [ "$(cat outputs/rollout_watcher.pid 2>/dev/null)" = "$$" ] && rm -f outputs/rollout_watcher.pid  # only ours
    rm -f outputs/ilia_serve.pid
}
trap 'cleanup; exit 130' INT TERM

# ---- 1. wait for GPUs: server on the freest card, sim on the other (or same if it fits both) --------
pick_gpus() {   # sets SERVER_GPU / SIM_GPU, returns 1 if not enough memory yet
    local f0 f1
    f0=$(gpu_free 0); f1=$(gpu_free 1)
    STATUS="gpu0:${f0}MiB gpu1:${f1}MiB"
    if (( f0 >= f1 )); then SERVER_GPU=0; SIM_GPU=1; SF=$f0; MF=$f1; else SERVER_GPU=1; SIM_GPU=0; SF=$f1; MF=$f0; fi
    (( SF < SERVER_MIN_FREE_MIB )) && return 1
    if (( MF >= SIM_MIN_FREE_MIB )); then return 0; fi
    if (( SF >= SERVER_MIN_FREE_MIB + SIM_MIN_FREE_MIB )); then SIM_GPU=$SERVER_GPU; return 0; fi
    return 1
}

while true; do
    while ! pick_gpus; do
        log "waiting: $STATUS (server needs >= ${SERVER_MIN_FREE_MIB} MiB, sim >= ${SIM_MIN_FREE_MIB} MiB)"
        sleep "$POLL_SEC"
    done
    log "GPUs ready: $STATUS -> policy server on GPU ${SERVER_GPU}, sim on GPU ${SIM_GPU}"

    # ---- 2. policy server ------------------------------------------------------------------------
    # -P: don't put b1k-train/ on sys.path (its vendored openpi/ dir would shadow the eval venv's openpi).
    # PREALLOCATE=true: claim the memory now so a job started during our boot cannot take it.
    SERVE_LOG="outputs/logs/ilia_serve.$(date +%Y%m%d-%H%M%S).log"
    ln -sfn "$(basename "$SERVE_LOG")" outputs/logs/ilia_serve.log
    mkdir -p "$EVAL_RUNS/$RUN_NAME/stage_logs"
    CUDA_VISIBLE_DEVICES="$SERVER_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION="$POLICY_MEM_FRACTION" \
    TORCHDYNAMO_DISABLE=1 OMNIGIBSON_DATA_PATH="$DATA_PATH" L1_STAGE_LOG_DIR="$EVAL_RUNS/$RUN_NAME/stage_logs" \
    "$POLICY_VENV/bin/python" -P "$ROOT/$SERVE_SCRIPT" --solution-repo "$ROOT" --port "$PORT" \
        policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir "$CKPT" > "$SERVE_LOG" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > outputs/ilia_serve.pid
    log "policy server pid ${SERVER_PID}, log ${SERVE_LOG}"

    ok=0
    for _ in $(seq 1 60); do   # up to 5 min (Aug run: ~50 s)
        sleep 5
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/healthz" || true)
        if [ "$code" = "200" ]; then ok=1; break; fi
    done
    if (( ! ok )); then
        kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""
        if grep -Eq "RESOURCE_EXHAUSTED|CUDA_ERROR_OUT_OF_MEMORY|Failed to set cuDNN stream|CUDNN_STATUS" "$SERVE_LOG"; then
            log "policy server could not get GPU memory (see ${SERVE_LOG}); waiting again"
            sleep "$POLL_SEC"; continue
        fi
        log "policy server failed to come up (see ${SERVE_LOG}). Stopping."
        cleanup; exit 1
    fi
    log "policy server healthy on :${PORT}"

    # ---- 3. sim rollouts -------------------------------------------------------------------------
    OUT="$EVAL_RUNS/$RUN_NAME"
    mkdir -p "$OUT"
    log "starting sim: tasks=${TASKS} mode=${MODE} instances=${INSTANCES} -> ${OUT}"
    docker run --rm --runtime=nvidia --network host --name "$CONTAINER" \
        -e OMNIGIBSON_HEADLESS=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
        -e CUDA_VISIBLE_DEVICES="$SIM_GPU" \
        -v "$DATA_PATH":/data \
        -v "$EVAL_RUNS":/scratch \
        -v "$EVAL_SRC_MOUNT" \
        "$SIM_IMAGE" /opt/conda/envs/behavior/bin/python -u /scratch/eval_rollout.py \
            --tasks "$TASKS" --mode "$MODE" --instances "$INSTANCES" \
            --host 127.0.0.1 --port "$PORT" \
            --output-dir "/scratch/$RUN_NAME" \
            --robot-config /behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml \
            --env-wrapper "$ENV_WRAPPER" \
            --write-video --video-crf "$VIDEO_CRF" --min-free-gb "$MIN_FREE_GB_DISK" \
            --submission-out "/scratch/$RUN_NAME/submission.json" \
            --team "$TEAM" > "outputs/logs/rollout_sim.$(date +%Y%m%d-%H%M%S).log" 2>&1
    rc=$?
    log "sim finished with exit ${rc}; rollout files: $(find "$OUT" -path '*/json/*.json' 2>/dev/null | wc -l)"

    kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""
    log "policy server stopped"
    break
done

rm -f outputs/rollout_watcher.pid outputs/ilia_serve.pid
if [ "$RESTART_TRAINING_WATCHER" = "1" ]; then
    log "restarting training watcher"
    setsid nohup "$ROOT/scripts/wait_for_gpu_and_train.sh" >> "$ROOT/outputs/logs/watcher.log" 2>&1 < /dev/null &
fi
log "done: results in ${OUT} (submission.json, <task>/json/*.json, <task>/videos/*.mp4)"
