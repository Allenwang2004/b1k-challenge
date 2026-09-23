#!/usr/bin/env bash
# Wait until both GPUs have enough free memory, then launch the task-1 training (2-GPU FSDP).
#
#   cd b1k-train && setsid nohup scripts/wait_for_gpu_and_train.sh > outputs/logs/watcher.log 2>&1 &
#   tail -f outputs/logs/watcher.log            # watcher status
#   tail -f outputs/logs/<EXP_NAME>.log         # training log once started
#   kill "$(cat outputs/watcher.pid)"           # stop waiting (does not kill a running training)
#
# Behaviour:
#   * polls nvidia-smi every POLL_SEC; starts when GPU 0 and GPU 1 each have >= MIN_FREE_MIB free
#   * caps JAX at (min free - HEADROOM_MIB) so other people's jobs are not squeezed
#   * if training dies with RESOURCE_EXHAUSTED, waits for more free memory and resumes from the last checkpoint
#     with the SAME batch size (OOM_HALVE=1 restores the old behaviour of halving the batch; that silently
#     changed the recipe to batch 4 on 2026-09-21)
#   * if a checkpoint for EXP_NAME already exists, resumes it instead of overwriting
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs
echo $$ > outputs/watcher.pid

GPUS="${GPUS:-0,1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-40960}"      # 40 GB per GPU
HEADROOM_MIB="${HEADROOM_MIB:-2048}"
POLL_SEC="${POLL_SEC:-60}"
EXP_NAME="${EXP_NAME:-task1_picking_up_trash}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
KEEP_PERIOD="${KEEP_PERIOD:-2000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CONFIG="${CONFIG:-pi_behavior_b1k_fast}"
CKPT_DIR="outputs/checkpoints/${CONFIG}/${EXP_NAME}"
# WANDB=1 logs to Weights & Biases (project "B1K" from the config; needs `wandb login` or WANDB_API_KEY),
# otherwise wandb is disabled and metrics only go to the log file ("Step N: loss=...").
WANDB_FLAG=$([ "${WANDB:-0}" = "1" ] && echo "--wandb_enabled" || echo "--no-wandb_enabled")
# PREDELETE=1: delete the previous checkpoint before saving the next (one 44 GB copy on disk instead of two).
PREDELETE_FLAG=$([ "${PREDELETE:-0}" = "1" ] && echo "--delete_previous_checkpoint_before_save" || echo "")
OOM_HALVE="${OOM_HALVE:-0}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

log() { echo "$(date '+%F %T') $*"; }

gpu_free_mib() {  # prints "free total" for one GPU index
    nvidia-smi -i "$1" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', ' '{print $2-$1, $2}'
}

wait_for_gpus() {  # sets MIN_FREE / TOTAL
    while true; do
        MIN_FREE=999999; TOTAL=0; ok=1; status=""
        for g in "${GPU_LIST[@]}"; do
            read -r free total < <(gpu_free_mib "$g")
            status+=" gpu$g:${free}MiB"
            (( free < MIN_FREE )) && MIN_FREE=$free
            TOTAL=$total
            (( free < MIN_FREE_MIB )) && ok=0
        done
        if (( ok )); then
            log "GPUs ready:$status (need >= ${MIN_FREE_MIB} MiB each)"
            return
        fi
        log "waiting:$status (need >= ${MIN_FREE_MIB} MiB each)"
        sleep "$POLL_SEC"
    done
}

while true; do
    wait_for_gpus
    frac=$(awk -v f="$MIN_FREE" -v h="$HEADROOM_MIB" -v t="$TOTAL" 'BEGIN{x=(f-h)/t; if (x>0.95) x=0.95; printf "%.2f", x}')

    if [ -d "$CKPT_DIR" ] && ls "$CKPT_DIR" 2>/dev/null | grep -Eq '^[0-9]+$'; then
        mode="--resume"
    else
        mode="--overwrite"
    fi
    train_log="outputs/logs/${EXP_NAME}.$(date +%Y%m%d-%H%M%S).log"   # one file per attempt
    ln -sfn "$(basename "$train_log")" "outputs/logs/${EXP_NAME}.log"     # ...and a stable name for tail -f
    log "launching: batch_size=${BATCH_SIZE} fsdp_devices=${NUM_GPUS} mem_fraction=${frac} ${mode} -> ${train_log}"

    # PREALLOCATE=true: claim our share (what was free minus headroom) up front, so a job another user
    # starts during our ~2 min compile cannot take it from under us (2026-09-17 crash: cuDNN init failed
    # when a vLLM instance launched mid-compile).
    CUDA_VISIBLE_DEVICES="$GPUS" XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION="$frac" \
    .venv/bin/python scripts/train.py "$CONFIG" \
        --exp_name "$EXP_NAME" \
        --batch_size "$BATCH_SIZE" \
        --fsdp_devices "$NUM_GPUS" \
        --num_train_steps "$NUM_TRAIN_STEPS" \
        --save_interval "$SAVE_INTERVAL" \
        --keep_period "$KEEP_PERIOD" \
        --log_interval "$LOG_INTERVAL" \
        --num_workers "$NUM_WORKERS" \
        $WANDB_FLAG \
        $PREDELETE_FLAG \
        $mode >> "$train_log" 2>&1
    rc=$?

    if (( rc == 0 )); then
        log "training finished (exit 0). Checkpoints: ${CKPT_DIR}"
        rm -f outputs/watcher.pid
        exit 0
    fi
    if grep -q "RESOURCE_EXHAUSTED" "$train_log"; then
        if [ "$OOM_HALVE" != "1" ]; then
            log "training OOM (exit $rc); keeping batch_size=${BATCH_SIZE}, will resume from last checkpoint once more memory is free"
        elif (( BATCH_SIZE > 2 )); then
            BATCH_SIZE=$(( BATCH_SIZE / 2 ))
            log "training OOM (exit $rc); retrying with batch_size=${BATCH_SIZE} once GPUs are free again"
        else
            log "training OOM (exit $rc) even at batch_size=2; will wait for more free memory and retry"
        fi
        MIN_FREE_MIB=$(( MIN_FREE_MIB + 4096 ))
        sleep "$POLL_SEC"
        continue
    fi
    # GPU-level init failures (cuDNN/CUDA could not get memory or a context, typically because another
    # job grabbed the GPU while we were compiling): transient, keep the batch size and wait again.
    if grep -Eq "Failed to set cuDNN stream|CUDNN_STATUS_(INTERNAL_ERROR|ALLOC_FAILED|NOT_INITIALIZED)|CUDA_ERROR_OUT_OF_MEMORY|CUDA_ERROR_ILLEGAL_ADDRESS" "$train_log"; then
        log "GPU init/runtime failure (exit $rc), treating as transient; waiting for GPUs again"
        sleep "$POLL_SEC"
        continue
    fi
    log "training failed (exit $rc), not an OOM -- see ${train_log}. Stopping watcher."
    rm -f outputs/watcher.pid
    exit "$rc"
done
