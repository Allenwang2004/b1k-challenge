#!/bin/bash
# Sequentially run `omnigibson.eval.eval` for tasks listed in a task_map.json
# (categories like F1/F2/F3/F4/baseline/..., each split into "2025"/"2026" numeric
# task-ID lists), against an already-running `policy-dev` server.
#
# Assumes:
#   - run from the directory containing docker-compose.yml
#   - a `policy-dev` service exists in docker-compose.yml (mount-based dev server,
#     command reads $TASK_NAME/$POLICY_DIR/$POLICY_CONFIG)
#
# Because the sim<->policy websocket protocol carries no per-request task/prompt
# field (the server's prompt is fixed for its whole lifetime), evaluating many
# different tasks against one multi-task checkpoint requires restarting policy-dev
# with the right TASK_NAME before each task's sim run. This script does that.
#
# Usage:
#   ./scripts/sim_eval.sh \
#     --task-map task_map.json \
#     --checkpoint "100ep:/home/b1k-challenge/evaluation/behavior_checkpoints/100ep:pi05_b1k" \
#     [--checkpoint "200ep:/home/b1k-challenge/evaluation/behavior_checkpoints/200ep:pi05_b1k"] \
#     [--category F1] [--category F2_fast] \
#     [--instance-indices "0 1 2"] \
#     [--num-rollouts 1] \
#     [--host policy-dev] [--port 8000] \
#     [--policy-gpu 1] [--sim-gpu 0] [--healthy-timeout 1200] \
#     [--task-misc-csv <path to B100_task_misc.csv>] \
#     [--output-root /data/outputs] \
#     [--dry-run]
#
# --checkpoint takes "name:policy_dir:policy_config" and may be given more than
# once. Checkpoints run one at a time, in the order given — a checkpoint's entire
# run (every selected category, every task) finishes before the next checkpoint
# starts.
#
# --category may be given more than once. Without it, EVERY category in
# task_map.json is run. Categories run one at a time, in the order given — a
# category's tasks all finish before the next category starts.
#
# --output-dir per run: <output-root>/<category>/<checkpoint-name>/<year>/
#
# A task that fails to load (policy-dev never becomes healthy) or whose sim run
# exits non-zero is logged and skipped, not fatal to the whole batch. A summary
# of failures prints at the end.
set -uo pipefail

TASK_MAP=""
CHECKPOINTS=()
CATEGORIES=()
INSTANCE_INDICES="0"
NUM_ROLLOUTS=1
HOST="policy-dev"
PORT=8000
POLICY_GPU=""
SIM_GPU=""
HEALTHY_TIMEOUT=1200
TASK_MISC_CSV="$HOME/evaluation/remote-datasets/2026-challenge-task-instances/metadata/B100_task_misc.csv"
OUTPUT_ROOT="/data/outputs"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-map) TASK_MAP="$2"; shift 2 ;;
    --checkpoint) CHECKPOINTS+=("$2"); shift 2 ;;
    --category) CATEGORIES+=("$2"); shift 2 ;;
    --instance-indices) INSTANCE_INDICES="$2"; shift 2 ;;
    --num-rollouts) NUM_ROLLOUTS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --sim-gpu) SIM_GPU="$2"; shift 2 ;;
    --healthy-timeout) HEALTHY_TIMEOUT="$2"; shift 2 ;;
    --task-misc-csv) TASK_MISC_CSV="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$TASK_MAP" ]] && { echo "--task-map is required" >&2; exit 1; }
[[ ${#CHECKPOINTS[@]} -eq 0 ]] && { echo "at least one --checkpoint is required" >&2; exit 1; }
[[ ! -f "$TASK_MAP" ]] && { echo "task map not found: $TASK_MAP" >&2; exit 1; }
[[ ! -f "$TASK_MISC_CSV" ]] && { echo "task misc csv not found: $TASK_MISC_CSV" >&2; exit 1; }

CHECKPOINT_NAMES=()
POLICY_DIRS=()
POLICY_CONFIGS=()
for entry in "${CHECKPOINTS[@]}"; do
  IFS=':' read -r name dir config <<< "$entry"
  if [[ -z "$name" || -z "$dir" || -z "$config" ]]; then
    echo "--checkpoint must be 'name:policy_dir:policy_config', got: $entry" >&2
    exit 1
  fi
  CHECKPOINT_NAMES+=("$name")
  POLICY_DIRS+=("$dir")
  POLICY_CONFIGS+=("$config")
done

if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
  mapfile -t CATEGORIES < <(python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1])).keys()))" "$TASK_MAP")
fi

# Resolve "<category> <year> <task_name>" triples for the selected categories, in the
# order given (categories are kept separate, not merged, since each writes to its own
# output folder). Map numeric task id -> task name via B100_task_misc.csv, dedupe
# within the same category+year. This job list does not depend on the checkpoint, so
# it's resolved once and reused for every checkpoint.
mapfile -t JOBS < <(python3 - "$TASK_MAP" "$TASK_MISC_CSV" "${CATEGORIES[@]}" << 'PYEOF'
import csv, json, sys

task_map_path, csv_path = sys.argv[1], sys.argv[2]
categories = sys.argv[3:]

id_to_name = {}
with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        id_to_name[int(row["Task ID"])] = row["Task"]

with open(task_map_path) as f:
    task_map = json.load(f)

for cat in categories:
    if cat not in task_map:
        print(f"ERROR: category {cat!r} not found in {task_map_path}", file=sys.stderr)
        sys.exit(1)
    for year in ("2025", "2026"):
        seen = set()
        for tid in task_map[cat].get(year, []):
            if tid in seen:
                continue
            seen.add(tid)
            name = id_to_name.get(tid)
            if name is None:
                print(f"WARNING: task id {tid} not found in {csv_path}", file=sys.stderr)
                continue
            print(f"{cat} {year} {name}")
PYEOF
)

echo "Resolved ${#JOBS[@]} task runs per checkpoint from $TASK_MAP (categories: ${CATEGORIES[*]})"
echo "Checkpoints (run in order): ${CHECKPOINT_NAMES[*]}"

wait_for_policy_healthy() {
  local waited=0 cid status
  while true; do
    cid="$(docker-compose ps -q policy-dev)"
    if [[ -n "$cid" ]]; then
      status="$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo "unknown")"
      [[ "$status" == "healthy" ]] && return 0
      [[ "$status" == "unhealthy" ]] && { echo "policy-dev reported unhealthy" >&2; return 1; }
    fi
    if (( waited >= HEALTHY_TIMEOUT )); then
      echo "Timed out (${HEALTHY_TIMEOUT}s) waiting for policy-dev to become healthy" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

FAILED=()
TOTAL_RUNS=0

# Checkpoints are run one at a time, in the order given: a checkpoint's entire run
# (every selected category, every task — JOBS is already grouped by category) finishes
# before the next checkpoint starts.
for ckpt_i in "${!CHECKPOINT_NAMES[@]}"; do
  CHECKPOINT_NAME="${CHECKPOINT_NAMES[$ckpt_i]}"
  POLICY_DIR="${POLICY_DIRS[$ckpt_i]}"
  POLICY_CONFIG="${POLICY_CONFIGS[$ckpt_i]}"

  echo
  echo "########## Checkpoint: $CHECKPOINT_NAME ($((ckpt_i + 1))/${#CHECKPOINT_NAMES[@]}) ##########"

  for job in "${JOBS[@]}"; do
    read -r cat year task_name <<< "$job"
    out_dir="${OUTPUT_ROOT}/${cat}/${CHECKPOINT_NAME}/${year}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    restart_cmd=(env TASK_NAME="$task_name" POLICY_DIR="$POLICY_DIR" POLICY_CONFIG="$POLICY_CONFIG")
    [[ -n "$POLICY_GPU" ]] && restart_cmd+=(POLICY_GPU="$POLICY_GPU")
    restart_cmd+=(docker-compose up -d policy-dev)

    sim_cmd=(env)
    [[ -n "$SIM_GPU" ]] && sim_cmd+=(SIM_GPU="$SIM_GPU")
    sim_cmd+=(docker-compose run --rm --no-deps sim python -m omnigibson.eval.eval
         --task-name "$task_name"
         --host "$HOST" --port "$PORT"
         --instance-indices $INSTANCE_INDICES
         --num-rollouts "$NUM_ROLLOUTS"
         --write-video
         --output-dir "$out_dir"
         --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper)

    echo "=== [$CHECKPOINT_NAME | $cat/$year] $task_name -> $out_dir ==="

    if [[ "$DRY_RUN" == "1" ]]; then
      printf '[restart policy-dev] '; printf '%q ' "${restart_cmd[@]}"; echo
      printf '[wait for healthy, timeout=%ss]\n' "$HEALTHY_TIMEOUT"
      printf '[sim eval] '; printf '%q ' "${sim_cmd[@]}"; echo
      continue
    fi

    docker-compose stop policy-dev >/dev/null 2>&1
    docker-compose rm -f policy-dev >/dev/null 2>&1
    if ! "${restart_cmd[@]}"; then
      echo "Failed to start policy-dev for task $task_name, skipping" >&2
      FAILED+=("$CHECKPOINT_NAME $cat/$year $task_name (policy-dev start failed)")
      continue
    fi

    if ! wait_for_policy_healthy; then
      echo "policy-dev never became healthy for task $task_name, skipping" >&2
      FAILED+=("$CHECKPOINT_NAME $cat/$year $task_name (policy-dev not healthy)")
      continue
    fi

    if ! "${sim_cmd[@]}"; then
      echo "sim eval failed for task $task_name" >&2
      FAILED+=("$CHECKPOINT_NAME $cat/$year $task_name (sim eval failed)")
    fi
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo
  echo "Done: ${TOTAL_RUNS} task runs attempted across ${#CHECKPOINT_NAMES[@]} checkpoint(s), ${#FAILED[@]} failed."
  if (( ${#FAILED[@]} > 0 )); then
    printf ' - %s\n' "${FAILED[@]}"
    exit 1
  fi
fi
