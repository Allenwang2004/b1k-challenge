#!/usr/bin/env bash
# L1 pilot rollouts: Ilia's checkpoint 2 served with a stage log (serve_ilia_logged.py), rollouts recorded by
# omnigibson.eval.wrappers.l1_rollout_recorder.L1RolloutRecorder. Read L1_PILOT_GUIDE.md first.
#
#   bash l1_pilot_rollouts.sh --gpu N [--dry-run]            server and sim on GPU N, detached
#   bash l1_pilot_rollouts.sh --server-gpu A --sim-gpu B     server on A, sim on B
#   bash l1_pilot_rollouts.sh --smoke --gpu N                task 0, instance 301 only -> eval_runs/l1_smoke
#   bash l1_pilot_rollouts.sh --status                       driver, server, container, tasks done, disk, GPUs
#   bash l1_pilot_rollouts.sh --stop                         stops OUR container and OUR server, nothing else
#
# Resume = run the same start command again: eval_rollout.py skips tasks whose rollouts are all written.
# Overrides (environment): CKPT, PORT, TASKS, INSTANCES, RUN, TASK_TIMEOUT, POLICY_MEM_FRACTION, MIN_FREE_GB.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # b1k-train
EVAL_ROOT="$(cd "$HERE/.." && pwd)"                     # ~/evaluation
EVAL_RUNS="$EVAL_ROOT/eval_runs"

PILOT_TASKS="0,1,7,8,9,12,16,17,18,20"
RUN="${RUN:-l1_pilot}"
TASKS="${TASKS:-$PILOT_TASKS}"
INSTANCES="${INSTANCES:-0-1}"                         # public_test indices -> instances 301, 302
TASK_TIMEOUT="${TASK_TIMEOUT:-43200}"                 # per task (both instances); the longest pilot task ~7.5 h
CKPT="${CKPT:-$EVAL_ROOT/behavior_checkpoints/ilia/checkpoint_2}"
PORT="${PORT:-8010}"
POLICY_VENV="${POLICY_VENV:-$EVAL_ROOT/b1k-evaluation/baselines/openpi/.venv}"
POLICY_MEM_FRACTION="${POLICY_MEM_FRACTION:-0.12}"   # a cap; the server uses ~7.5 GB with preallocation off
DATA_PATH="${DATA_PATH:-$EVAL_ROOT/BEHAVIOR-1K/datasets}"
KIT="${KIT:-$EVAL_ROOT/sim_eval_kit}"
EVAL_SRC="$EVAL_ROOT/BEHAVIOR-1K/OmniGibson/omnigibson/eval"
SIM_IMAGE="${SIM_IMAGE:-b1k-sim:latest}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
SERVER_MIN_FREE_MIB="${SERVER_MIN_FREE_MIB:-10240}"
SIM_MIN_FREE_MIB="${SIM_MIN_FREE_MIB:-16384}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-600}"
CONTAINER="l1-pilot-sim"
LOCK="$EVAL_RUNS/.l1_pilot.lock"
WRAPPER="omnigibson.eval.wrappers.l1_rollout_recorder.L1RolloutRecorder"
UID_GID="$(id -u):$(id -g)"

SERVER_GPU="" SIM_GPU="" DRY=0 MODE=start FOREGROUND=0 SMOKE=0
while (($#)); do
  case "$1" in
    --gpu) SERVER_GPU="${2:?--gpu needs a value}"; SIM_GPU="$2"; shift 2 ;;
    --server-gpu) SERVER_GPU="${2:?--server-gpu needs a value}"; shift 2 ;;
    --sim-gpu) SIM_GPU="${2:?--sim-gpu needs a value}"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --status) MODE=status; shift ;;
    --stop) MODE=stop; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
if ((SMOKE)); then
  RUN=l1_smoke TASKS=0 INSTANCES=0 TASK_TIMEOUT=2700   # one rollout, ~25 min; 45 min budget
fi
OUT="$EVAL_RUNS/$RUN"
LOGS="$OUT/logs"

log() { echo "$(date '+%F %T') $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
show() { printf '  '; printf '%q ' "$@"; printf '\n'; }

gpu_free_mib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
gpu_users() {   # "pid user MiB" for every compute process on GPU $1
  local uuid
  uuid=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits |
    while IFS=', ' read -r u pid mem; do
      [[ "$u" == "$uuid" ]] && echo "    pid $pid user $(ps -o user= -p "$pid" 2>/dev/null || echo '?') ${mem} MiB"
    done
}
# a pid is ours only if it is alive, owned by us, and its command line names the expected script
pid_is() {
  local pid="$1" pattern="$2"
  [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]] || return 1
  [[ "$(stat -c %u "/proc/$pid")" == "$(id -u)" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q -- "$pattern"
}
container_state() {
  docker ps -a --filter "name=^/${CONTAINER}$" --filter "label=l1-pilot=1" --format '{{.Status}}' 2>/dev/null
}
server_cmd() {   # sets SERVER_CMD
  SERVER_CMD=(env CUDA_VISIBLE_DEVICES="$SERVER_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false
    XLA_PYTHON_CLIENT_MEM_FRACTION="$POLICY_MEM_FRACTION" TORCHDYNAMO_DISABLE=1
    OMNIGIBSON_DATA_PATH="$DATA_PATH" L1_STAGE_LOG_DIR="$OUT/stage_logs"
    "$POLICY_VENV/bin/python" -P "$HERE/serve_ilia_logged.py" --solution-repo "$HERE" --port "$PORT"
    policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir "$CKPT")
}
record_meta() {
  local commit
  commit=$(git -C "$EVAL_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
  printf '{"run": "%s", "checkpoint": "%s", "workspace_commit": "%s", "server_gpu": "%s", "sim_gpu": "%s", "port": %s, "driver": "l1_pilot_rollouts.sh"}' \
    "$RUN" "$CKPT" "$commit" "$SERVER_GPU" "$SIM_GPU" "$PORT"
}
sim_args() {   # one argument per line, consumed with mapfile
  printf '%s\n' run -d --name "$CONTAINER" --label l1-pilot=1 --label "l1-run=$RUN" \
    --runtime=nvidia --network host --user "$UID_GID" \
    -e NVIDIA_VISIBLE_DEVICES="$SIM_GPU" -e CUDA_VISIBLE_DEVICES=0 \
    -e OMNIGIBSON_HEADLESS=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e HOME=/kit_home -e OMNIGIBSON_APPDATA_PATH=/kit_home/appdata \
    -e "L1_RECORD_META=$(record_meta)" \
    -v "$KIT:/kit_home" -v "$KIT/passwd:/etc/passwd:ro" -v "$KIT/group:/etc/group:ro" \
    -v "$KIT/isaacsim_apps:/opt/conda/envs/behavior/lib/python3.11/site-packages/isaacsim/apps" \
    -v "$DATA_PATH:/data" -v "$EVAL_RUNS:/scratch" \
    -v "$EVAL_SRC:/behavior-src/OmniGibson/omnigibson/eval:ro" \
    "$SIM_IMAGE" /opt/conda/envs/behavior/bin/python -u /scratch/eval_rollout.py \
    --tasks "$TASKS" --mode public_test --instances "$INSTANCES" --num-rollouts 1 \
    --host 127.0.0.1 --port "$PORT" --output-dir "/scratch/$RUN" \
    --robot-config /behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml \
    --env-wrapper "$WRAPPER" --write-video --min-free-gb "$MIN_FREE_GB" --task-timeout "$TASK_TIMEOUT" \
    --submission-out "/scratch/$RUN/submission.json" --team "l1-pilot-rlc-ckpt2"
}

preflight() {   # prints what it checked; exits on the first hard failure
  [[ -n "$SERVER_GPU" && -n "$SIM_GPU" ]] || die "choose a GPU: --gpu N (or --server-gpu A --sim-gpu B)"
  local f
  for f in "$CKPT/params" "$POLICY_VENV/bin/python" "$HERE/serve_ilia.py" "$HERE/serve_ilia_logged.py" \
           "$EVAL_SRC/wrappers/l1_rollout_recorder.py" "$EVAL_SRC/wrappers/l1_record_io.py" \
           "$EVAL_SRC/r1pro.yaml" "$EVAL_RUNS/eval_rollout.py" "$KIT/passwd" "$KIT/group" \
           "$KIT/isaacsim_apps" "$KIT/appdata" "$DATA_PATH"; do
    [[ -e "$f" ]] || die "missing $f"
  done
  grep -q "^[^:]*:x:$(id -u):" "$KIT/passwd" || die "$KIT/passwd has no entry for uid $(id -u)"
  docker image inspect "$SIM_IMAGE" >/dev/null 2>&1 || die "docker image $SIM_IMAGE not found"
  local free_gb
  free_gb=$(df -P --block-size=1G "$EVAL_RUNS" | awk 'NR == 2 {print $4}')
  ((free_gb >= MIN_FREE_GB)) || die "$EVAL_RUNS has ${free_gb} GB free, need >= ${MIN_FREE_GB} GB"
  echo "  disk: ${free_gb} GB free under $EVAL_RUNS"
  local fs fm
  fs=$(gpu_free_mib "$SERVER_GPU") || die "cannot query GPU $SERVER_GPU"
  fm=$(gpu_free_mib "$SIM_GPU") || die "cannot query GPU $SIM_GPU"
  if [[ "$SERVER_GPU" == "$SIM_GPU" ]]; then
    ((fs >= SERVER_MIN_FREE_MIB + SIM_MIN_FREE_MIB)) ||
      die "GPU $SERVER_GPU has ${fs} MiB free, server + sim need $((SERVER_MIN_FREE_MIB + SIM_MIN_FREE_MIB)) MiB"
  else
    ((fs >= SERVER_MIN_FREE_MIB)) || die "GPU $SERVER_GPU has ${fs} MiB free, the server needs ${SERVER_MIN_FREE_MIB} MiB"
    ((fm >= SIM_MIN_FREE_MIB)) || die "GPU $SIM_GPU has ${fm} MiB free, the sim needs ${SIM_MIN_FREE_MIB} MiB"
  fi
  local g
  for g in $(printf '%s\n' "$SERVER_GPU" "$SIM_GPU" | sort -u); do
    echo "  GPU $g: $(gpu_free_mib "$g") MiB free; processes on it:"
    gpu_users "$g" | grep . || echo "    (none)"
  done
  [[ -z "$(container_state)" ]] || die "container $CONTAINER already exists ($(container_state)); run --status / --stop"
  docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER" && die "a container named $CONTAINER exists but is not ours"
  if curl -s -o /dev/null --max-time 3 "http://127.0.0.1:${PORT}/healthz"; then
    die "something already answers on port $PORT"
  fi
  echo "  checkpoint $CKPT, port $PORT, run dir $OUT"
}

stop_server() {
  local pid
  pid=$(cat "$OUT/server.pid" 2>/dev/null) || return 0
  if pid_is "$pid" serve_ilia_logged.py; then
    log "stopping policy server pid $pid"
    kill -TERM "$pid"
    for _ in $(seq 1 30); do pid_is "$pid" serve_ilia_logged.py || break; sleep 1; done
    pid_is "$pid" serve_ilia_logged.py && { log "server pid $pid ignored TERM; sending KILL"; kill -KILL "$pid"; }
  fi
  rm -f "$OUT/server.pid"
}
stop_container() {
  if [[ -n "$(container_state)" ]]; then
    log "stopping container $CONTAINER"
    docker logs "$CONTAINER" > "$LOGS/sim_container.$(date +%Y%m%d-%H%M%S).log" 2>&1 || true
    docker stop -t 60 "$CONTAINER" >/dev/null 2>&1 || true
    docker rm "$CONTAINER" >/dev/null 2>&1 || true
  fi
}

driver() {   # the detached body: one server, one container, then clean up
  exec 9> "$LOCK"
  flock -n 9 || die "another L1 driver is running (lock $LOCK held)"
  echo "$RUN $$" >&9
  echo $$ > "$OUT/driver.pid"
  cleanup() { stop_container; stop_server; rm -f "$OUT/driver.pid"; }
  trap cleanup EXIT
  trap 'log "driver: signal received, cleaning up"; exit 130' INT TERM

  log "driver $$: run=$RUN tasks=$TASKS instances=$INSTANCES server_gpu=$SERVER_GPU sim_gpu=$SIM_GPU"
  preflight || exit 1

  log "starting policy server -> $LOGS/server.log"
  server_cmd
  (cd "$HERE" && exec "${SERVER_CMD[@]}") >> "$LOGS/server.log" 2>&1 < /dev/null &
  local spid=$!
  echo "$spid" > "$OUT/server.pid"
  local ok=0 waited=0
  while ((waited < HEALTH_TIMEOUT_S)); do
    sleep 5; waited=$((waited + 5))
    pid_is "$spid" serve_ilia_logged.py || break
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${PORT}/healthz")" == 200 ]] && { ok=1; break; }
  done
  ((ok)) || { log "policy server not healthy after ${waited}s; see $LOGS/server.log"; exit 1; }
  log "policy server healthy on :$PORT after ${waited}s (pid $spid)"

  local args
  mapfile -t args < <(sim_args)
  log "starting sim container $CONTAINER on GPU $SIM_GPU as uid $UID_GID"
  docker "${args[@]}" > /dev/null || { log "docker run failed"; exit 1; }

  local state
  while :; do
    sleep 30
    state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo gone)
    [[ "$state" == running ]] || break
    if ! pid_is "$spid" serve_ilia_logged.py; then
      log "policy server died mid-sweep; stopping the sim (see $LOGS/server.log)"
      exit 1
    fi
  done
  local code
  code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER" 2>/dev/null || echo "?")
  log "sim container finished (state $state, exit code $code)"
  if [[ "$code" == 0 ]]; then
    date -Is > "$OUT/DONE"
    log "sweep complete: $OUT"
  fi
  [[ "$code" == 0 ]]
}

status() {
  local r holder
  holder=$(cat "$LOCK" 2>/dev/null)
  echo "lock: ${holder:-none}"
  for r in l1_smoke l1_pilot; do
    local o="$EVAL_RUNS/$r"
    [[ -d "$o" ]] || continue
    echo "== $r ($o)"
    local d s
    d=$(cat "$o/driver.pid" 2>/dev/null); s=$(cat "$o/server.pid" 2>/dev/null)
    if pid_is "$d" l1_pilot_rollouts.sh; then echo "  driver: running (pid $d)"; else echo "  driver: not running"; fi
    if pid_is "$s" serve_ilia_logged.py; then echo "  server: running (pid $s)"; else echo "  server: not running"; fi
    [[ -f "$o/DONE" ]] && echo "  DONE at $(cat "$o/DONE")"
    local t n
    for t in "$o"/*/json; do
      [[ -d "$t" ]] || continue
      n=$(find "$t" -name '*.json' | wc -l)
      echo "  $(basename "$(dirname "$t")"): $n rollout(s) written, $(find "$(dirname "$t")/l1_record/meta" -name '*.json' 2>/dev/null | wc -l) recorded"
    done
    echo "  stage log lines: $(cat "$o"/stage_logs/*.jsonl 2>/dev/null | wc -l)"
    echo "  driver.log:"; tail -n 3 "$o/logs/driver.log" 2>/dev/null | sed 's/^/    /'
    local latest
    latest=$(ls -t "$o"/logs/[0-9][0-9]_*.log 2>/dev/null | head -1)
    [[ -n "$latest" ]] && { echo "  $(basename "$latest"):"; tail -n 3 "$latest" | cut -c1-200 | sed 's/^/    /'; }
  done
  echo "container: $(container_state || true)"
  df -h "$EVAL_RUNS" | tail -1 | awk '{print "disk: " $4 " free (" $5 " used)"}'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/gpu /'
}

stop_all() {
  local r o d
  for r in l1_smoke l1_pilot; do
    o="$EVAL_RUNS/$r"
    d=$(cat "$o/driver.pid" 2>/dev/null)
    if pid_is "$d" l1_pilot_rollouts.sh; then
      log "stopping driver pid $d ($r); it stops its container and server"
      kill -TERM "$d"
      for _ in $(seq 1 90); do pid_is "$d" l1_pilot_rollouts.sh || break; sleep 1; done
    fi
    # anything the driver left behind (it was killed hard, or it was already gone)
    OUT="$o" LOGS="$o/logs"
    mkdir -p "$LOGS"
    stop_container
    stop_server
  done
  log "stopped (only $CONTAINER and our own server pid were touched)"
}

case "$MODE" in
  status) status; exit 0 ;;
  stop) stop_all; exit 0 ;;
esac

if ((FOREGROUND)); then
  mkdir -p "$LOGS" "$OUT/stage_logs"
  driver
  exit $?
fi

if ((DRY)); then
  echo "preflight:"
  (preflight) || echo "  preflight FAILED: a real start would refuse (see the error above)"
  echo "policy server (cwd $HERE, log $LOGS/server.log):"
  server_cmd
  show "${SERVER_CMD[@]}"
  echo "sim container:"
  mapfile -t _args < <(sim_args)
  show docker "${_args[@]}"
  echo "dry run: nothing started"
  exit 0
fi

preflight
mkdir -p "$LOGS" "$OUT/stage_logs"
passthru=(--server-gpu "$SERVER_GPU" --sim-gpu "$SIM_GPU")
((SMOKE)) && passthru+=(--smoke)
nohup setsid bash "$0" --foreground "${passthru[@]}" >> "$LOGS/driver.log" 2>&1 < /dev/null &
log "driver started (pid $!), run $RUN -> $OUT"
echo "  watch: bash $0 --status    |   tail -f $LOGS/driver.log"
