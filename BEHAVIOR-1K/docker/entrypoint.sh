#!/bin/bash
# Runtime task selector for the multi-task submission image.
#
# One image bundles every task's checkpoint under /checkpoints/<checkpoint_dir>/
# (per docker/checkpoint_map.json). The evaluator restarts this container once
# per task (docs/challenge/submission.md: "we run OmniGibson outside the
# container and connect to your policy through the WebSocket policy client" —
# eval.py itself only ever evaluates one --task-name per process), so the
# checkpoint to serve is selected at container start via $TASK_NAME rather
# than baked into CMD at build time.
#
# Usage: docker run -e TASK_NAME=turning_on_radio -p 8000:8000 <image>
set -euo pipefail

: "${TASK_NAME:?TASK_NAME env var must be set to select which checkpoint to serve, e.g. -e TASK_NAME=turning_on_radio}"

CKPT_DIR="$(python -c "
import json, sys
m = json.load(open('/checkpoint_map.json'))
task = sys.argv[1]
if task not in m:
    sys.exit(1)
print(m[task])
" "$TASK_NAME")" || {
    echo "No checkpoint mapped for TASK_NAME='$TASK_NAME' in /checkpoint_map.json" >&2
    exit 1
}

exec /openpi/.venv/bin/python /openpi/scripts/serve_b1k_patched.py \
    --task_name="$TASK_NAME" policy:checkpoint \
    --policy.config=pi0_b1k \
    --policy.dir="/checkpoints/$CKPT_DIR"
