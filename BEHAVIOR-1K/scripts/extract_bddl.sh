#!/bin/bash
# Copies out just the BDDL activity definitions used by the 2026 BEHAVIOR Challenge's
# 100-task subset (per B100_task_misc.csv) from the full ~1018-activity
# bddl3/bddl/activity_definitions/ library, into their own directory.
#
# Usage:
#   ./scripts/extract_challenge_bddl.sh \
#     [--task-misc-csv <path to B100_task_misc.csv>] \
#     [--activity-definitions-dir bddl3/bddl/activity_definitions] \
#     [--output-dir challenge_bddl]
set -euo pipefail

TASK_MISC_CSV="datasets/2026-challenge-task-instances/metadata/B100_task_misc.csv"
ACTIVITY_DEFINITIONS_DIR="bddl3/bddl/activity_definitions"
OUTPUT_DIR="challenge_bddl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-misc-csv) TASK_MISC_CSV="$2"; shift 2 ;;
    --activity-definitions-dir) ACTIVITY_DEFINITIONS_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ ! -f "$TASK_MISC_CSV" ]] && { echo "task misc csv not found: $TASK_MISC_CSV" >&2; exit 1; }
[[ ! -d "$ACTIVITY_DEFINITIONS_DIR" ]] && { echo "activity definitions dir not found: $ACTIVITY_DEFINITIONS_DIR" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

mapfile -t TASKS < <(python3 -c "
import csv
with open('$TASK_MISC_CSV', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        print(row['Task'])
")

echo "Found ${#TASKS[@]} challenge tasks in $TASK_MISC_CSV"

copied=0
missing=()
for task in "${TASKS[@]}"; do
  src="$ACTIVITY_DEFINITIONS_DIR/$task"
  if [[ ! -d "$src" ]]; then
    missing+=("$task")
    continue
  fi
  cp -r "$src" "$OUTPUT_DIR/"
  copied=$((copied + 1))
done

echo "Copied $copied/${#TASKS[@]} task BDDL definitions to $OUTPUT_DIR/"
if (( ${#missing[@]} > 0 )); then
  echo "WARNING: ${#missing[@]} tasks had no matching directory in $ACTIVITY_DEFINITIONS_DIR:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
fi