# Environment Setup Guide

This project uses **two separate Python environments** that must never be mixed. They communicate over a WebSocket when collecting data with a live policy.

---

## Environment Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Terminal 1  ·  behavior conda env                              │
│  ─────────────────────────────────────────────────────────────  │
│  Python 3.11 · PyTorch 2.7 · lerobot 0.5.2 · OmniGibson 3.8   │
│  Isaac Sim 5.1 (NVIDIA Omniverse)                               │
│                                                                 │
│  Runs:  probe_torque.py                                         │
│         collect_torque_dataset.py  ◄──── WebSocket :8000 ───┐  │
└─────────────────────────────────────────────────────────────│──┘
                                                              │
┌─────────────────────────────────────────────────────────────│──┐
│  Terminal 2  ·  openpi Python venv                          │  │
│  ─────────────────────────────────────────────────────────  │  │
│  Python 3.11 · JAX 0.5 · lerobot 0.3.4 · openpi            │  │
│  b1k-baselines/baselines/openpi/.venv/                      │  │
│                                                             │  │
│  Runs:  serve_b1k.py  ──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Why two environments?

| Reason | Detail |
|---|---|
| **lerobot version conflict** | The behavior env needs `lerobot 0.5.2` (LeRobotDataset API). The openpi checkpoint server was built against `lerobot 0.3.4`. Incompatible APIs. |
| **PyTorch vs JAX** | Isaac Sim / OmniGibson runs on PyTorch. The Pi0 policy runs on JAX. |
| **Isaac Sim isolation** | Isaac Sim injects its own native libraries at startup. Foreign packages installed into the behavior env can break it silently (see Troubleshooting). |

---

## Path variables

Set these once before following any instructions in this guide. Adjust to match your local clone locations:

```bash
# Root of this repo (BEHAVIOR-1K)
export BEHAVIOR_1K="$HOME/path/to/BEHAVIOR-1K"

# Root of the b1k-baselines repo (https://github.com/StanfordVL/b1k-baselines)
export B1K_BASELINES="$HOME/path/to/b1k-baselines"

# Where you want to store downloaded policy checkpoints
export CKPT_DIR="$HOME/path/to/behavior_checkpoints"

# Shortcut to the openpi venv Python binary
export OPENPI_PYTHON="$B1K_BASELINES/baselines/openpi/.venv/bin/python"
```

---

## Environment 1 — `behavior` (conda)

**Purpose:** Run Isaac Sim, OmniGibson, and the data-collection scripts.

The `behavior` conda env is created by the repo's `setup.sh`:

```bash
cd $BEHAVIOR_1K
bash setup.sh --new-env behavior --omnigibson --bddl --dataset
```

**Key packages (after install):**

| Package | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.7+ (CUDA) |
| OmniGibson | 3.8.0 |
| Isaac Sim | 5.1.0 |
| lerobot | 0.5.2 |
| websockets | 16.0 |

### Activation

```bash
conda activate behavior
```

### Verification

```bash
conda activate behavior
python -c "import omnigibson; print('OmniGibson OK')"
python -c "from lerobot.datasets import LeRobotDataset; print('lerobot OK')"
```

### ⚠ Do NOT pip install packages here without checking first

Isaac Sim pins specific versions of `click`, `pillow`, `websockets`, and `typing_extensions`. Installing anything that upgrades these will break the simulator. If you accidentally upgrade them:

```bash
# Example: restore click to the version Isaac Sim requires
pip install "click==8.1.7"

# Always verify Isaac Sim still imports after any pip install
python -c "import omnigibson"
```

---

## Environment 2 — openpi venv

**Purpose:** Serve the Pi0 / π0.5 policy over a WebSocket.

**Location:** `$B1K_BASELINES/baselines/openpi/.venv/`

This venv is created by [b1k-baselines](https://github.com/StanfordVL/b1k-baselines) using `uv`. If you have not set it up yet:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd $B1K_BASELINES/baselines/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Key packages:**

| Package | Version |
|---|---|
| Python | 3.11 |
| JAX | 0.5+ |
| openpi | (local editable) |
| openpi_client | (local editable) |
| lerobot | 0.3.4 |

### Activation

```bash
# Option A: run scripts directly via the venv Python
$OPENPI_PYTHON scripts/serve_b1k.py ...

# Option B: activate the venv as a standard Python venv
source $B1K_BASELINES/baselines/openpi/.venv/bin/activate
python scripts/serve_b1k.py ...
```

### Verification

```bash
$OPENPI_PYTHON -c "import openpi; print('openpi OK')"
$OPENPI_PYTHON -c "import openpi_client; print('openpi_client OK')"
$OPENPI_PYTHON -c "from omnigibson.eval.datas import BehaviorLerobotDatasetMetadata; print('metadata OK')"
```

### ⚠ Known issue — broken Python symlink (uv-managed installs)

If `uv` stores its Python in a location that becomes stale (e.g. inside an application's sandboxed data directory that gets wiped on update), the `.venv/bin/python` symlink will break. To fix it:

```bash
# 1. Find a valid Python 3.11 binary that uv downloaded
find ~ -name "python3.11" -path "*/uv/python/*" 2>/dev/null

# 2. Re-point the symlink to a valid path (substitute the path you found above)
VALID_PYTHON="/path/to/uv/managed/cpython-3.11.x/bin/python3.11"
VENV_BIN="$B1K_BASELINES/baselines/openpi/.venv/bin"
ln -sf "$VALID_PYTHON" "$VENV_BIN/python3.11"
ln -sf "$VALID_PYTHON" "$VENV_BIN/python3"
ln -sf "$VALID_PYTHON" "$VENV_BIN/python"

# 3. Verify
"$VENV_BIN/python" -c "print('symlink fixed')"
```

---

## Checkpoint Setup

The policy server needs a trained checkpoint.

### Option A — b1k-baselines Pi0 baseline

Fine-tuned Pi0 checkpoints for individual tasks. Download instructions are in the [b1k-baselines openpi tutorial](https://github.com/StanfordVL/b1k-baselines/blob/main/tutorials/openpi.md).

Expected directory structure after download:

```
$B1K_BASELINES/openpi_<task_name>/
└── <step_number>/
    ├── assets/
    ├── params             ← model weights
    ├── train_state/
    └── _CHECKPOINT_METADATA
```

### Option B — IliaLarchenko 1st-place π0.5 (all 50 tasks)

Four checkpoints covering all 50 challenge tasks from the 2025 BEHAVIOR Challenge winning solution. Available at [IliaLarchenko/behavior_submission](https://huggingface.co/IliaLarchenko/behavior_submission).

Download **from inside the openpi venv** to avoid polluting the behavior env:

```bash
$OPENPI_PYTHON -m huggingface_hub.commands.huggingface_cli download \
  IliaLarchenko/behavior_submission \
  --include "checkpoint_2/*" \
  --local-dir $CKPT_DIR/
```

Checkpoint coverage:

| Checkpoint | Task IDs covered |
|---|---|
| checkpoint_1 | 2,3,5,6,10,11,13,14,15,19,23,24,25,28,29,34,42,44,47,48 |
| checkpoint_2 | 0,1,7,8,9,12,16,17,18,20,21,22,26,30,43,45 |
| checkpoint_3 | 4,27,31,32,33,35,36,37,38,39,41,46,49 |
| checkpoint_4 | 40 |

Full task name → ID mapping: see `TASK_NAMES_TO_INDICES` in `OmniGibson/omnigibson/eval/utils/eval_utils.py`.

---

## Two-Terminal Workflow

Every live-policy collection run requires both terminals active simultaneously.

### Terminal 1 — Policy server (openpi venv)

```bash
cd $B1K_BASELINES/baselines/openpi

# Pi0 baseline checkpoint
$OPENPI_PYTHON scripts/serve_b1k.py \
  --task_name=<task_name> \
  policy:checkpoint \
  --policy.config=pi0_b1k \
  --policy.dir=<path_to_checkpoint_step_dir>

# IliaLarchenko 1st-place (no --policy.config needed)
$OPENPI_PYTHON scripts/serve_b1k.py \
  --task_name=<task_name> \
  policy:checkpoint \
  --policy.dir=$CKPT_DIR/checkpoint_2
```

Wait for:
```
INFO - Listening on 0.0.0.0:8000
```

### Terminal 2 — Data collection (behavior env)

```bash
conda activate behavior
cd $BEHAVIOR_1K

python scripts/collect_torque_dataset.py \
  --task <task_name> \
  --episodes 3 \
  --steps 1500 \
  --policy-host localhost \
  --policy-port 8000
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'LeRobotDataset' from 'lerobot.datasets'` | Running collect script in the openpi venv (wrong env) | Use `conda activate behavior` |
| `AttributeError: 'NoneType' object has no attribute 'items'` | `external_sensors=None` in env config | Set `external_sensors: []` (empty list, not null) |
| `IndexError: list index out of range` in `get_lerobot_obs_mapping` | `flatten_obs_space=False` in env config | Set `flatten_obs_space=True` |
| Isaac Sim version error after `pip install` | Incompatible package version installed in behavior env | Restore the pinned version (see behavior env section) |
| `.venv/bin/python: No such file or directory` | Broken uv Python symlink | Re-link using the fix in the openpi venv section |
| `Segmentation fault` at end of collect script | Isaac Sim native cleanup crash after a Python exception | Look for the Python traceback printed above the segfault — that is the real error |
| `uv: command not found` | `uv` not installed or not in PATH | Install via `curl -LsSf https://astral.sh/uv/install.sh | sh`, or use `.venv/bin/python` directly |

## Known env pitfalls (post v3.9.0 sync)

- **warp-lang must stay <1.13** in the `behavior` env: Isaac Sim 5.1 extensions
  break on warp 1.13+ (`module 'warp.types' has no attribute 'array'` at
  startup). Installing `OmniGibson[eval]` (whose pinned lerobot has loose deps)
  can silently bump it — if Isaac starts spewing warp import errors, run
  `pip install "warp-lang<1.13"`.
- The lerobot pin also upgrades pillow/packaging/click/typing_extensions past
  isaacsim's declared pins — those produce pip warnings but are benign in
  practice (validated by a full eval rollout).
- The wensi-ai lerobot pin reports version 0.5.2 (same as PyPI's): pip may
  consider it already satisfied. Force it with
  `pip install --force-reinstall --no-deps "lerobot @ git+https://github.com/wensi-ai/lerobot@release/b1k"`.
