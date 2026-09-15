# BEHAVIOR-1K Scripts — Start Guide

Scripts for Phase 0 (validation) and Phase 1 (torque-augmented dataset collection) of the vision-centric contact reasoning project.

> **First time?** Read [SETUP.md](SETUP.md) first to configure your environments and path variables.

---

## Prerequisites

- `behavior` conda env installed and activated (see [SETUP.md](SETUP.md))
- NVIDIA RTX GPU (2080 Ti or better, 8 GB+ VRAM)
- OmniGibson assets downloaded (`bash setup.sh --omnigibson --bddl --dataset` from repo root)
- For live-policy collection: openpi venv set up and a checkpoint downloaded

### Path variables (set once per shell session)

```bash
export BEHAVIOR_1K="$HOME/path/to/BEHAVIOR-1K"
export B1K_BASELINES="$HOME/path/to/b1k-baselines"
export OPENPI_PYTHON="$B1K_BASELINES/baselines/openpi/.venv/bin/python"
export CKPT_DIR="$HOME/path/to/behavior_checkpoints"
```

### Headless vs. with viewer

```bash
OMNIGIBSON_HEADLESS=1 python scripts/probe_torque.py   # no display needed
python scripts/probe_torque.py                          # opens OmniGibson viewer
```

> Isaac Sim startup always takes 60–120 s. The wall of deprecation warnings is normal.

---

## `probe_torque.py` — Phase 0.3 Signal Validation

Loads a BEHAVIOR task, steps 50 frames, and confirms that the joint-effort and gripper-contact APIs work correctly. Run this first to verify the environment before attempting data collection.

**What it checks:**
- Full proprioception dict shapes (confirms `joint_qeffort` at shape `(28,)`)
- Joint effort values (min/max/norm) every 10 steps
- Per-arm gripper contact pairs at the end

### Run

```bash
conda activate behavior
cd $BEHAVIOR_1K
python scripts/probe_torque.py
```

### Expected output (success)

```
[probe HH:MM:SS] Environment created in ~60s
...
=== Proprioception Dict (all available keys) ===
  joint_qeffort   shape=(28,)  dtype=torch.float32  <-- EFFORTS
...
=== Joint Efforts at settle (28 DOFs) ===
  min=-9.56  max=15.14  mean=0.19
...
=== Gripper Contact (per arm, non-self only) ===
  arm=left:  no external contacts (expected at rest)
  arm=right: no external contacts (expected at rest)
Probe complete — joint efforts and contact APIs confirmed accessible.
```

---

## `collect_torque_dataset.py` — Phase 1 Dataset Collection

Runs N episodes and records a LeRobot v3 dataset to `data/torque_aug_v0/` with the standard BEHAVIOR-1K channels **plus** two extra:

| Channel | Shape | Units | Description |
|---|---|---|---|
| `observation.joint_efforts` | (28,) | N·m | Measured joint torques for all 28 R1Pro DOFs |
| `observation.contact_left` | (1,) | flag | `1.0` if left gripper has ≥1 non-self contact |
| `observation.contact_right` | (1,) | flag | `1.0` if right gripper has ≥1 non-self contact |

Standard channels also recorded: RGB (head + left/right wrist at 128×128), full 256-dim proprio vector.

---

### Mode A — Zero-action (schema validation, no server needed)

```bash
conda activate behavior
cd $BEHAVIOR_1K
python scripts/collect_torque_dataset.py \
  --task picking_up_trash \
  --episodes 3 \
  --steps 200
```

---

### Mode B — Pi0 policy (b1k-baselines checkpoint)

Two terminals required. See [SETUP.md](SETUP.md) for checkpoint download instructions.

#### Terminal 1 — Start the policy server

```bash
cd $B1K_BASELINES/baselines/openpi
$OPENPI_PYTHON scripts/serve_b1k.py \
  --task_name=turning_on_radio \
  policy:checkpoint \
  --policy.config=pi0_b1k \
  --policy.dir=<path_to_checkpoint_step_dir>
```

Wait for: `INFO - Listening on 0.0.0.0:8000`

#### Terminal 2 — Run collection

```bash
conda activate behavior
cd $BEHAVIOR_1K
python scripts/collect_torque_dataset.py \
  --task turning_on_radio \
  --episodes 3 \
  --steps 1500 \
  --policy-host localhost \
  --policy-port 8000
```

---

### Mode C — IliaLarchenko 1st-place checkpoints (π0.5, all 50 tasks)

Download from [HuggingFace](https://huggingface.co/IliaLarchenko/behavior_submission) using the openpi venv (see [SETUP.md](SETUP.md)):

```bash
$OPENPI_PYTHON -m huggingface_hub.commands.huggingface_cli download \
  IliaLarchenko/behavior_submission \
  --include "checkpoint_2/*" \
  --local-dir $CKPT_DIR/
```

#### Terminal 1 — Start the server (no `--policy.config` needed)

```bash
cd $B1K_BASELINES/baselines/openpi
$OPENPI_PYTHON scripts/serve_b1k.py \
  --task_name=picking_up_trash \
  policy:checkpoint \
  --policy.dir=$CKPT_DIR/checkpoint_2
```

#### Terminal 2 — Run collection (same as Mode B)

```bash
conda activate behavior
cd $BEHAVIOR_1K
python scripts/collect_torque_dataset.py \
  --task picking_up_trash \
  --episodes 3 \
  --steps 1500 \
  --policy-host localhost \
  --policy-port 8000
```

---

### All flags

| Flag | Default | Description |
|---|---|---|
| `--task` | `picking_up_trash` | BEHAVIOR activity name |
| `--episodes` | `3` | Number of episodes to collect |
| `--steps` | `1500` | Max steps per episode |
| `--policy-host` | *(not set)* | Host of the policy server. Omit for zero-action mode |
| `--policy-port` | `8000` | Port of the policy server |
| `--no-overwrite` | *(flag)* | Append to existing dataset instead of overwriting |

---

### Verify the output

```bash
python -c "
import json
with open('data/torque_aug_v0/meta/info.json') as f:
    info = json.load(f)
print('Episodes:', info['total_episodes'])
print('Frames:  ', info['total_frames'])
for k, v in info['features'].items():
    print(f'  {k}: shape={v[\"shape\"]} dtype={v[\"dtype\"]}')
"
```

---

## Task checkpoint reference

| Task | Task ID | IliaLarchenko checkpoint |
|---|---|---|
| `turning_on_radio` | 0 | checkpoint_2 |
| `picking_up_trash` | 1 | checkpoint_2 |
| `putting_away_Halloween_decorations` | 2 | checkpoint_1 |
| `slicing_vegetables` | 43 | checkpoint_2 |

Full task name → ID mapping: `OmniGibson/omnigibson/eval/utils/eval_utils.py` (`TASK_NAMES_TO_INDICES`).  
Full checkpoint coverage: [IliaLarchenko/behavior_submission](https://huggingface.co/IliaLarchenko/behavior_submission).
