# Deployment guide — what runs where, and how

Three machines, one architecture: **the sim and the policy never share a
process.** They talk over the challenge WebSocket protocol (msgpack,
`/healthz`; `omnigibson/eval/utils/network_utils.py`). Any sim-capable box can
evaluate against any policy-serving box — including itself, within VRAM limits.

Companion docs: [docker/README.b1k-force.md](docker/README.b1k-force.md)
(container stack, image transfer, submission packaging),
[scripts/SETUP.md](scripts/SETUP.md) (native env setup + known pitfalls).

---

## Box 1 — Local workstation (RTX 3080 Ti 12 GB)

The only box that renders. Primary machine for eval rollouts, data
collection/replay with cameras, and BDDL/task debugging.

**Runs**
- OmniGibson / Isaac Sim 5.1, native (`behavior` conda env) or `b1k-sim`
  container — full camera rendering.
- The official eval harness: `OMNIGIBSON_HEADLESS=1 python -m
  omnigibson.eval.eval --task-name <task> --host <policy-host> --port <port> …`
- Policy containers (`b1k-policy-prod`/`-dev`) for protocol tests.
- openpi π0 inference — **GPU or sim, not both**. π0 (pi0_b1k) holds ~9.8 GB
  of the 12 GB card at 177 ms/inference, which starves Isaac. For local live
  rollouts, serve π0 on **CPU** (~12 s per 50-step chunk) and give Isaac the
  GPU — see "Serving a π0 checkpoint" below.

**Does not run**
- π0-on-GPU concurrently with the sim (12 GB is not enough for both).
- Training of any kind (VRAM).

**Assets**: `datasets/behavior-1k-assets` (v3.9.0) +
`datasets/2026-challenge-task-instances` are installed and version-gate clean.
Keep `warp-lang<1.13` in the behavior env (see SETUP.md).

---

## Box 2a — idlab1 (RTX PRO 6000 Blackwell, driver 595.x)

Driver 595.x crashes Isaac Sim 5.1, and we are **not** rolling it back —
this box never simulates. It is a GPU-serving and training box.

**Runs**
- Policy training: openpi finetunes, ACT skill experts, System-2 work
  (`b1k-policy-dev` container or native).
- π0 GPU serving at full quality (`temporal_ensemble` mode, one inference per
  control step) — the intended partner for two-box eval:
  serve here, then on the local box point the eval at it:
  `python -m omnigibson.eval.eval … --host <idlab-ip> --port 8123`.

**Does not run**
- Isaac Sim / OmniGibson (driver), containerized or not — the NVIDIA container
  toolkit injects the host driver into containers, so `b1k-sim` crashes here too.

## Box 2b — idlab2 (2× RTX 5090 32 GB, driver 570.211)

**Isaac Sim 5.1 validated here** (2026-07-17): full raw-demo replay of
turning_on_radio demo 10 ran clean on GPU 0 — driver 570 + Blackwell consumer
is a working sim combo. This is the replay/data-generation workhorse
(2.5× the local box's VRAM, camera replay ~4 steps/s at full challenge
resolution, camera-less ~10 steps/s).

**Setup that exists** (no root, no docker group — everything user-space):
- `~/miniforge3/envs/behavior` — byte-copy of the local validated env (envs are
  path-dependent: rsync only works because username/home match; invoke
  `~/miniforge3/envs/behavior/bin/python` directly, no conda needed).
- `~/Documents/dev/BEHAVIOR-1K` — fork checkout + `datasets/` with
  behavior-1k-assets 3.9.0, 2026-challenge-task-instances,
  **omnigibson-robot-assets and omnigibson.key** (see pitfall below).
- `~/b1k-data/2026-challenge-rawdata/task-0000/` — raw demo episodes (HF
  `behavior-1k/2026-challenge-rawdata`, ~5–15 MB each).

Replay commands (GPU 0; GPU 1 is usually someone's training):
```bash
cd ~/Documents/dev/BEHAVIOR-1K
CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES \
  ~/miniforge3/envs/behavior/bin/python OmniGibson/scripts/learning/replay_obs.py \
  --data_folder ~/b1k-data --demo_id 10 --output_format hdf5    # stock: RGB+depth+61-dim proprio
# fork: physics-on torque replay (controllers + contacts active, joint_qeffort recorded)
CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES \
  ~/miniforge3/envs/behavior/bin/python OmniGibson/scripts/learning/replay_torque.py \
  --data_folder ~/b1k-data --demo_id 10
```

**Pitfall discovered here**: `REGISTERED_ROBOTS` is not code — it is populated
by globbing `$DATA_PATH/*/models/*/*.yaml`, i.e. **`omnigibson-robot-assets`
must be present in `datasets/` or the robot registry is silently empty** and
scene merges fail with "Scene must have exactly one robot, found 0". Ship
`omnigibson.key` alongside or asset decryption fails later.

---

## Box 3 — Zettabyte (H100, SSL-whitelisted egress)

Datacenter GPU without RT cores: physics yes, rendering no.

**Runs**
- Policy training and π0 GPU serving (as Box 2).
- **Camera-less replay** in the `b1k-sim` container: re-rolling recorded
  trajectories with physics on for effort/contact/BDDL logging (Milestone 1
  pipeline; final validation smoke still pending on that box).

**Does not run**
- Anything needing sensor images: camera replay, video eval, visual data
  collection (no RT cores → Isaac's RTX renderer produces nothing).

**Getting things there**: registries are blocked, HuggingFace is whitelisted —
`./docker/transfer_image_hf.sh save|load <image> ecappiell/b1k-artifacts`.
Image tarballs also live on the local Windows volume:
`/media/ecappiell/C460CA2360CA1BD41/b1k-artifacts/images/`
(`b1k-sim` 14.9 GB, `b1k-policy-prod` 8.0 GB, `b1k-policy-dev` 8.2 GB).

---

## Serving a π0 checkpoint (b1k-baselines)

The 2025-era `serve_b1k.py` is broken against v3.9.0 (imports the removed
`omnigibson.learning`). Use the fork-added
`b1k-baselines/baselines/openpi/scripts/serve_b1k_2026.py` — no omnigibson
dependency; it shims the 61-dim 2026 proprio layout and vendors the 2026 wire
protocol:

```bash
cd ~/Documents/dev/b1k-baselines/baselines/openpi
# GPU box (ID lab / Zettabyte): full-quality serving
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/serve_b1k_2026.py \
    --dir <ckpt>/49999_radio --port 8123
# 12 GB box sharing the GPU with Isaac: CPU serving, chunked control
JAX_PLATFORMS=cpu .venv/bin/python scripts/serve_b1k_2026.py \
    --dir <ckpt>/49999_radio --port 8123 --control-mode receeding_horizon
```

Checkpoints are openpi/Orbax dirs (`params/` + `assets/<repo_id>/norm_stats.json`);
`--config pi0_b1k` is the default. The venv gotcha: it was originally created
from snap-VSCode's python and breaks when the snap updates — repair by
installing `uv python install 3.11` and re-pointing `.venv/pyvenv.cfg`'s
`home =` line (plus the three `bin/python*` symlinks) at
`~/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin`.

State-layout note: the model consumes 23 dims selected **by name**
(base_qvel 3, trunk_qpos 4, arm qpos 7+7, gripper widths 1+1) from whatever
proprio vector arrives, so the 2026 61-dim obs reproduces the training-time
state exactly. Actions are 23-dim absolute joint targets, 50-step chunks.

## Port map

| Port | What |
|---|---|
| 8000 | policy containers (`b1k-policy-*`), challenge submission default |
| 8123 | native openpi serve (`serve_b1k_2026.py`) |
| 8124 | ad-hoc container port-mapping used in smoke tests |
