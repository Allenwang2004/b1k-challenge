# Fork container stack — simulation / policy split

Two independent container families, communicating only over the WebSocket
policy protocol (port 8000, `/healthz`, msgpack — `omnigibson/eval/utils/network_utils.py`).
This mirrors the official 2026 challenge architecture: submissions are policy
containers; OmniGibson always runs outside them.

| Image | Dockerfile | Contents | Runs on |
|---|---|---|---|
| `b1k-sim` | `docker/sim.Dockerfile` | Isaac Sim 5.1 + OmniGibson (fork overlay on `stanfordvl/behavior:3.9.0`) | RTX GPU, driver 580.65.06+ (NOT 595.x). H100: camera-less replay only |
| `b1k-policy-dev` | `docker/policy.Dockerfile` (target `dev`) | torch cu124 + OmniGibson[eval] + training tooling; `b1k-baselines` mounted at runtime | any CUDA host |
| `b1k-policy-prod` | `docker/policy.Dockerfile` (target `prod`) | slim serving image — the challenge submission artifact (≤ 24 GB VRAM policies) | any CUDA host |

## Why the split

- Every `import` problem in `CHANGES_SUMMARY.md` came from mixing the Isaac/OmniGibson
  stack with the policy stack — the images make the boundary physical.
- The policy family never touches Isaac Sim → immune to the NVIDIA-driver/Isaac
  compatibility problem (595.x crash) and runs on H100.
- **What containers do NOT fix**: the host driver. The NVIDIA container toolkit
  injects the host's driver user-space into every container, so Isaac Sim's
  595.x crash reproduces inside Docker. Sim hosts must be on a validated
  driver branch (580.65.06 Linux; open kernel modules on Blackwell).

## Per-server placement

| Host | GPU / driver | Sim container | Policy containers |
|---|---|---|---|
| Local workstation | RTX 3080 Ti, working driver | ✅ (primary eval/replay box) | ✅ (12 GB — small policies only) |
| ID lab | RTX PRO 6000 Blackwell, driver 595.x | ❌ (595.x crashes Isaac Sim; staying on 595) | ✅ training + serving |
| Zettabyte | H100, SSL-whitelisted egress | camera-less replay only (validate with smoke test) | ✅ training + serving |

## Zettabyte transfer (registries blocked, HF allowed)

Build images on a machine with normal egress, then ship via a private HF repo:

```bash
# builder machine
./docker/transfer_image_hf.sh save b1k-policy-dev ecappiell/b1k-artifacts
# zettabyte
./docker/transfer_image_hf.sh load b1k-policy-dev ecappiell/b1k-artifacts
```

Assets download directly from HF on any machine
(`behavior-1k/zipped-datasets`): `behavior-1k-assets`, `omnigibson-robot-assets`,
`2026-challenge-task-instances` — unpack into the directory mounted at `/data`.

## Local two-container smoke test

```bash
docker compose up -d policy          # zero-action LocalPolicy server on :8000
curl -sf http://localhost:8000/healthz && echo healthy
docker compose run sim python -m omnigibson.eval.eval \
  --task-name turning_on_radio --host policy --port 8000 \
  --instance-indices 0 --num-rollouts 1 --output-dir /data/outputs/eval
```

## Submission packaging (challenge)

`b1k-policy-prod` with the default CMD replaced by your checkpoint server is
the submission. Constraints from behavior.stanford.edu/challenge/submission.html:
single 24 GB VRAM GPU (eval hardware: RTX 3090 / A5000 / TitanRTX), websocket
server on the advertised port with `/healthz`, challenge-track observations
only (RGB + depth + proprio via `RGBDFullResWrapper`), robot config
(`omnigibson/eval/r1pro.yaml` or your custom YAML) shipped with the
submission, `--write-video` outputs for instances 0–9.

## Branch ↔ image mapping

- `feat/force-replay` → `b1k-sim` (force/BDDL replay + re-recording)
- `feat/policy-training` → `b1k-policy-dev`
- `feat/policy-eval` → `b1k-policy-prod`

Upstream's `docker/Dockerfile` (sim base) and `build_docker.sh` are kept
untouched for upstream syncs; `docker/submission.Dockerfile` is superseded by
`policy.Dockerfile` (upstream's copy is broken on v3.9.0: imports the removed
`omnigibson.learning` package and installs `-e bddl` instead of `bddl3`).
