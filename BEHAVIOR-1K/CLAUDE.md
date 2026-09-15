# CLAUDE.md

@AGENTS.md

---

## Research Objective

Two parallel tracks (only this fork covers Track A):

| Track | Sim | Goal |
|---|---|---|
| **A: Vision-centric contact reasoning + torque-augmented dataset** (this repo) | BEHAVIOR-1K / OmniGibson | Predict contact from visual input; build dataset that also records torque to compare vision-only vs. force-aware |
| B: Physically-grounded F/T estimation | RoboSuite / MuJoCo | Separate repo |

**Core experiment (Sophia's idea):** roll out a top BEHAVIOR Challenge model in OmniGibson, record **force/torque** signals alongside vision to build a dataset. BEHAVIOR is the vision-centric benchmark; the torque channel enables contrast with the RoboSuite force-aware work.

OmniGibson already exposes **Joint Efforts** (joint torques) as a built-in proprioception modality from `get_obs()`. Add torque/contact to the recorded observation — do not hack the physics engine.

---

## Key Files — Read Before Writing Anything

- **1st place policy (IliaLarchenko/behavior_submission on HuggingFace)** — π0.5-based, 26% q-score. Checkpoints public; served via `b1k-baselines/baselines/openpi/scripts/serve_b1k.py`. Primary rollout policy for data collection.
- **`challenge_submissions/` does not exist in this fork** — submission code lives in the teams' own repos.
- `OmniGibson/omnigibson/eval/eval.py` + `evaluator.py` — 2026 rollout/evaluation entry points (`python -m omnigibson.eval.eval`, argparse; the 2025-era `learning/` package was renamed `eval/` upstream).
- `OmniGibson/omnigibson/eval/policies.py` — policy loading (`WebsocketPolicy`, `LocalPolicy`); wire protocol in `eval/utils/network_utils.py` (port 8000, `/healthz`, msgpack).
- `OmniGibson/omnigibson/eval/r1pro.yaml` — bundled challenge robot config (controllers, `eval.camera_sensor_names`).
- `OmniGibson/omnigibson/eval/utils/obs_utils.py`, `eval_utils.py`, `dataset_utils.py` — observation packing + dataset utils; `eval_utils.PROPRIOCEPTION_INDICES` is now the 61-dim challenge state layout (no efforts inside — efforts come from `robot.get_joint_efforts()`).
- `OmniGibson/omnigibson/envs/torque_aug_wrapper.py` — fork-local `TorqueAugDataWrapper`: emits `observation.joint_efforts` + `observation.contact_left/right`.
- `OmniGibson/omnigibson/envs/data_wrapper.py` — data collection/playback wrapper; key for re-rolling trajectories while logging extra signals.
- `OmniGibson/scripts/learning/replay_obs.py` — 2026 raw-demo replay (reads `2026-challenge-task-instances` metadata). NOTE: stock settings produce physically meaningless efforts (visual-only physics) — see plan/journal.

**Proprioception:** `get_obs()` returns `proprio` per robot with Joint Positions, Velocities, **Efforts** (torques), base pose & velocities. Contact via object states (`touching`) and prim-level contact APIs (`physx_utils`, rigid-prim contact lists).

> **Always verify APIs in source before assuming they exist.** Follow imports via `omnigibson.lazy` for Isaac Sim modules (only available after `launch_app`).

---

## Branch Strategy

- `main` — upstream mirror (no direct commits here)
- `dev` — shared integration branch; branch off this (synced with upstream v3.9.0, 2026 edition)
- `feat/force-replay` — force/BDDL replay pipeline: re-record the 100 tasks with physics-on effort+contact+BDDL logging → `b1k-sim` container
- `feat/policy-training` — System 2 + ACT skill experts + openpi finetune configs → `b1k-policy-dev` container
- `feat/policy-eval` — eval harness usage + submission packaging → `b1k-policy-prod` container (challenge submission artifact)
- `feat/torque-dataset` — Phase 1 (legacy, pre-sync): torque/contact-augmented live collection
- `feat/contact-reasoning` — Phase 2: Charles Sosmeña's vision-centric modeling

Remotes: `origin` = **private** EduCappiello/BEHAVIOR-1K; `upstream` = StanfordVL/BEHAVIOR-1K (sync via `git fetch upstream && git merge` into a `sync/*` branch, then PR into `dev`). The old public fork is archived at EduCappiello/BEHAVIOR-1K-public-archive.

---

## Phase 1 Deliverable ✅

`data/torque_aug_v0/` in LeRobot format — **done** (3 eps schema validation; live-policy collection ongoing):
- `observation.joint_efforts` (N·m, 28-dim) — standalone torque channel
- `observation.contact_left/right` (binary flag) — gripper contact
- RGB (head + both wrists) + 256-dim proprio preserved

## Phase 2 — FD-VLA-style Contact Reasoning

Architecture reference: **FD-VLA** (arXiv:2602.02142). A Force Distillation Module (FDM) predicts a latent force token from RGB + robot state, supervised by `observation.joint_efforts` from `torque_aug_v0` during training, and sensor-free at inference. Directional attention masking keeps VLM semantics frozen. Ablation: vision-only baseline vs. FDM-augmented policy.

---

## Research journaling → `lerobot-journals/`

Experiments, evaluations, theory, and notable development in this repo are recorded in the shared
**research-journals hub** (single source of truth), not in scattered notes. This project's experiment
journal is **`EXPERIMENT_JOURNAL.md`** at the repo root (symlinked into `lerobot-journals/behavior-1k/`).
Add/update an entry (local numbering from `E00` = problem/hypothesis) whenever you run a collection/eval,
derive an insight, or make a significant change on `feat/contact-reasoning`. Roles: Eduardo supervises /
architecture, Charles Sosmeña implements — the journal is the shared experiment record. Conventions + the
per-project source map: `lerobot-journals/handoff.md`. Shared cross-project planner: `WEEKLY_PLAN.md`
(symlinked at the repo root).
