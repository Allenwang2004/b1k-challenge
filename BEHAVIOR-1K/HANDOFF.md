# BEHAVIOR-1K — Fork Setup & Handoff

**Fork:** `EduCappiello/BEHAVIOR-1K` (branched from upstream `StanfordVL/BEHAVIOR-1K` main, which is ahead of v3.7.2)  
**Last updated:** 2026-06-27

### Roles

| Person | Role |
|---|---|
| **Charles Sosmeña** | Implementation lead — owns Phase 1 dataset scale-out and Phase 2 contact reasoning end-to-end. Runs experiments, debugs, iterates. |
| **Eduardo** | Supervisor — provides concepts, high-level architecture decisions, and code review. More hands-on in Phase 2 (contact reasoning). |
| **Dr. Lien / Sophia** | PI — sets research direction and deliverable targets. |

---

## 0. Objective

Two parallel tracks:

| Track | Owner | Sim | Goal |
|---|---|---|---|
| **Vision-centric contact reasoning + torque dataset** (this repo) | **Charles Sosmeña** (implementation) · Eduardo (supervision) | BEHAVIOR-1K / OmniGibson | Predict contact from visual input; dataset records torque to enable vision-only vs. force-aware comparison |
| Physically-grounded F/T estimation | Eduardo | RoboSuite / MuJoCo | Separate repo, not tracked here |

**Core experiment (Sophia's idea):** roll out a top BEHAVIOR Challenge policy in OmniGibson, record joint efforts (torques) alongside RGB + proprioception. BEHAVIOR provides the vision-centric benchmark; the torque channel enables contrast with the force-aware work.

---

## 1. Branch Strategy

All branches are pushed to `EduCappiello/BEHAVIOR-1K`:

| Branch | Purpose |
|---|---|
| `main` | Clean upstream mirror — no direct commits |
| `dev` | Shared integration branch — branch off this |
| `feat/torque-dataset` | **Active** — Phase 0 + Phase 1 work lives here |
| `feat/contact-reasoning` | Charles Sosmeña's Phase 2 modeling work (empty, ready) |

```bash
git clone https://github.com/EduCappiello/BEHAVIOR-1K.git
cd BEHAVIOR-1K
git remote add upstream https://github.com/StanfordVL/BEHAVIOR-1K.git
git checkout feat/torque-dataset    # start here
```

---

## 2. What's in the Repo Now

### Scripts (`scripts/`)

| File | Purpose | Status |
|---|---|---|
| `scripts/probe_torque.py` | Phase 0.3 — validates joint effort + contact APIs in OmniGibson | ✅ Runs, confirmed working |
| `scripts/collect_torque_dataset.py` | Phase 1 — collects torque-augmented LeRobot dataset | ✅ Runs in zero-action mode; policy-server mode set up |
| `scripts/README.md` | How to run the scripts (modes, flags, verification) | ✅ Written |
| `scripts/SETUP.md` | Two-environment architecture, checkpoint download, troubleshooting | ✅ Written |

### New source files

| File | Purpose |
|---|---|
| `OmniGibson/omnigibson/envs/torque_aug_wrapper.py` | `TorqueAugDataWrapper` — subclass of `LeRobotDataWrapper` that injects `observation.joint_efforts` and `observation.contact_{arm}` into every recorded frame |
| `OmniGibson/omnigibson/learning/datas.py` | `BehaviorLerobotDatasetMetadata` — minimal stub needed by `serve_b1k.py` to resolve task text prompts without requiring the full dataset download |

### Modified source files

| File | Change |
|---|---|
| `OmniGibson/omnigibson/envs/lerobot_data_wrapper.py` | Made lerobot imports lazy (inside methods) so the module can be imported in the openpi venv (lerobot 0.3.4) without crashing |
| `OmniGibson/omnigibson/envs/__init__.py` | Added `TorqueAugDataWrapper` export |

### Documentation

| File | Purpose |
|---|---|
| `CLAUDE.md` | Research context + file map for Claude Code / agents |
| `NOTES.md` | Open questions, Phase 0 status, handoff checklist |
| `HANDOFF.md` | This file |

### Dataset

`data/torque_aug_v0/` — **3 episodes, 600 frames** collected with zero actions (schema validation mode):

```
features:
  action                               (23,)   float32
  observation.state                   (256,)   float32  ← full proprio incl. efforts at [112:140]
  observation.joint_efforts            (28,)   float32  ← NEW: standalone torque channel (N·m)
  observation.contact_left              (1,)   float32  ← NEW: left gripper contact flag
  observation.contact_right             (1,)   float32  ← NEW: right gripper contact flag
  observation.rgb.zed_link_camera_0  (128,128,3)  video  ← head camera
  observation.rgb.left_realsense_...  (128,128,3)  video  ← left wrist
  observation.rgb.right_realsense_... (128,128,3)  video  ← right wrist
```

---

## 3. Phase Status

### Phase 0 — Basis ✅ Complete

| Item | Status | Notes |
|---|---|---|
| 0.1 Install | ✅ | `behavior` conda env pre-existing, verified |
| 0.2 Smoke-test policy rollout | ⚠️ Partial | Infrastructure set up (see §5); not yet run end-to-end with live policy due to server test pending |
| 0.3 `scripts/probe_torque.py` | ✅ | `joint_qeffort` shape=(28,) confirmed, contact API confirmed |
| 0.4 `CLAUDE.md` | ✅ | At repo root |
| 0.5 `NOTES.md` + open questions | ✅ | Updated below |

### Phase 1 — Torque-augmented dataset ✅ Functional (Charles Sosmeña continues)

Eduardo set up the infrastructure. **Charles Sosmeña takes over from the live-policy test onward.**

| Item | Status | Owner |
|---|---|---|
| `TorqueAugDataWrapper` | ✅ done | Eduardo |
| LeRobot v3 dataset schema | ✅ done — `data/torque_aug_v0/` written and verified | Eduardo |
| Zero-action collection (schema validation) | ✅ done — 3 episodes × 200 steps | Eduardo |
| Live-policy collection (Pi0 / π0.5) | ⏳ server setup complete; needs live run | **Charles Sosmeña** |
| Scale to contact-rich tasks | ⏳ pending live-policy validation | **Charles Sosmeña** |
| `validate_dataset.py` + effort-vs-contact plots | ⏳ not yet written | **Charles Sosmeña** |

### Phase 2 — FD-VLA-style contact reasoning ⏳ Not started (Charles Sosmeña leads, Eduardo reviews)

Branch `feat/contact-reasoning` is ready. Architecture follows **FD-VLA** (arXiv:2602.02142):
- FDM predicts a force token from RGB + state (supervised by `observation.joint_efforts` from `torque_aug_v0`)
- Directional attention masking keeps the VLM's perceptual stream frozen
- At inference: force-aware without physical sensors

Ablation compares vision-only baseline vs. FD-VLA with force distillation. Eduardo provides architecture direction; Charles Sosmeña implements and runs experiments.

---

## 4. Key Discoveries (not in original plan)

### 4.1 IliaLarchenko = Robot Learning Collective (1st place)

The 1st-place team published their code and checkpoints publicly:
- **GitHub:** https://github.com/IliaLarchenko/behavior-1k-solution
- **HuggingFace:** https://huggingface.co/IliaLarchenko/behavior_submission (4 checkpoints, all 50 tasks)
- Serve command: `$OPENPI_PYTHON scripts/serve_b1k.py policy:checkpoint --policy.dir <ckpt_dir>` (no `--policy.config`)

### 4.2 FD-VLA — the Phase 2 architecture

**FD-VLA** (NUS/A*STAR, arXiv:2602.02142) is the architecture to follow for Phase 2. Core idea: a **Force Distillation Module (FDM)** predicts a latent force token from RGB + robot state, supervised by actual force signals during training, and sensor-free at inference. The distilled token is injected into the VLM via directional attention masking so perceptual stream semantics are preserved.

This fits directly onto `torque_aug_v0`: `observation.joint_efforts` serves as the actual force supervision signal for the FDM, and RGB + `observation.state` are the vision/state inputs. At inference, the FDM predicts contact without needing a physical sensor.

**TA-VLA** (arXiv:2509.07962) is a separate paper for Eduardo's RoboSuite/MuJoCo track — it uses estimated forces from motor load as direct input, not relevant to this repo.

### 4.3 Two-environment architecture required

The policy server (openpi, JAX, lerobot 0.3.4) and the OmniGibson simulator (PyTorch, lerobot 0.5.2, Isaac Sim) cannot share a Python environment. They communicate over WebSocket on port 8000. Full details in `scripts/SETUP.md`.

### 4.4 `challenge_submissions/` does not exist in this fork

The original plan referenced `submission_robot_learning_collective.py` etc. These files are not in the upstream repo. The submission code is in the teams' own repos (see §4.1 for 1st place).

### 4.5 Pi0 baselines available from b1k-baselines

`github.com/StanfordVL/b1k-baselines` provides Pi0 checkpoints for individual tasks (not the π0.5 1st-place model). The `turning_on_radio` checkpoint is already downloaded locally. See `scripts/SETUP.md → Checkpoint Setup`.

---

## 5. What Needs to Happen Next

All items below are **Charles Sosmeña's responsibility** to run and debug. Eduardo is available for architecture decisions and code review, and more hands-on during Phase 2.

### 5.1 Complete the live-policy test (unblocked, do first)

This is the one item from Phase 0.2 that is set up but not yet run end-to-end. Once it passes, commit all pending changes.

**Terminal 1 — Policy server:**
```bash
cd $B1K_BASELINES/baselines/openpi
$OPENPI_PYTHON scripts/serve_b1k.py \
  --task_name=turning_on_radio \
  policy:checkpoint \
  --policy.config=pi0_b1k \
  --policy.dir=$B1K_BASELINES/openpi_turning_on_radio/49999_radio
```

**Terminal 2 — Collection:**
```bash
conda activate behavior
cd $BEHAVIOR_1K
python scripts/collect_torque_dataset.py \
  --task turning_on_radio \
  --episodes 1 \
  --steps 1500 \
  --policy-host localhost \
  --policy-port 8000
```

Expected: script runs, dataset frames accumulate, summary prints at end, no Python exceptions. Once confirmed → commit + push.

### 5.2 Scale Phase 1 data collection

After the live-policy test passes, collect real data across contact-rich tasks. Priority order:

1. `turning_on_radio` — already have checkpoint, start here
2. `picking_up_trash` — checkpoint available ([Google Drive](https://drive.google.com/file/d/1G_ACu3uUP_9RmXDgqa7307aFt28G-vJN/view))
3. `slicing_vegetables` — contact-rich, use IliaLarchenko checkpoint_2

Use `--no-overwrite` to append episodes across runs. Target: 10+ episodes per task before calling the dataset v0 complete.

### 5.3 Write `scripts/validate_dataset.py`

Per original plan §4 item 5: load `torque_aug_v0`, plot `joint_qeffort` norm vs. `observation.contact_left` for one episode. Spike alignment = sanity check that the signals are physically meaningful.

### 5.4 Start Phase 2 scaffold

Work on `feat/contact-reasoning`. Architecture follows FD-VLA (arXiv:2602.02142). Discuss design with Eduardo before implementing. Proposed scaffold:

- `contact_reasoning/data.py` — LeRobot dataset loader for `torque_aug_v0`; loads RGB, state, and `observation.joint_efforts`
- `contact_reasoning/fdm.py` — Force Distillation Module: learnable query attends to vision + state tokens, predicts force token; supervised by `observation.joint_efforts` during training (Eq. 1 & 5 in paper)
- `contact_reasoning/model.py` — full model: VLM backbone (SmolVLA or π0) + FDM + directional attention masking + action expert
- `contact_reasoning/train.py` — combined loss: flow matching (policy) + distillation (L2 force token alignment)

Baseline comparison: same model without FDM (vision-only) vs. FD-VLA with FDM. The torque data we collect is the supervision signal that enables this ablation cleanly.

---

## 6. Open Questions — Updated

| # | Question | Status |
|---|---|---|
| Q1 | Are challenge policy checkpoints public? | ✅ **Yes.** IliaLarchenko published all 4 checkpoints on HuggingFace. Pi0 baselines from b1k-baselines also available. |
| Q2 | Joint effort units + gravity compensation | ⏳ `get_measured_joint_efforts()` from Isaac Sim articulation view. Units appear to be N·m. Gravity comp status unconfirmed — check probe output at rest vs. arm extended. |
| Q3 | Joint effort vs. direct contact force | ✅ **Decision made:** record both. `observation.joint_efforts` (continuous, 28-dim) + `observation.contact_{arm}` (binary flag via `_find_gripper_contacts()`). |
| Q4 | Compute budget for batch rollouts | ⏳ Tasks average ~6.6 min each. Start with 10 episodes × 2–3 tasks; use `eval_with_jobqueue.py` for scale-out. |
| Q5 | Does `rich_obs_wrapper` expose effort/contact? | ✅ **No.** Confirmed by reading source. `TorqueAugDataWrapper` is the correct hook — it augments `_process_obs()` at the data-wrapper level. |

---

## 7. Checklist for Charles Sosmeña — Active (2026-06-27)

**Setup**
- [ ] Clone `feat/contact-reasoning` (it has all of Phase 0 + Phase 1 merged in)
- [ ] Read `scripts/SETUP.md` — understand the two-environment architecture
- [ ] Run `OMNIGIBSON_HEADLESS=1 python scripts/probe_torque.py` — confirms your env is working

**Phase 1 completion (you own this)**
- [ ] Run the live-policy test (§5.1) — confirm the full collection pipeline end-to-end
- [ ] Scale data collection to 2–3 contact-rich tasks (§5.2) — pick which tasks to prioritize
- [ ] Write `scripts/validate_dataset.py` (§5.3) — plot effort vs. contact flag to sanity-check signals

**Phase 2 start**
- [ ] Read 1st-place paper (arXiv:2512.06951) for the π0.5 rollout policy backbone
- [ ] Read FD-VLA paper (arXiv:2602.02142) — this is the Phase 2 architecture
- [ ] Discuss Phase 2 design with Eduardo before implementing — he provides the high-level architecture
- [ ] Scaffold `contact_reasoning/` module on `feat/contact-reasoning` (§5.4)
