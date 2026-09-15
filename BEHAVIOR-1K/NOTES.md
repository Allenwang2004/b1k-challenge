# Open Questions & Notes

## Phase 0 Status

- [x] 0.1 Install — pre-existing `behavior` conda env, verify with `scripts/probe_torque.py`
- [x] 0.3 `scripts/probe_torque.py` — run to confirm joint efforts + contact APIs work
- [x] 0.4 `CLAUDE.md` written at repo root
- [ ] 0.2 Smoke-test a policy rollout (confirm checkpoint availability; see Q1 below)
- [ ] 0.5 Resolve open questions below before scaling Phase 1

---

## Open Questions

### Q1 — Challenge policy checkpoint availability
Are the checkpoints for the top submissions publicly downloadable?
- **Robot Learning Collective (1st)**: paper is `arxiv 2512.06951`; check if weights are hosted.
- **Comet (2nd, post-challenge)**: code mirrored at `github.com/mli0603/openpi-comet`; check for released weights.
- **Fallback**: use the repo's built-in action primitives (`--primitives` install flag) to generate trajectories if checkpoints are not public. This unblocks Phase 1 dataset collection at the cost of trajectory quality.

### Q2 — Joint effort units and gravity compensation
`robot.get_joint_efforts()` calls `get_measured_joint_efforts()` from the Isaac Sim articulation view.
- Confirm: are these **applied** torques or **measured** (reaction) torques?
- Confirm units: N·m for rotational joints?
- Are gravity-compensation torques included or stripped? Relevant for interpreting effort spikes at contact vs. effort at rest.
- **How to check**: compare `joint_qeffort` values at rest vs. arm fully extended (gravity torque on a long lever arm should show clear signal).

### Q3 — Contact: joint-effort signal vs. direct contact-force readout
Two options for capturing gripper contact:
1. **Joint efforts** (`joint_qeffort[gripper_control_idx]`) — already in proprio, simple.
2. **Direct contact force** via `robot._find_gripper_contacts()` / `RigidContactAPI.get_contact_pairs()` — gives object identity + contact location.
- Decision: **record both** in `torque_aug_v0`. Effort = continuous signal for training; contact pairs = ground-truth label for supervision.

### Q4 — Compute budget for batch rollouts
Long-horizon tasks average ~6.6 min each (per BEHAVIOR benchmark paper). For Phase 1:
- Start with 1 task × 5–10 episodes to validate the schema.
- Contact-rich tasks to prioritize: `slicing_vegetables`, `chopping_apples`, `wiping_countertop`, `picking_up_trash` (baseline).
- Use `eval_with_jobqueue.py` for parallelized batch collection once schema is validated.

### Q5 — `rich_obs_wrapper.py` reuse
`RichObservationWrapper` adds `normal` + `flow` modalities and task obs. Check if it already captures any effort/contact data before extending it. If not, it is the cleanest insertion point for Phase 1 augmentation (adds per-step extra obs without touching core sim).

---

## Handoff Notes for Charles Sosmeña — Active (2026-06-27)

**Roles:** Charles Sosmeña leads implementation for both Phase 1 (dataset scale-out) and Phase 2 (contact reasoning). Eduardo supervises and provides high-level architecture; more hands-on during Phase 2.

1. Clone `feat/contact-reasoning` — it has all of Phase 0 + Phase 1 already merged in.
2. Read `scripts/SETUP.md` before running anything — two environments are required.
3. Run `OMNIGIBSON_HEADLESS=1 python scripts/probe_torque.py` to confirm your env works.
4. Complete the live-policy test (see §5.1 of HANDOFF.md) — this is the first unblocked task.
5. Pick 2–3 contact-rich tasks for Phase 1 scale-out and discuss with Eduardo.
6. Read both papers before Phase 2: 1st-place (arXiv:2512.06951) for the rollout policy backbone, FD-VLA (arXiv:2602.02142) for the Phase 2 architecture.
7. Discuss Phase 2 architecture with Eduardo before implementing — he provides the high-level design.
