#!/usr/bin/env python3
"""
Phase 0.3 — Verify joint-effort and gripper-contact signals are accessible at runtime.

Run with:
    OMNIGIBSON_HEADLESS=1 conda run -n behavior python scripts/probe_torque.py

Expected output:
  - Full proprioception dict with shapes (confirms joint_qeffort is present)
  - Min/max of joint efforts at settle time and during stepping
  - Per-arm gripper contact pairs (empty when no objects grasped, non-empty on contact events)
"""
import os
import sys
import time
import yaml
import torch as th
import omnigibson as og
from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES

TASK = "picking_up_trash"
N_SETTLE = 30   # physics steps to let objects settle before reading
N_STEPS = 50    # steps to run with zero action


def log(msg):
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Reading config ...")
    cfg_path = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["task"]["activity_name"] = TASK
    cfg["task"]["termination_config"]["max_steps"] = N_SETTLE + N_STEPS + 100
    cfg["env"]["flatten_action_space"] = True  # needed so action_space.shape is a plain tuple
    # Proprio only — no RGB/depth needed for this probe
    cfg["robots"][0]["obs_modalities"] = ["proprio"]
    cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())

    log(f"Launching OmniGibson (task={TASK}) — Isaac Sim init takes ~60-120 s ...")
    t0 = time.time()
    env = og.Environment(configs=cfg)
    log(f"Environment created in {time.time() - t0:.1f}s")

    robot = env.robots[0]
    log("Resetting environment ...")
    env.reset()
    log("Reset done")

    log(f"Settling physics for {N_SETTLE} steps ...")
    for i in range(N_SETTLE):
        og.sim.step()
        if (i + 1) % 10 == 0:
            log(f"  settled {i + 1}/{N_SETTLE}")

    # ── Proprioception layout ──────────────────────────────────────────────────
    log("Reading proprioception dict ...")
    proprio_dict = robot._get_proprioception_dict()
    print("\n=== Proprioception Dict (all available keys) ===")
    for k, v in proprio_dict.items():
        print(f"  {k:35s}  shape={tuple(v.shape)}  dtype={v.dtype}")

    efforts = proprio_dict["joint_qeffort"]

    print("\n=== Proprioception Dict (all available keys) ===", flush=True)
    for k, v in proprio_dict.items():
        marker = " <-- EFFORTS" if k == "joint_qeffort" else ""
        print(f"  {k:35s}  shape={tuple(v.shape)}  dtype={v.dtype}{marker}", flush=True)

    print(f"\n=== Joint Efforts at settle ({efforts.shape[0]} DOFs) ===", flush=True)
    print(f"  min={efforts.min():.4f}  max={efforts.max():.4f}  mean={efforts.mean():.4f}", flush=True)
    print(f"  values: {efforts.numpy().round(4)}", flush=True)

    # ── Step loop ──────────────────────────────────────────────────────────────
    log(f"Stepping {N_STEPS} frames with zero action ...")
    zero_action = th.zeros(env.action_space.shape)
    for i in range(N_STEPS):
        env.step(zero_action)
        if i % 10 == 9:
            e = robot._get_proprioception_dict()["joint_qeffort"]
            log(f"  step {i + 1:3d}/{N_STEPS}  effort  min={e.min():.3f}  max={e.max():.3f}  norm={e.norm():.3f}")

    # ── Gripper contact readout ────────────────────────────────────────────────
    log("Querying gripper contact (non-self) ...")
    print("\n=== Gripper Contact (per arm, non-self only) ===", flush=True)
    for arm in robot.arm_names:
        contact_paths, contact_links = robot._find_gripper_contacts(arm=arm)
        if contact_paths:
            print(f"  arm={arm}: {len(contact_paths)} external contact(s)", flush=True)
            for path in sorted(contact_paths):
                via = sorted(contact_links.get(path, set()))
                print(f"    object={path}", flush=True)
                print(f"    via finger links={via}", flush=True)
        else:
            print(f"  arm={arm}: no external contacts (expected at rest)", flush=True)

    log("Shutting down ...")
    og.shutdown()
    print("\nProbe complete — joint efforts and contact APIs confirmed accessible.", flush=True)


if __name__ == "__main__":
    main()
