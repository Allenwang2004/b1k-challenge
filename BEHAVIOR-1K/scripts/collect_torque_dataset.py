#!/usr/bin/env python3 -u
"""
Phase 1 — Collect torque-augmented dataset (torque_aug_v0).

Runs N episodes on a BEHAVIOR task, recording per-step:
  • RGB video (head + both wrists)
  • Proprio vector (61-dim R1Pro challenge layout — matches official 2026 demos;
    joint efforts are NOT part of it, they live in the standalone channel below)
  • observation.joint_efforts  — standalone 28-dim effort channel (N·m)
  • observation.contact_left   — left gripper non-self contact flag
  • observation.contact_right  — right gripper non-self contact flag

Policy modes:
  --policy-host (not set)   →  zero-action (schema validation)
  --policy-host localhost   →  Pi0/openpi policy server via websocket

Server must be started first in the openpi env:
  cd b1k-baselines/baselines/openpi && source .venv/bin/activate
  uv run scripts/serve_b1k.py --task_name=picking_up_trash \\
      policy:checkpoint --policy.config=pi0_b1k --policy.dir=PATH/49999

Output: data/torque_aug_v0/  (LeRobot v3 format)

Usage:
  # Zero-action (validate schema):
  python scripts/collect_torque_dataset.py

  # Live policy:
  python scripts/collect_torque_dataset.py --policy-host localhost --policy-port 8000
"""
import argparse
import numpy as np
import os
import sys
import time
import yaml
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.envs.torque_aug_wrapper import TorqueAugDataWrapper
from omnigibson.eval.policies import WebsocketPolicy
from omnigibson.eval.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    TASK_NAMES_TO_INDICES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(REPO_ROOT, "data")
DATASET_NAME = "torque_aug_v0"


def log(msg: str) -> None:
    print(f"[collect {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="picking_up_trash")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps", type=int, default=1500,
                   help="Max steps per episode (default 1500 ≈ 2× human avg for trash task)")
    p.add_argument("--policy-host", default=None,
                   help="Host of the openpi policy server (e.g. localhost). Omit for zero-action.")
    p.add_argument("--policy-port", type=int, default=8000)
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false", default=True)
    return p.parse_args()


def build_env_config(task: str, max_steps: int) -> dict:
    cfg_path = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["task"]["activity_name"] = task
    cfg["task"]["termination_config"]["max_steps"] = max_steps
    cfg["env"]["flatten_action_space"] = False  # DataWrapper uses action_space[robot.name]
    cfg["env"]["flatten_obs_space"] = True       # LeRobotDataWrapper requires flat robot::sensor::modality keys
    cfg["env"]["external_sensors"] = []           # empty list so external_sensors returns {} not None
    # Force robot name to robot_r1 so obs keys match what B1KPolicyWrapper expects
    cfg["robots"][0]["name"] = "robot_r1"
    # RGB + proprio (61-dim challenge layout; joint efforts recorded separately by TorqueAugDataWrapper)
    cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
    cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())

    return cfg


# Camera roles for R1Pro, mirroring omnigibson/eval/r1pro.yaml eval.camera_sensor_names
# (ROBOT_CAMERA_NAMES was removed upstream in favor of the robot eval config).
R1PRO_CAMERA_SENSOR_NAMES = {
    "left_wrist": "robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1:right_realsense_link:Camera:0",
    "head": "robot_r1:zed_link:Camera:0",
}


def preprocess_obs_for_policy(obs: dict, robot, task_name: str) -> dict:
    """Prepare obs for the websocket policy server (mirrors eval.py _preprocess_obs)."""
    base_pose = robot.get_position_orientation()
    cam_rel_poses = []
    for sensor_name in R1PRO_CAMERA_SENSOR_NAMES.values():
        camera = robot.sensors.get(sensor_name) or robot.sensors[sensor_name.split(":", 1)[1]]
        direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
        if np.allclose(direct_cam_pose, np.zeros(16)):
            cam_rel_poses.append(
                th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
            )
        else:
            cam_pose = T.mat2pose(
                th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32)
            )
            cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
    obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
    obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[task_name]], dtype=th.int64)
    return obs


def run_episode(
    env: TorqueAugDataWrapper,
    robot,
    policy,
    task_name: str,
    max_steps: int,
    ep_idx: int,
) -> dict:
    log(f"  Episode {ep_idx}: resetting ...")
    obs, info = env.reset()
    if policy is not None:
        policy.reset()

    terminated = truncated = False
    step = 0

    while not (terminated or truncated) and step < max_steps:
        if policy is not None:
            # Policy mode: preprocess and query server
            policy_obs = preprocess_obs_for_policy(dict(obs), robot, task_name)
            action_tensor = policy.forward(obs=policy_obs)  # shape (23,)
            action = {robot.name: action_tensor}
        else:
            # Zero-action mode: schema validation / baseline
            action = {robot.name: th.zeros(env.action_space[robot.name].shape)}

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if step % 50 == 0:
            efforts = robot.get_joint_efforts()
            contact_l, _ = robot._find_gripper_contacts(arm="left")
            contact_r, _ = robot._find_gripper_contacts(arm="right")
            success = info.get("done", {}).get("success", False)
            log(
                f"    step {step:4d}/{max_steps}"
                f"  effort_norm={efforts.norm():.2f}"
                f"  contact=({len(contact_l) > 0},{len(contact_r) > 0})"
                f"  success={success}"
            )

    success = info.get("done", {}).get("success", False)
    result = {"steps": step, "terminated": terminated, "truncated": truncated, "success": success}
    log(f"  Episode {ep_idx} done — {step} steps, success={success}")
    return result


def main():
    args = parse_args()
    policy_mode = args.policy_host is not None

    log(f"Phase 1 collection: task={args.task}, episodes={args.episodes}, max_steps={args.steps}")
    log(f"Policy mode: {'websocket @ ' + args.policy_host + ':' + str(args.policy_port) if policy_mode else 'zero-action'}")
    log("Building env config ...")
    cfg = build_env_config(task=args.task, max_steps=args.steps)

    log("Launching OmniGibson — Isaac Sim init takes ~60-120 s ...")
    t0 = time.time()
    env = og.Environment(configs=cfg)
    log(f"Environment created in {time.time() - t0:.1f}s")

    robot = env.robots[0]
    log(f"Robot: {robot.name}  n_dof={robot.n_dof}  arms={robot.arm_names}")

    # Create policy if requested
    policy = None
    if policy_mode:
        log(f"Connecting to policy server at {args.policy_host}:{args.policy_port} ...")
        policy = WebsocketPolicy(host=args.policy_host, port=args.policy_port, allow_reconnect=False)
        log("Policy server connected.")

    os.makedirs(DATA_ROOT, exist_ok=True)
    dataset_path = os.path.join(DATA_ROOT, DATASET_NAME)
    log(f"Wrapping env with TorqueAugDataWrapper → {dataset_path}")

    try:
        wrapped = TorqueAugDataWrapper(
            env=env,
            output_path=DATASET_NAME,
            root_dir=DATA_ROOT,
            overwrite=args.overwrite,
            only_successes=False,
            flush_every_n_traj=1,
            task_name=args.task.replace("_", " "),
        )
        log(f"Dataset schema created. Collecting {args.episodes} episode(s) ...")

        results = []
        for ep in range(args.episodes):
            result = run_episode(
                wrapped, robot, policy,
                task_name=args.task,
                max_steps=args.steps,
                ep_idx=ep + 1,
            )
            results.append(result)

        if len(wrapped.current_traj_history) > 1:
            wrapped.flush_current_traj()

        log("Finalizing dataset ...")
        wrapped.close_dataset()

        total_steps = sum(r["steps"] for r in results)
        n_success = sum(1 for r in results if r["success"])
        print(f"\n{'='*60}", flush=True)
        print(f"Dataset:  {dataset_path}", flush=True)
        print(f"Episodes: {len(results)}   Steps: {total_steps}   Successes: {n_success}/{len(results)}", flush=True)
        print(f"Extra channels per frame:", flush=True)
        print(f"  observation.joint_efforts  ({robot.n_dof}-dim, N·m)", flush=True)
        for arm in robot.arm_names:
            print(f"  observation.contact_{arm}    (1-dim, contact flag)", flush=True)
        print(f"{'='*60}", flush=True)

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        log("Shutting down Isaac Sim ...")
        og.shutdown()
        log("Done.")


if __name__ == "__main__":
    main()
