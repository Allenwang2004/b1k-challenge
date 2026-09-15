"""Physics-on raw-demo replay that records joint efforts (torque).

Fork variant of replay_obs.py for the force-aware dataset track. Differences
from the stock script, all via existing DataPlaybackWrapper flags:

- include_robot_control=True: the robot's own controllers stay active, so the
  recorded action drives real joint drives during each playback step
  (stock playback zero-gains the controllers and the robot is state-teleported,
  which makes joint efforts physically meaningless).
- include_contacts=True: objects stay physical (no keep_still/visual-only), so
  robot-object contact produces real reaction forces. The wrapper then runs
  physics at 1000 Hz, meaning each step advances only 1 ms from the restored
  recorded state: efforts are teacher-forced — computed at the recorded
  (state, action) pair — and the trajectory cannot diverge from the demo.
- joint_qpos/qvel/qeffort appended to the recorded proprio keys
  (joint_qeffort = robot.get_joint_efforts(), full joint-space torque).
- Cameras off by default (--with-cameras to include RGB) — the torque channel
  does not need rendering and proprio-only replay is much faster.

Output: <data_folder>/replayed_torque/episode_<demo_id>.hdf5
"""

import argparse
import os

import omnigibson as og
from omnigibson.envs import HDF5PlaybackWrapper
from omnigibson.eval.utils.dataset_utils import makedirs_with_mode
from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger

from replay_obs import (
    _find_full_scene_file,
    _get_task_name_from_task_id,
    _infer_task_id_from_demo_id,
    _load_challenge_available_tasks,
    _load_room_instances,
)

log = create_module_logger(module_name="replay_torque")
log.setLevel(20)

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128


# Force-decomposition channels appended to the recorded proprio, all 28-dim,
# read from the same physics step as joint_qeffort (measured/net):
#   joint_qeffort_applied — drive/actuator torque only
#   joint_gravity         — generalized gravity forces g(q) (compensation sign)
#   joint_coriolis        — Coriolis/centrifugal c(q, q̇) (compensation sign)
# contact+friction contribution is the post-hoc residual of these channels.
DECOMP_KEYS = ["joint_qeffort_applied", "joint_gravity", "joint_coriolis"]


def install_decomposition_channels() -> None:
    """Class-level patch (must run before env creation so proprio-key validation
    at robot load sees the new keys)."""
    from omnigibson.robots.robot import Robot

    orig = Robot._get_proprioception_dict

    def patched(self):
        import torch as th

        dic = orig(self)
        try:
            av = self._articulation_view
            dic["joint_qeffort_applied"] = av.get_applied_joint_efforts().view(self.n_dof)
            dic["joint_gravity"] = av.get_generalized_gravity_forces().view(self.n_dof)
            dic["joint_coriolis"] = av.get_coriolis_and_centrifugal_forces().view(self.n_dof)
        except Exception:
            # physx view not ready (obs-space sizing before play) — zeros keep shapes stable
            for key in DECOMP_KEYS:
                dic[key] = th.zeros(self.n_dof)
        return dic

    Robot._get_proprioception_dict = patched


# Placeholder engineering estimates (N·m) pending vendor spec for R1Pro;
# matched by substring against joint names, first hit wins.
DEFAULT_EFFORT_CAPS = {"gripper": 30.0, "arm": 60.0, "torso": 150.0, "trunk": 150.0}
DEFAULT_EFFORT_CAP_FALLBACK = 300.0  # base/wheel and anything unmatched


def apply_effort_limits(robot, log) -> None:
    for joint_name, joint in robot.joints.items():
        cap = next(
            (v for pattern, v in DEFAULT_EFFORT_CAPS.items() if pattern in joint_name.lower()),
            DEFAULT_EFFORT_CAP_FALLBACK,
        )
        old = joint.max_effort
        joint.max_effort = cap
        log.info(f"effort cap: {joint_name}: {old} -> {cap} N·m")


def replay_torque(
    data_folder: str,
    demo_id: int,
    with_cameras: bool = False,
    flush_every_n_steps: int = 1000,
    use_longest_demo: bool = False,
    preserve_frequencies: bool = False,
    effort_limits: bool = False,
    suffix: str = "",
    output: str | None = None,
) -> str:
    task_id = _infer_task_id_from_demo_id(demo_id)
    task_name = _get_task_name_from_task_id(task_id)
    replay_dir = os.path.dirname(output) if output else os.path.join(data_folder, "replayed_torque")
    makedirs_with_mode(replay_dir)

    gm.ENABLE_TRANSITION_RULES = False

    available_tasks = _load_challenge_available_tasks()
    scene_model = available_tasks[task_name][0]["scene_model"]
    full_scene_file = _find_full_scene_file(task_name=task_name, scene_model=scene_model)
    load_room_instances = _load_room_instances(task_name=task_name)

    input_path = f"{data_folder}/2026-challenge-rawdata/task-{task_id:04d}/episode_{demo_id:08d}.hdf5"
    output_path = output or os.path.join(replay_dir, f"episode_{demo_id:08d}{suffix}.hdf5")

    install_decomposition_channels()
    proprio_keys = (
        list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        + ["joint_qpos", "joint_qvel", "joint_qeffort"]
        + DECOMP_KEYS
    )

    robot_sensor_config = {
        "VisionSensor": {"sensor_kwargs": {"image_height": 480, "image_width": 480}},
        "zed_link:Camera:0": {
            "sensor_kwargs": {"horizontal_aperture": 40.0, "image_height": 720, "image_width": 720}
        },
    }

    env = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=input_path,
        output_path=output_path,
        full_scene_file=full_scene_file,
        load_room_instances=load_room_instances,
        robot_sensor_config=robot_sensor_config,
        n_render_iterations=1,
        flush_every_n_steps=flush_every_n_steps,
        flush_every_n_traj=1,
        include_robot_control=True,
        include_contacts=True,
        preserve_recorded_frequencies=preserve_frequencies,
        robot_proprio_keys=proprio_keys,
        robot_obs_modalities=["proprio", "rgb"] if with_cameras else ["proprio"],
        compression={"compression": "lzf"},
    )
    env.load_observation_space()

    if effort_limits:
        apply_effort_limits(env.robots[0], log)

    demo_ids = sorted(int(key.split("_", 1)[1]) for key in env.input_hdf5["data"].keys() if key.startswith("demo_"))
    if not demo_ids:
        raise ValueError(f"No demo groups found in {input_path}")
    if use_longest_demo:
        episode_id = max(
            demo_ids, key=lambda eid: env.input_hdf5["data"][f"demo_{eid}"].attrs["num_samples"]
        )
    else:
        episode_id = demo_ids[-1]
    num_samples = env.input_hdf5["data"][f"demo_{episode_id}"].attrs["num_samples"]
    log.info(f" >>> Torque replay of episode {episode_id} with {num_samples} steps (control+contacts ON)")

    env.playback_episode(episode_id=episode_id, record_data=True)
    log.info("Playback complete. Saving data...")
    env.save_data()

    # Join-safety: stamp identity + channel layout so sidecars can never be
    # matched to the wrong demo, even if a file is renamed.
    import json as _json

    import h5py as _h5py

    n_dof = 28
    layout, cursor = {}, 0
    for key, width in [("challenge_61", 61)] + [
        (k, n_dof) for k in ("joint_qpos", "joint_qvel", "joint_qeffort", *DECOMP_KEYS)
    ]:
        layout[key] = [cursor, cursor + width]
        cursor += width
    with _h5py.File(output_path, "r+") as f:
        f["data"].attrs["task_id"] = task_id
        f["data"].attrs["demo_id"] = demo_id
        f["data"].attrs["task_name"] = task_name
        f["data"].attrs["replay_cell"] = "B-natural-frequency" if preserve_frequencies else "1kHz-microstep"
        f["data"].attrs["proprio_layout"] = _json.dumps(layout)
    log.info(f"Saved torque replay to {output_path} (proprio {cursor}-dim)")
    return output_path


def replay_torque_chunk(
    data_folder: str,
    demo_ids: list[int],
    output: str,
    use_longest_demo: bool = False,
    lean: bool = False,
) -> str:
    """Chained replay: one Isaac boot + one scene load, N episodes of the same
    task (verified: instances share the object set, so per-episode
    scene_file swaps only need state restore, which playback_episode already
    does). Output is a single multi-demo HDF5 (groups demo_0..demo_{N-1} in
    replay order) with a manifest attr mapping groups to raw demo_ids.

    Always cell B (natural frequencies) — the production configuration.
    """
    import json as _json

    import h5py as _h5py

    from omnigibson.envs.data_wrapper import _align_scene_object_states_with_recorded_schema

    task_id = _infer_task_id_from_demo_id(demo_ids[0])
    assert all(_infer_task_id_from_demo_id(d) == task_id for d in demo_ids), "chunk must be single-task"
    task_name = _get_task_name_from_task_id(task_id)
    makedirs_with_mode(os.path.dirname(output))

    gm.ENABLE_TRANSITION_RULES = False

    available_tasks = _load_challenge_available_tasks()
    scene_model = available_tasks[task_name][0]["scene_model"]
    full_scene_file = _find_full_scene_file(task_name=task_name, scene_model=scene_model)
    load_room_instances = _load_room_instances(task_name=task_name)

    def raw_path(demo_id):
        return f"{data_folder}/2026-challenge-rawdata/task-{task_id:04d}/episode_{demo_id:08d}.hdf5"

    install_decomposition_channels()
    proprio_keys = (
        list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        + ["joint_qpos", "joint_qvel", "joint_qeffort"]
        + DECOMP_KEYS
    )

    # --lean: proprio-only replay needs no render products at all. The recorded
    # env config re-creates three 1080x1080 external cameras and the robot's
    # cameras otherwise — pure VRAM/render waste here. Physics is untouched.
    lean_kwargs = dict(external_sensors_config=[], exclude_sensor_names=["Camera"]) if lean else {}

    env = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=raw_path(demo_ids[0]),
        output_path=output,
        full_scene_file=full_scene_file,
        load_room_instances=load_room_instances,
        robot_sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": 480, "image_width": 480}}},
        n_render_iterations=1,
        **lean_kwargs,
        # whole-episode writes: the incremental-flush path breaks on the second
        # playback_episode in one process (upstream only ever replays one per
        # process); proprio-only episodes are ~2 MB so buffering is free
        flush_every_n_steps=0,
        flush_every_n_traj=1,
        include_robot_control=True,
        include_contacts=True,
        preserve_recorded_frequencies=True,
        robot_proprio_keys=proprio_keys,
        robot_obs_modalities=["proprio"],
        compression={"compression": "lzf"},
    )
    env.load_observation_space()

    manifest = []
    for i, demo_id in enumerate(demo_ids):
        if i > 0:
            env.input_hdf5.close()
            env.input_hdf5 = _h5py.File(raw_path(demo_id), "r")
            recorded = _json.loads(env.input_hdf5["data"].attrs["scene_file"])
            env.recorded_scene_file = recorded
            # Baseline for playback_episode's scene.restore must be the LIVE
            # scene's own save (matches create_from_hdf5's load_room_instances
            # path): recorded scene jsons can predate current robot state keys
            # (e.g. controller_groups), and per-step recorded states are loaded
            # from the hdf5 vectors afterwards anyway — restore only needs a
            # schema-consistent baseline, not the new episode's initial state.
            env.scene_file = env.scene.save(as_dict=True)
            _align_scene_object_states_with_recorded_schema(scene=env.scene, recorded_scene_file=recorded)

        eps = sorted(int(k.split("_", 1)[1]) for k in env.input_hdf5["data"].keys() if k.startswith("demo_"))
        episode_id = (
            max(eps, key=lambda e: env.input_hdf5["data"][f"demo_{e}"].attrs["num_samples"])
            if use_longest_demo
            else eps[-1]
        )
        log.info(f" >>> [{i + 1}/{len(demo_ids)}] chained torque replay of demo {demo_id}")
        env.playback_episode(episode_id=episode_id, record_data=True)
        manifest.append({"group": f"demo_{i}", "demo_id": demo_id})

    log.info("Chunk playback complete. Saving data...")
    env.save_data()

    n_dof = 28
    layout, cursor = {}, 0
    for key, width in [("challenge_61", 61)] + [
        (k, n_dof) for k in ("joint_qpos", "joint_qvel", "joint_qeffort", *DECOMP_KEYS)
    ]:
        layout[key] = [cursor, cursor + width]
        cursor += width
    with _h5py.File(output, "r+") as f:
        f["data"].attrs["task_id"] = task_id
        f["data"].attrs["task_name"] = task_name
        f["data"].attrs["replay_cell"] = "B-natural-frequency"
        f["data"].attrs["proprio_layout"] = _json.dumps(layout)
        f["data"].attrs["manifest"] = _json.dumps(manifest)
    log.info(f"Saved {len(manifest)}-episode torque chunk to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Replay a raw demo with physics-on torque recording")
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--demo_id", type=int, required=False, default=None)
    parser.add_argument("--demo-ids", type=str, default=None,
                        help="Comma-separated demo ids for chained single-process replay (chunk mode)")
    parser.add_argument("--with-cameras", action="store_true", help="Also record RGB (slower)")
    parser.add_argument("--flush_every_n_steps", type=int, default=1000)
    parser.add_argument("--use_longest_demo", action="store_true")
    parser.add_argument(
        "--preserve-frequencies",
        action="store_true",
        help="Replay at the recorded control frequency instead of 1 kHz micro-steps",
    )
    parser.add_argument(
        "--effort-limits",
        action="store_true",
        help="Clamp joint drives to placeholder per-group effort caps before playback",
    )
    parser.add_argument("--suffix", type=str, default="", help="Output filename suffix (calibration cell tag)")
    parser.add_argument("--output", type=str, default=None, help="Explicit output path (overrides suffix layout)")
    parser.add_argument("--lean", action="store_true",
                        help="Skip all camera/render-product creation (proprio-only replay)")
    parser.add_argument("--gpu-dynamics", action="store_true",
                        help="EXPERIMENTAL: run PhysX dynamics on GPU (validate effort equivalence first)")
    parser.add_argument("--physics-threads", type=int, default=0,
                        help="Cap PhysX CPU dispatcher threads (0 = Isaac default, i.e. all cores). "
                        "Set to ~cores/total_workers when running multiple instances per box.")
    args = parser.parse_args()

    if args.gpu_dynamics:
        gm.USE_GPU_DYNAMICS = True
        log.info("EXPERIMENTAL: gm.USE_GPU_DYNAMICS enabled")
    if args.physics_threads > 0:
        # Kit consumes --/path=value overrides from sys.argv at SimulationApp
        # launch (see simulator.py launch); the app has not launched yet here.
        import sys as _sys

        _sys.argv.append(f"--/physics/numThreads={args.physics_threads}")
        log.info(f"PhysX CPU threads capped at {args.physics_threads}")

    if args.demo_ids:
        assert args.output, "chunk mode requires --output"
        replay_torque_chunk(
            data_folder=args.data_folder,
            demo_ids=[int(x) for x in args.demo_ids.split(",")],
            output=args.output,
            use_longest_demo=args.use_longest_demo,
            lean=args.lean,
        )
        og.shutdown()
        return

    assert args.demo_id is not None, "either --demo_id or --demo-ids is required"
    replay_torque(
        data_folder=args.data_folder,
        demo_id=args.demo_id,
        with_cameras=args.with_cameras,
        flush_every_n_steps=args.flush_every_n_steps,
        use_longest_demo=args.use_longest_demo,
        preserve_frequencies=args.preserve_frequencies,
        effort_limits=args.effort_limits,
        suffix=args.suffix,
        output=args.output,
    )
    og.shutdown()


if __name__ == "__main__":
    main()
