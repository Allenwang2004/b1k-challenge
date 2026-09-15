"""Read demo frames -- RGB included -- from ``behavior-1k/2026-challenge-demos``.

This is the repo that actually carries images. The other two do not, and the
distinction costs a GPU-week if you get it wrong:

  ``2026-challenge-rawdata``   state (T, 1100) + action. No images.
  torque sidecars (ours)       proprio 229 + task::low_dim. No images.
  ``2026-challenge-demos``     LeRobot v3.0: RGB **and** depth for all three
                               cameras, the 61-dim challenge state, the 23-dim
                               action, and ``observation.robot2cam_pose.*``
                               -- which is the ``cam_rel_poses`` the evaluator
                               feeds the policy. Nothing has to be re-rendered.

Layout (20,000 episodes = 100 tasks x 200 demos, 210.9M frames):

  ``meta/episodes/chunk-{task_index:03d}/file-000.parquet`` -- 200 rows, one
  per demo of that task, carrying ``raw_episode_id`` (the same id our sidecars
  use, e.g. 500010 for task 50 demo 0010), ``length``, the parquet
  chunk/file, and per video key the file plus the ``[from_timestamp,
  to_timestamp]`` window of this episode inside a shared ~200 MB mp4.

So ``chunk == task_index`` and ``episode_index == task_index*200 +
demo_index_within_task``. Fetching one episode pulls its data parquet (~82 MB)
and one mp4 per camera (~200 MB each); those files are shared with neighbouring
demos of the same task, so the marginal cost drops fast when you take several.

Validated against our torque sidecar for the same episode: the 61-dim
``observation.state`` here and ``robot::proprio[:, :61]`` there agree, which
also confirms the two id conventions line up.

CLI::

    python lerobot_demo_source.py --task 50 --demo-index 0 --probe
    python lerobot_demo_source.py --task 50 --demo-index 0 \\
        --verify-against /path/to/chunk_00500010_1.hdf5
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

import numpy as np

REPO = "behavior-1k/2026-challenge-demos"
FPS = 30.0
EPISODES_PER_TASK = 200

# The evaluator concatenates cam_rel_poses in robot_camera_names order, which
# r1pro.yaml declares as left_wrist, right_wrist, head. Same order here, or the
# policy receives a silently permuted 21-vector.
CAMERA_ORDER = [
    ("left_wrist", "left_realsense_link_camera_0"),
    ("right_wrist", "right_realsense_link_camera_0"),
    ("head", "zed_link_camera_0"),
]
# Flattened obs names the policy expects (see eval/r1pro.yaml).
SENSOR_NAMES = {
    "left_wrist": "robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1:right_realsense_link:Camera:0",
    "head": "robot_r1:zed_link:Camera:0",
}
ROBOT = "robot_r1"


def _hf(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, path, repo_type="dataset")


@functools.lru_cache(maxsize=128)
def _episode_meta(task_index: int) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(_hf(f"meta/episodes/chunk-{task_index:03d}/file-000.parquet")).to_pylist()


def resolve(task_index: int, demo_index: int | None = None, raw_episode_id: int | None = None) -> dict:
    """The metadata row for one demo, by position within the task or by raw id."""
    rows = _episode_meta(task_index)
    if raw_episode_id is not None:
        hit = [r for r in rows if int(r["raw_episode_id"]) == int(raw_episode_id)]
        if not hit:
            raise KeyError(f"raw_episode_id {raw_episode_id} not in task {task_index} "
                           f"(range {rows[0]['raw_episode_id']}..{rows[-1]['raw_episode_id']})")
        return hit[0]
    if demo_index is None:
        raise ValueError("give demo_index or raw_episode_id")
    hit = [r for r in rows if int(r["demo_index_within_task"]) == int(demo_index)]
    if not hit:
        raise KeyError(f"demo_index {demo_index} not in task {task_index}")
    return hit[0]


def sidecar_name_to_ids(name: str) -> tuple[int, int]:
    """'chunk_00500010_1.hdf5' -> (task_index=50, raw_episode_id=500010)."""
    stem = os.path.basename(name).split("_")
    raw = int(stem[1])
    return raw // 10000, raw


class LeRobotDemoSource:
    """One episode, packed the way the evaluator packs observations."""

    def __init__(self, task_index: int, demo_index: int | None = None,
                 raw_episode_id: int | None = None, with_images: bool = True,
                 max_frames: int | None = None, frame_indices=None):
        self.row = resolve(task_index, demo_index, raw_episode_id)
        self._skills: list[dict] | None = None
        self.task_index = task_index
        self.task_name = self.row["tasks"][0] if isinstance(self.row["tasks"], list) else str(self.row["tasks"])
        self.raw_episode_id = int(self.row["raw_episode_id"])
        self.episode_index = int(self.row["episode_index"])
        self.T = int(self.row["length"])
        if max_frames:
            self.T = min(self.T, max_frames)

        import pyarrow.parquet as pq

        table = pq.read_table(_hf(f"data/chunk-{int(self.row['data/chunk_index']):03d}"
                                  f"/file-{int(self.row['data/file_index']):03d}.parquet"))
        mask = np.asarray(table["episode_index"]) == self.episode_index
        if not mask.any():
            raise ValueError(f"episode {self.episode_index} not present in its data parquet")
        idx = np.flatnonzero(mask)
        sub = table.take(idx[: self.T])

        def col(name: str) -> np.ndarray:
            return np.stack([np.asarray(v, dtype=np.float32) for v in sub[name].to_pylist()])

        self.state = col("observation.state")
        self.action = col("action")
        self.reward = np.asarray(sub["next.reward"].to_pylist(), dtype=np.float32).reshape(-1)
        self.cam_rel_poses = np.concatenate(
            [col(f"observation.robot2cam_pose.{key}") for _, key in CAMERA_ORDER], axis=1
        )
        self.T = min(self.T, len(self.state))

        # Decoding a whole episode costs ~3 MB/frame across the three cameras --
        # 30 GB for a 10k-frame episode. Callers that sample sparsely should say
        # which frames they want; only those are kept.
        self.wanted = (np.array(sorted(set(int(i) for i in frame_indices if i < self.T)))
                       if frame_indices is not None else None)
        self._pos = ({int(t): i for i, t in enumerate(self.wanted)}
                     if self.wanted is not None else None)
        self.frames: dict[str, np.ndarray] = {}
        if with_images:
            for cam_id, key in CAMERA_ORDER:
                self.frames[cam_id] = self._decode(key)

    def _frame(self, cam_id: str, t: int) -> np.ndarray:
        arr = self.frames[cam_id]
        return arr[self._pos[t]] if self._pos is not None else arr[t]

    def _decode(self, video_key: str) -> np.ndarray:
        """Frames of this episode from the shared mp4, using its timestamp window.

        Uses OpenCV rather than PyAV: ``import av`` is broken in the `behavior`
        env (openvino drags in a libstdc++ that lacks CXXABI_1.3.15). Seeking is
        verified rather than trusted -- a keyframe-only seek would silently
        return the wrong episode out of a file that packs several.
        """
        import cv2

        pre = f"videos/observation.rgb.{video_key}"
        path = _hf(f"{pre}/chunk-{int(self.row[pre + '/chunk_index']):03d}"
                   f"/file-{int(self.row[pre + '/file_index']):03d}.mp4")
        start = int(round(float(self.row[pre + "/from_timestamp"]) * FPS))

        cap = cv2.VideoCapture(path)
        try:
            if start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start:
                    # Seek landed elsewhere; walk there instead of trusting it.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    for _ in range(start):
                        if not cap.grab():
                            raise ValueError(f"{video_key}: ran out of frames before {start}")
            out = []
            if self.wanted is None:
                while len(out) < self.T:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                need = self.T
            else:
                # One sequential pass; skipped frames are grabbed but not decoded
                # to RGB, which is far cheaper than a seek per wanted frame.
                nxt, cursor = set(self.wanted.tolist()), 0
                last = int(self.wanted[-1])
                while cursor <= last:
                    if cursor in nxt:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    elif not cap.grab():
                        break
                    cursor += 1
                need = len(self.wanted)
        finally:
            cap.release()
        if len(out) < need:
            raise ValueError(f"{video_key}: decoded {len(out)} frames from offset {start}, expected {need}")
        return np.stack(out)

    @property
    def skills(self) -> list[dict]:
        """Official per-episode skill segmentation from ``annotations/``.

        Each entry has ``skill_description`` ("move to", "pick up from",
        "open door", ...), the objects involved, ``skill_type``
        (navigation / uncoordinated / coordinated) and ``frame_duration``
        ``[start, end]``. This is ground truth for the phase a frame belongs to
        -- strictly better than inferring phases from base motion and grasp
        flags, and it exists for all 20,000 episodes.
        """
        if self._skills is None:
            import json

            with open(_hf(str(self.row["annotation_path"]))) as fp:
                doc = json.load(fp)
            # The file also carries `primitive_annotation`, which is empty for
            # every episode checked -- skills are the populated level.
            self._skills = doc.get("skill_annotation", [])
        return self._skills

    def skill_labels(self, field: str = "skill_description") -> np.ndarray:
        """Per-frame label, '' outside any annotated segment.

        ``field`` is ``skill_description`` ("move to", "open door", ...) or
        ``skill_type`` (navigation / uncoordinated / coordinated).
        """
        labels = np.array([""] * self.T, dtype=object)
        for e in self.skills:
            fd = e.get("frame_duration")
            if not fd or len(fd) < 2:
                continue
            val = e.get(field) or [""]
            s, t = int(fd[0]), min(int(fd[1]), self.T)
            if s < self.T:
                labels[s:t] = val[0] if isinstance(val, list) else str(val)
        return labels

    def obs_at(self, t: int) -> dict:
        import torch as th

        obs = {f"{ROBOT}::proprio": th.tensor(self.state[t], dtype=th.float32),
               f"{ROBOT}::cam_rel_poses": th.tensor(self.cam_rel_poses[t], dtype=th.float32)}
        for cam_id, _ in CAMERA_ORDER:
            if cam_id in self.frames:
                obs[f"{ROBOT}::{SENSOR_NAMES[cam_id]}::rgb"] = th.tensor(self._frame(cam_id, t))
        return obs

    def close(self):
        self.frames.clear()


# ------------------------------------------------------------------ CLI checks


def _verify(src: LeRobotDemoSource, sidecar: str) -> int:
    """Cross-check against our torque sidecar for the same episode.

    The **action** streams must be bit-identical: the sidecar is a replay driven
    by the recorded actions, so anything else means the ids do not line up.
    That is the real identity test.

    The **state** must not be expected to match exactly. The sidecar is a
    physics-on re-simulation, so joint positions track the original to ~1e-3
    while velocities, being the noisier quantity, drift up to ~1 rad/s. That is
    re-simulation, not a mismatch -- but it does mean the policy has to be fed
    the LeRobot state together with the LeRobot images, since those are the pair
    that actually co-occurred. Phase labels derived from the sidecar transfer
    fine (positions agree, and the episodes are frame-aligned and equal length).
    """
    import h5py

    with h5py.File(sidecar, "r") as f:
        demo = next(k for k in f["data"] if k.startswith("demo_"))
        obs = f["data"][demo]["obs"]
        pkey = next(k for k in obs if "proprio" in k)
        pr = np.asarray(obs[pkey])[:, :61]
        act = np.asarray(f["data"][demo]["action"])
        name = str(f["data"].attrs.get("task_name"))
    n = min(len(pr), src.T)
    da = np.abs(act[:n] - src.action[:n])
    print(f"\nsidecar {os.path.basename(sidecar)} (task_name={name}) vs LeRobot episode "
          f"{src.episode_index} ({src.task_name}), {n} frames compared")
    print(f"  action  max|diff|={da.max():.6f}  mean|diff|={da.mean():.6e}  "
          f"-> {'IDENTICAL (ids line up)' if da.max() < 1e-6 else 'DIFFERENT -- ids do NOT line up'}")
    qpos_cols = np.r_[3:10, 17:24, 28:35, 42:49, 53:57]
    qvel_cols = np.r_[0:3, 10:17, 26:28, 35:42, 51:53, 57:61]
    dq = np.abs(pr[:n][:, qpos_cols] - src.state[:n][:, qpos_cols])
    dv = np.abs(pr[:n][:, qvel_cols] - src.state[:n][:, qvel_cols])
    print(f"  state pos  max={dq.max():.5f} mean={dq.mean():.3e}")
    print(f"  state vel  max={dv.max():.5f} mean={dv.mean():.3e}   "
          "(velocity drift is expected: the sidecar is a re-simulation)")
    ok = da.max() < 1e-6 and dq.mean() < 1e-2
    print(f"  -> {'CONSISTENT' if ok else 'UNEXPECTED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--demo-index", type=int, default=0)
    ap.add_argument("--raw-episode-id", type=int, default=None)
    ap.add_argument("--probe", action="store_true", help="metadata only, no downloads of data/video")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--verify-against", default=None, help="torque sidecar hdf5 for the same episode")
    args = ap.parse_args()

    if args.probe:
        row = resolve(args.task, args.demo_index, args.raw_episode_id)
        for k in ("episode_index", "tasks", "length", "task_index", "demo_index_within_task",
                  "raw_episode_id", "data/chunk_index", "data/file_index", "annotation_path"):
            print(f"  {k}: {row[k]}")
        for cam_id, key in CAMERA_ORDER:
            pre = f"videos/observation.rgb.{key}"
            print(f"  {cam_id}: chunk={row[pre + '/chunk_index']} file={row[pre + '/file_index']} "
                  f"t=[{row[pre + '/from_timestamp']:.2f}, {row[pre + '/to_timestamp']:.2f}]")
        return 0

    src = LeRobotDemoSource(args.task, args.demo_index, args.raw_episode_id,
                            with_images=not args.no_images, max_frames=args.max_frames)
    print(f"task {src.task_index} ({src.task_name}) episode {src.episode_index} "
          f"raw {src.raw_episode_id}: {src.T} frames")
    print(f"  state {src.state.shape}  action {src.action.shape}  cam_rel_poses {src.cam_rel_poses.shape}")
    for cam_id, arr in src.frames.items():
        print(f"  rgb {cam_id}: {arr.shape} {arr.dtype} range [{arr.min()}, {arr.max()}]")
    if src.frames:
        obs = src.obs_at(0)
        print("  obs keys:", *[f"\n    {k} {tuple(v.shape)}" for k, v in obs.items()])
    try:
        lab = src.skill_labels()
        uniq, cnt = np.unique(lab, return_counts=True)
        order = np.argsort(-cnt)
        print(f"  skills ({len(src.skills)} segments): "
              + ", ".join(f"{uniq[i] or '<none>'}={cnt[i]}" for i in order[:8]))
    except Exception as exc:  # noqa: BLE001
        print(f"  skills: unavailable ({str(exc)[:100]})")
    if args.verify_against:
        return _verify(src, args.verify_against)
    return 0


if __name__ == "__main__":
    sys.exit(main())
