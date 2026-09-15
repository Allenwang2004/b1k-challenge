"""Off-policy action agreement: push demo frames through a policy and compare
its prediction to what the human actually did.

Why this rather than a rollout. A rollout conflates two things -- the policy
misreading a frame, and the policy having driven itself somewhere the demos
never go. Feeding the policy *demo* frames removes the second: every query is
made from a state the human reached, so the residual is perception-and-policy
error alone, measured against a known-good action. It also costs nothing in
simulation, which is the difference between an overnight run and 400 GPU-hours.

It answers the "does the model lock onto the right object" question more
directly than comparing visual embeddings would. Embedding distance between two
systems is not on an interpretable scale and a VLM latent does not cleanly
encode "attending to the mug"; the commanded base velocity does. So the headline
metric here is **base-command agreement in the window before the base moves**:
at the frames where the human is about to set off toward an object, does the
policy command motion in the same direction?

Metrics, per frame and aggregated:

  mae / cos          per action group (base, torso, arms, grippers) and overall
  base_cos           cosine between commanded and demonstrated planar base
                     velocity -- the "is it heading the same way" number
  base_sign_agree    fraction of frames where both agree on moving vs holding
  chunk_divergence   with --chunk K, the policy is queried once and its whole
                     predicted chunk compared against the next K demo actions,
                     so short-horizon drift is visible rather than averaged away

Frames are tagged with the phase they fall in, using the grasp index from
``demo_grasp_index.py``: ``pre_motion`` (before the base first moves toward the
object it will grasp), ``approach`` (base driving), ``settle`` (parked, arm
reaching) and ``hold`` (something in the gripper). Agreement is reported per
phase, because a policy that tracks the human while driving but diverges the
moment it has to reach is a different failure from one that never aims right.

## Where the frames come from

``behavior-1k/2026-challenge-demos`` (LeRobot v3.0) carries everything needed:
RGB for all three cameras, the 61-dim challenge state, the 23-dim action, and
``observation.robot2cam_pose.*`` -- the ``cam_rel_poses`` the evaluator sends
the policy. Nothing has to be re-rendered. See ``lerobot_demo_source.py``.

The other two sources do **not** have images: ``2026-challenge-rawdata`` is
state-only, and our torque sidecars were recorded proprio-only. The sidecars are
still useful here as a second opinion -- they carry ``task::low_dim``, hence
ground-truth grasp flags and object poses -- but they are a physics
re-simulation, so their state drifts from the recording (positions ~1e-3,
velocities up to ~1 rad/s). Feed the policy the LeRobot state and the LeRobot
images together; those are the pair that actually co-occurred.

Phases come from the official ``annotations/`` skill segmentation shipped with
the demos ("move to", "pick up from", "open door", ... with frame ranges and a
navigation/uncoordinated/coordinated type). That is ground truth and beats
inferring phases from base motion; the sidecar-derived labelling is kept as a
fallback for files with no annotation.

``--dry-run`` exercises the whole pipeline with the recorded action standing in
for the policy, and must report exact self-agreement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTION_GROUPS = {
    "base": slice(0, 3),
    "torso": slice(3, 7),
    "left_arm": slice(7, 14),
    "left_gripper": slice(14, 15),
    "right_arm": slice(15, 22),
    "right_gripper": slice(22, 23),
}
CAMERAS = {
    "left_wrist": "robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1:right_realsense_link:Camera:0",
    "head": "robot_r1:zed_link:Camera:0",
}
ROBOT = "robot_r1"
PROPRIO_61 = slice(0, 61)  # challenge_61 block at the head of the 229-dim vector
BASE_MOVE_EPS = 1e-3  # commanded base velocity below this counts as "hold"

# action slice -> matching measured-position slice inside challenge_61, for the
# position-controlled groups. Grippers are excluded: one command scalar drives
# two finger joints, so there is no like-for-like displacement to take.
DELTA_PAIRS = {
    "torso": (slice(3, 7), slice(53, 57)),        # trunk_qpos
    "left_arm": (slice(7, 14), slice(3, 10)),     # arm_left_qpos
    "right_arm": (slice(15, 22), slice(28, 35)),  # arm_right_qpos
}


# --------------------------------------------------------------------- inputs


def _find(grp: h5py.Group, *needles: str) -> str | None:
    keys = []
    grp.visit(lambda n: keys.append(n))
    for k in keys:
        if all(n in k for n in needles):
            return k
    return None


def preflight(path: str, demo: str | None = None) -> dict:
    """Report exactly which required channels a file has, and which it lacks."""
    with h5py.File(path, "r") as f:
        demo = demo or next(k for k in f["data"] if k.startswith("demo_"))
        g = f["data"][demo]
        obs = g["obs"] if "obs" in g else None
        report = {
            "path": path, "demo": demo,
            "task_name": str(f["data"].attrs.get("task_name", "?")),
            "frames": int(g.attrs.get("num_samples", 0)),
            "action": "action" in g and list(g["action"].shape),
        }
        report["proprio"] = None
        if obs is not None:
            pk = _find(obs, "proprio")
            if pk:
                report["proprio"] = [pk, list(obs[pk].shape)]
            for cam_id, sensor in CAMERAS.items():
                key = _find(obs, sensor, "rgb")
                report[f"rgb_{cam_id}"] = [key, list(obs[key].shape)] if key else None
            ck = _find(obs, "cam_rel_poses")
            report["cam_rel_poses"] = [ck, list(obs[ck].shape)] if ck else None
    missing = [k for k in
               ["action", "proprio", "rgb_head", "rgb_left_wrist", "rgb_right_wrist", "cam_rel_poses"]
               if not report.get(k)]
    report["missing"] = missing
    report["ready"] = not missing
    return report


class DemoSource:
    """Frames of one demo episode, packed the way the evaluator packs them."""

    def __init__(self, path: str, demo: str | None = None, require_images: bool = True):
        self.f = h5py.File(path, "r")
        self.demo = demo or next(k for k in self.f["data"] if k.startswith("demo_"))
        g = self.f["data"][self.demo]
        self.task_name = str(self.f["data"].attrs.get("task_name", "?"))
        self.action = np.asarray(g["action"])
        obs = g["obs"]
        pk = _find(obs, "proprio")
        if pk is None:
            raise ValueError("no proprio channel")
        self.proprio = np.asarray(obs[pk])
        if self.proprio.shape[1] >= 229:
            self.proprio = self.proprio[:, PROPRIO_61]
        elif self.proprio.shape[1] != 61:
            raise ValueError(f"unexpected proprio width {self.proprio.shape[1]}")
        self.rgb, self.cam_rel = {}, None
        if require_images:
            for cam_id, sensor in CAMERAS.items():
                key = _find(obs, sensor, "rgb")
                if key is None:
                    raise ValueError(
                        f"no RGB for {cam_id}. Demos must be re-rendered with cameras "
                        "(replay_torque.py --with-cameras) -- see this module's docstring."
                    )
                self.rgb[sensor] = obs[key]
            ck = _find(obs, "cam_rel_poses")
            if ck is None:
                raise ValueError(
                    "no cam_rel_poses. The evaluator sends it to the policy every step; "
                    "recording it during the RGB re-render is required, not optional."
                )
            self.cam_rel = obs[ck]
        self.T = min(len(self.action), len(self.proprio))

    def obs_at(self, t: int) -> dict:
        import torch as th
        out = {f"{ROBOT}::proprio": th.tensor(self.proprio[t], dtype=th.float32)}
        for sensor, ds in self.rgb.items():
            out[f"{ROBOT}::{sensor}::rgb"] = th.tensor(np.asarray(ds[t]))
        if self.cam_rel is not None:
            out[f"{ROBOT}::cam_rel_poses"] = th.tensor(np.asarray(self.cam_rel[t]), dtype=th.float32)
        return out

    def close(self):
        self.f.close()


# --------------------------------------------------------------------- phases


def phase_labels(path: str, demo: str, T: int) -> np.ndarray:
    """Per-frame phase tag using the same grasp/approach logic as the index."""
    from demo_grasp_index import _hold_intervals, _moving_mask, _segments
    from low_dim_layout import ARMS, grasp_flags, object_positions

    labels = np.array(["other"] * T, dtype=object)
    with h5py.File(path, "r") as f:
        try:
            flags, objs = grasp_flags(f, demo)
            pos, _, _ = object_positions(f, demo)
        except Exception:  # noqa: BLE001 - phases are a nicety, never a hard failure
            return labels
    n = min(T, flags.shape[0], pos.shape[0])
    mv = _moving_mask(pos[:n, 0, :2])
    holds_any = np.zeros(n, dtype=bool)
    onsets = []
    for oi in range(flags.shape[1]):
        for ai in range(len(ARMS)):
            for s, e in _hold_intervals(flags[:n, oi, ai])[0]:
                holds_any[s:e] = True
                onsets.append(s)

    # Precedence, applied in this order so the later one wins:
    #   hold < settle < approach < pre_motion
    # A frame can be several of these at once -- carrying an object while
    # driving is both hold and approach. Locomotion wins because every question
    # this harness is built for is about where the base is going.
    labels[:n][holds_any] = "hold"
    for s, e in _segments(mv):
        nxt = min([o for o in onsets if o >= e], default=None)
        if nxt is not None:
            labels[e:nxt] = "settle"
    labels[:n][mv] = "approach"
    # Everything before the base first moves is the "did it aim right" window.
    first_move = int(np.argmax(mv)) if mv.any() else n
    labels[:first_move] = "pre_motion"
    return labels


# -------------------------------------------------------------------- metrics


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def compare(pred: np.ndarray, true: np.ndarray, state: np.ndarray | None = None) -> dict:
    """Agreement between a predicted and a demonstrated action.

    The two halves of the action vector need different treatment, because
    r1pro.yaml gives them different controllers:

      base (0:3)   ``HolonomicBaseJointController``, motor_type **velocity**.
                   The command IS a direction of travel, so a cosine on the
                   planar components is meaningful as it stands.
      rest (3:23)  ``JointController`` with ``use_delta_commands: false`` --
                   **absolute** joint position targets. Comparing those
                   directly is close to meaningless: both policy and human
                   command something near the current pose, so the raw cosine
                   sits at ~0.998 no matter how differently they intend to
                   move. What carries the intent is the *displacement* from the
                   measured joint position, which is what ``delta_*`` reports.
    """
    out = {"mae": float(np.abs(pred - true).mean()), "cos": _cos(pred, true)}
    for name, sl in ACTION_GROUPS.items():
        out[f"mae_{name}"] = float(np.abs(pred[sl] - true[sl]).mean())

    # Base: a velocity command. Only score direction when the human is actually
    # driving -- the angle of a near-zero vector is noise, not disagreement.
    true_speed, pred_speed = float(np.linalg.norm(true[0:3])), float(np.linalg.norm(pred[0:3]))
    out["base_speed_true"] = true_speed
    out["base_speed_pred"] = pred_speed
    moving = true_speed > BASE_MOVE_EPS
    out["base_cos"] = _cos(pred[0:2], true[0:2]) if moving else float("nan")
    out["base_sign_agree"] = float((pred_speed > BASE_MOVE_EPS) == moving)
    out["base_speed_ratio"] = pred_speed / true_speed if moving else float("nan")

    # Arms/torso: compare commanded displacement from the current pose.
    if state is not None:
        dp, dt = [], []
        for a_sl, s_sl in DELTA_PAIRS.values():
            dp.append(pred[a_sl] - state[s_sl])
            dt.append(true[a_sl] - state[s_sl])
        dp, dt = np.concatenate(dp), np.concatenate(dt)
        out["delta_cos"] = _cos(dp, dt)
        out["delta_mae"] = float(np.abs(dp - dt).mean())
        out["delta_mag_ratio"] = (float(np.linalg.norm(dp) / np.linalg.norm(dt))
                                  if np.linalg.norm(dt) > 1e-6 else float("nan"))
        for name, (a_sl, s_sl) in DELTA_PAIRS.items():
            out[f"delta_cos_{name}"] = _cos(pred[a_sl] - state[s_sl], true[a_sl] - state[s_sl])
    return out


def run(src: DemoSource, policy, stride: int, chunk: int, labels: np.ndarray, limit: int | None) -> list[dict]:
    rows = []
    ts = range(0, src.T - chunk, stride)
    if limit:
        ts = list(ts)[:limit]
    for t in ts:
        if policy is None:  # dry run: the demo predicts itself, agreement must be perfect
            pred = src.action[t:t + max(chunk, 1)].copy()
        else:
            a = policy.forward(obs=src.obs_at(t))
            pred = np.atleast_2d(np.asarray(a, dtype=np.float32))
        k = min(len(pred), chunk if chunk > 1 else 1, src.T - t)
        state = getattr(src, "state", None)
        state = state[t] if state is not None else src.proprio[t]
        row = compare(pred[0], src.action[t], state)
        row.update(t=int(t), phase=str(labels[t]) if t < len(labels) else "other")
        if chunk > 1 and k > 1:
            errs = [float(np.abs(pred[j] - src.action[t + j]).mean()) for j in range(k)]
            row["chunk_mae_first"] = errs[0]
            row["chunk_mae_last"] = errs[-1]
            row["chunk_divergence"] = errs[-1] - errs[0]
        rows.append(row)
    return rows


def _stat(rs: list[dict], k: str) -> tuple[float, int]:
    """Median, not mean. ``delta_mag_ratio`` and ``base_speed_ratio`` are ratios
    whose denominator can be near zero, so a handful of frames where the human
    barely moved dominate any mean -- on task 50 the mean magnitude ratio reads
    17.0 against a median of 7.4, and mean base_cos reads 0.59 against a median
    of 0.94. The median is the honest summary here."""
    v = np.array([r.get(k, np.nan) for r in rs], dtype=float)
    v = v[np.isfinite(v)]
    return (float(np.median(v)), int(v.size)) if v.size else (float("nan"), 0)


def summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    out = {"n": len(rows), "overall": {}, "overall_mean": {}}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        out["overall"][k] = float(np.median(v)) if v.size else float("nan")
        out["overall_mean"][k] = float(np.mean(v)) if v.size else float("nan")
    by = defaultdict(list)
    for r in rows:
        by[r["phase"]].append(r)
    out["by_phase"] = {}
    for ph, rs in by.items():
        d = {"n": len(rs)}
        for k in PHASE_METRICS:
            d[k], d[f"{k}_n"] = _stat(rs, k)
        out["by_phase"][ph] = d
    return out


PHASE_METRICS = ("delta_cos", "delta_mag_ratio", "base_cos", "base_sign_agree",
                 "base_speed_ratio", "mae")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None, help="sidecar hdf5 (omit when using --task)")
    ap.add_argument("--task", type=int, default=None, help="task index -> pull frames from the LeRobot demos repo")
    ap.add_argument("--demo-index", type=int, default=0, help="demo within the task (with --task)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap episode length (with --task)")
    ap.add_argument("--demo", default=None)
    ap.add_argument("--preflight", action="store_true", help="report which required channels exist")
    ap.add_argument("--dry-run", action="store_true", help="no policy; the demo predicts itself")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--stride", type=int, default=30, help="query every N frames")
    ap.add_argument("--chunk", type=int, default=1, help="compare a K-step predicted chunk")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of queries")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.task is not None:
        from lerobot_demo_source import LeRobotDemoSource

        policy = None
        if not args.dry_run:
            if args.host is None:
                print("need --host (policy server) or --dry-run", file=sys.stderr)
                return 2
            from omnigibson.eval.policies import WebsocketPolicy
            policy = WebsocketPolicy(host=args.host, port=args.port)

        # Resolve the query schedule first so only those frames get decoded --
        # decoding a full episode is ~3 MB/frame across three cameras (30 GB for
        # a 10k-frame episode), and we look at a few dozen.
        meta = LeRobotDemoSource(args.task, args.demo_index, with_images=False,
                                 max_frames=args.max_frames)
        wanted = list(range(0, meta.T - args.chunk, args.stride))
        if args.limit:
            wanted = wanted[:args.limit]
        src = LeRobotDemoSource(args.task, args.demo_index, with_images=not args.dry_run,
                                max_frames=args.max_frames, frame_indices=wanted)
        labels = src.skill_labels()
        labels = np.array([lab or "unannotated" for lab in labels], dtype=object)
        print(f"{src.task_name} episode {src.episode_index} (raw {src.raw_episode_id}) {src.T} frames")
        uniq, cnt = np.unique(labels, return_counts=True)
        print("  skills: " + ", ".join(f"{u}={c}" for u, c in
                                       sorted(zip(uniq, cnt), key=lambda kv: -kv[1])[:8]))
        rows = run(src, policy, args.stride, args.chunk, labels, args.limit)
        src.close()
        return _emit(rows, args)

    if args.path is None:
        print("give a sidecar path or --task", file=sys.stderr)
        return 2

    if args.preflight:
        rep = preflight(args.path, args.demo)
        print(json.dumps(rep, indent=1))
        if not rep["ready"]:
            print(f"\nNOT READY -- missing: {', '.join(rep['missing'])}")
            print("Re-render this episode with cameras before running the harness; see the module docstring.")
        return 0 if rep["ready"] else 2

    policy = None
    if not args.dry_run:
        if args.host is None:
            print("need --host (policy server) or --dry-run", file=sys.stderr)
            return 2
        from omnigibson.eval.policies import WebsocketPolicy
        policy = WebsocketPolicy(host=args.host, port=args.port)

    src = DemoSource(args.path, args.demo, require_images=not args.dry_run)
    labels = phase_labels(args.path, src.demo, src.T)
    print(f"{src.task_name} [{src.demo}] {src.T} frames; "
          f"phases: {dict(zip(*np.unique(labels, return_counts=True)))}")
    rows = run(src, policy, args.stride, args.chunk, labels, args.limit)
    src.close()
    return _emit(rows, args)


def _emit(rows: list[dict], args) -> int:
    summary = summarise(rows)
    if args.dry_run:
        ok = summary["overall"]["mae"] < 1e-9 and abs(summary["overall"]["cos"] - 1.0) < 1e-6
        print(f"dry run self-agreement: mae={summary['overall']['mae']:.3e} "
              f"cos={summary['overall']['cos']:.6f} -> {'OK' if ok else 'BROKEN'}")
        if not ok:
            return 1

    print(f"\n{summary['n']} queries (medians)")
    print("overall: " + "  ".join(f"{k}={summary['overall'].get(k, float('nan')):.4f}"
                                  for k in ("delta_cos", "delta_mag_ratio", "base_cos",
                                            "base_speed_ratio", "base_sign_agree")))
    print("  (raw `cos` over the 23-dim action is ~1 by construction -- most of it is absolute\n"
          "   joint targets near the current pose. delta_cos is the one that carries intent.)")
    print("\nby phase (n_dir = queries where the human was actually driving):")
    print(f"  {'phase':<16} {'n':>5} {'delta_cos':>10} {'d_mag':>7} {'base_cos':>9} "
          f"{'n_dir':>6} {'base_agree':>11}")
    for ph, d in sorted(summary["by_phase"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {ph:<16} {d['n']:>5} {d['delta_cos']:>10.4f} {d['delta_mag_ratio']:>7.2f} "
              f"{d['base_cos']:>9.4f} {d['base_cos_n']:>6} {d['base_sign_agree']:>11.4f}")

    if args.out:
        with open(args.out, "w") as fp:
            json.dump({"summary": summary, "rows": rows}, fp, indent=1)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
