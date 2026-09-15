"""Grasp-event index and base-approach geometry for the human demo sidecars.

This is the demo-side reference distribution that policy rollouts will later be
compared against. It answers, per grasp, two questions the scalar leaderboard
metrics cannot: *when* does the robot take hold of something, and *how did the
base get into position to do it*.

Both come from ground truth rather than heuristics. Grasps are
``agent.is_grasping(arm, obj)`` as recorded in ``task::low_dim`` (see
``low_dim_layout``); a gripper-width threshold is not usable here -- on a
single episode it fires ~112 times on one arm where the truth is 1-3 grasps,
and gripper effort barely separates closed (13.97 N.m) from open (12.41 N.m).

Per grasp onset the script isolates the **approach segment**: the last
contiguous stretch of base motion that ends before the grasp. For that segment
it records

  heading_err_start  angle between base heading at the moment it starts moving
                     and the bearing to the object it will end up grasping.
                     This is "did it aim at the right thing before setting off".
  bearing_err_end    the same angle once the base has parked.
  park_dist          planar base-to-object distance at parking. The dexterity
                     hypothesis predicts rollouts park further out than humans.
  straightness       |displacement| / path length, in [0, 1]. A direct approach
                     is ~1; hunting for a pose drives it down.
  settle_frames      gap between the base stopping and the grasp -- how long the
                     arm needs once parked.

Every position here -- the base's included -- comes from the low_dim channel,
and only while the object's ``real`` bit is set. The base pose is read from the
*agent's own* low_dim entry rather than the ``joint_qpos`` base columns: those
are virtual-joint values in the articulation frame, not world coordinates. On
``freeze_fruit`` the two disagree by ~6 m, so mixing them silently destroys
every bearing and distance in this file.

Analysis venv only (h5py + numpy). Typical run::

    ~/.venv-b1k-analysis/bin/python demo_grasp_index.py \
        --archive /mnt/train-data-1-hdd/b1k-challenge/b1k-torque-sidecars-archive \
        --max-task 49 --episodes-per-task 20 --out-dir grasp_out
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from low_dim_layout import ARMS, grasp_flags, object_positions, recover_layout  # noqa: E402

FPS = 30.0

# Base counts as moving above this planar speed. 2 cm/s is well under the
# slowest purposeful drive and well over the standing jitter of a settled base.
MOVE_SPEED_MPS = 0.02
# Motion shorter than this is jitter, not an approach.
MIN_SEGMENT_FRAMES = 10
# A grasp may follow the base stopping by a while (the arm still has to reach),
# but a segment separated by more than this is a different manoeuvre.
MAX_SETTLE_FRAMES = int(20 * FPS)
# Bridge micro-stops inside one approach (re-planning, brief pauses).
BRIDGE_FRAMES = 15

# The in_gripper assertion is duty-cycled on some tasks: it reads true on
# exactly one frame in ten (1 on, 9 off, for the whole hold), which is the
# grasp check tracking the control cycle rather than the object being released
# and re-taken 780 times. Raw runs are therefore useless -- a single hold on
# spraying_fruit_trees splits into 782 one-frame "grasps". Closing gaps of half
# a second recovers the true holds: >=87% of observed gaps are <=15 frames,
# while genuine releases show up as gaps of hundreds of frames.
GRASP_BRIDGE_FRAMES = 15
# ...and then drop anything too short to be a hold at all.
MIN_GRASP_FRAMES = 5


def _wrap(a: np.ndarray | float):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _base_yaw(h5: h5py.File, demo: str, agent_inst: str) -> np.ndarray:
    """World yaw of the base, from the agent's low_dim orientation.

    ``_get_obs`` stores cos and sin of the world rpy, so yaw comes back exactly
    (and unwrapped-safe) as ``atan2(sin_z, cos_z)``.
    """
    layout = recover_layout(h5, demo)
    ld = h5["data"][demo]["obs/task::low_dim"]
    cs, _ = layout[f"{agent_inst}_ori_cos"]
    ss, _ = layout[f"{agent_inst}_ori_sin"]
    return np.arctan2(np.asarray(ld[:, ss + 2]), np.asarray(ld[:, cs + 2]))


def _moving_mask(bxy: np.ndarray) -> np.ndarray:
    """Per-frame boolean: is the base translating? Micro-stops are bridged."""
    step = np.linalg.norm(np.diff(bxy, axis=0), axis=1) * FPS
    mv = np.concatenate([[False], step > MOVE_SPEED_MPS])
    # Bridge short gaps so one approach with a pause stays one segment.
    idx = np.flatnonzero(mv)
    if idx.size:
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= BRIDGE_FRAMES:
                mv[a:b] = True
    return mv


def _runs(v: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], v.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _hold_intervals(v: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Debounced hold intervals plus the raw duty cycle (held frames / span).

    The duty cycle is reported so the artifact stays visible: a value near 0.1
    means the assertion was duty-cycled and the raw runs were meaningless.

    A channel is treated as duty-cycled when its typical ON run is 1-2 frames,
    and only then is the bridge widened to three median gaps. Widening it
    unconditionally would fuse genuinely separate holds on the clean channels,
    where the median gap *is* the real release.
    """
    raw = _runs(v)
    if not raw:
        return [], float("nan")
    on = np.array([e - s for s, e in raw])
    gaps = np.array([s2 - e1 for (_, e1), (s2, _) in zip(raw[:-1], raw[1:])])
    bridge = GRASP_BRIDGE_FRAMES
    if gaps.size and np.median(on) <= 2:
        bridge = max(bridge, int(3 * np.median(gaps)))
    merged = []
    for s, e in raw:
        if merged and s - merged[-1][1] <= bridge:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    held = sum(e - s for s, e in raw)
    span = sum(e - s for s, e in merged)
    return [(s, e) for s, e in merged if e - s >= MIN_GRASP_FRAMES], float(held / span) if span else float("nan")


def _segments(mv: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs of True, longer than MIN_SEGMENT_FRAMES."""
    d = np.diff(np.concatenate([[0], mv.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= MIN_SEGMENT_FRAMES]


def _episode(path: str, demo: str) -> tuple[list[dict], list[dict], dict] | None:
    with h5py.File(path, "r") as f:
        grp = f["data"][demo]
        task_id = int(f["data"].attrs.get("task_id", -1))
        task_name = f["data"].attrs.get("task_name", "?")
        task_name = task_name.decode() if isinstance(task_name, bytes) else task_name
        flags, objs = grasp_flags(f, demo)
        pos, real, insts = object_positions(f, demo)
        yaw = _base_yaw(f, demo, insts[0])
        n_steps = int(grp.attrs.get("num_samples", len(yaw)))

    T = min(len(yaw), flags.shape[0], n_steps)
    flags, pos, real, yaw = flags[:T], pos[:T], real[:T], yaw[:T]
    bxy = pos[:, 0, :2]  # slot 0 is the agent; same frame as every object here
    mv = _moving_mask(bxy)
    segs = _segments(mv)
    slot_of = {inst: i for i, inst in enumerate(insts)}

    events, approaches = [], []
    held_arm = np.zeros((T, len(ARMS)), dtype=bool)  # debounced, for the summary
    for oi, inst in enumerate(objs):
        for ai, arm in enumerate(ARMS):
            v = flags[:, oi, ai]
            if not v.any():
                continue
            holds, duty = _hold_intervals(v)
            for t0, t1 in holds:
                held_arm[t0:t1, ai] = True
                events.append(
                    dict(task_id=task_id, task_name=task_name, demo=demo, object=inst, arm=arm,
                         onset=int(t0), release=int(t1), frames=int(t1 - t0),
                         seconds=float((t1 - t0) / FPS), onset_frac=float(t0 / T), duty_cycle=duty)
                )
                ap = _approach(int(t0), inst, slot_of, segs, bxy, yaw, pos, real)
                if ap:
                    ap.update(task_id=task_id, task_name=task_name, demo=demo, object=inst, arm=arm,
                              grasp_onset=int(t0), episode_frames=int(T))
                    approaches.append(ap)

    summary = dict(
        task_id=task_id, task_name=task_name, demo=demo, path=os.path.basename(path), frames=int(T),
        n_grasps=len(events), n_objects_grasped=len({e["object"] for e in events}),
        n_approaches=len(approaches), n_base_segments=len(segs),
        moving_frac=float(mv.mean()),
        first_grasp_frac=float(min((e["onset"] for e in events), default=np.nan) / T) if events else float("nan"),
        bimanual_frames=int(np.sum(held_arm[:, 0] & held_arm[:, 1])),
        holding_frac=float(held_arm.any(axis=1).mean()),
        min_duty_cycle=float(min((e["duty_cycle"] for e in events), default=float("nan"))),
    )
    return events, approaches, summary


def _approach(grasp_t: int, inst: str, slot_of, segs, bxy, yaw, pos, real) -> dict | None:
    """The last base-motion segment ending before ``grasp_t``, geometrically described."""
    prior = [(s, e) for s, e in segs if e <= grasp_t]
    if not prior:
        return None
    s, e = prior[-1]
    settle = grasp_t - e
    if settle > MAX_SETTLE_FRAMES:
        return None
    slot = slot_of.get(inst)
    if slot is None:
        return None
    end = min(e, len(pos) - 1)
    # The object must actually exist at both ends of the approach for the
    # bearing to mean anything (sliced products pop into being mid-episode).
    if not (real[s, slot] and real[end, slot]):
        return None

    def bearing_err(t: int) -> float:
        d = pos[t, slot, :2] - bxy[t]
        if np.linalg.norm(d) < 1e-6:
            return float("nan")
        return float(abs(_wrap(np.arctan2(d[1], d[0]) - yaw[t])))

    path = float(np.linalg.norm(np.diff(bxy[s:e + 1], axis=0), axis=1).sum())
    disp = float(np.linalg.norm(bxy[end] - bxy[s]))
    return dict(
        seg_start=int(s), seg_end=int(e), seg_frames=int(e - s), settle_frames=int(settle),
        heading_err_start=bearing_err(s), bearing_err_end=bearing_err(end),
        dist_start=float(np.linalg.norm(pos[s, slot, :2] - bxy[s])),
        park_dist=float(np.linalg.norm(pos[end, slot, :2] - bxy[end])),
        object_height=float(pos[end, slot, 2]),
        path_len=path, displacement=disp,
        straightness=float(disp / path) if path > 1e-6 else float("nan"),
    )


def _worker(args):
    path, max_demos = args
    out_e, out_a, out_s, errs = [], [], [], []
    try:
        with h5py.File(path, "r") as f:
            demos = sorted((k for k in f["data"] if k.startswith("demo_")),
                           key=lambda k: int(k.split("_")[1]))[:max_demos]
    except Exception as exc:  # noqa: BLE001
        return [], [], [], [(path, str(exc).splitlines()[0][:120])]
    for demo in demos:
        try:
            e, a, s = _episode(path, demo)
            out_e += e
            out_a += a
            out_s.append(s)
        except Exception as exc:  # noqa: BLE001
            errs.append((f"{path}:{demo}", str(exc).splitlines()[0][:120]))
    return out_e, out_a, out_s, errs


def _collect(archive: str, max_task: int, per_task: int, demos_per_file: int) -> list[str]:
    files = []
    for name in sorted(os.listdir(archive)):
        m = re.fullmatch(r"task-(\d+)", name)
        if not m or int(m.group(1)) > max_task:
            continue
        d = os.path.join(archive, name)
        hs = sorted(os.path.join(d, n) for n in os.listdir(d) if n.endswith(".hdf5"))
        files += hs[:per_task]
    return files


def _write_csv(path: str, rows: list[dict], first: list[str]) -> None:
    if not rows:
        print(f"  (no rows for {os.path.basename(path)})")
        return
    keys = first + [k for k in rows[0] if k not in first]
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows):6d} rows -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--max-task", type=int, default=49)
    ap.add_argument("--files-per-task", type=int, default=8)
    ap.add_argument("--demos-per-file", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="grasp_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = _collect(args.archive, args.max_task, args.files_per_task, args.demos_per_file)
    print(f"scanning {len(files)} files with {args.workers} workers")

    events, approaches, summaries, errors = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, (p, args.demos_per_file)): p for p in files}
        for i, fut in enumerate(as_completed(futs), 1):
            e, a, s, er = fut.result()
            events += e
            approaches += a
            summaries += s
            errors += er
            if i % 25 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} files, {len(events)} grasps, {len(approaches)} approaches")

    _write_csv(os.path.join(args.out_dir, "grasp_events.csv"), events, ["task_id", "task_name", "demo"])
    _write_csv(os.path.join(args.out_dir, "approach_segments.csv"), approaches, ["task_id", "task_name", "demo"])
    _write_csv(os.path.join(args.out_dir, "episode_summary.csv"), summaries, ["task_id", "task_name", "demo"])

    with open(os.path.join(args.out_dir, "errors.json"), "w") as fp:
        json.dump(errors, fp, indent=1)
    print(f"\n{len(errors)} episode-level failures (see errors.json)")
    for p, m in errors[:10]:
        print(f"  {os.path.basename(p)}: {m}")

    if approaches:
        arr = {k: np.array([a[k] for a in approaches], dtype=float) for k in
               ("heading_err_start", "bearing_err_end", "park_dist", "straightness", "settle_frames")}
        print("\ndemo-side reference distribution (median [p10, p90]):")
        for k, v in arr.items():
            v = v[np.isfinite(v)]
            if v.size:
                unit = " rad" if "err" in k else (" m" if "dist" in k else "")
                print(f"  {k:20s} {np.median(v):7.3f} [{np.percentile(v, 10):.3f}, {np.percentile(v, 90):.3f}]{unit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
