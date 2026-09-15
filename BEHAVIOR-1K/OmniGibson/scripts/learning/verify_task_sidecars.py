"""QC for a task's torque sidecars: completeness + schema + effort sanity.

No sim imports — safe anywhere. Exit 0 only if the task is fully covered and
every episode passes checks.

  python verify_task_sidecars.py --data_folder ~/b1k-data --task-id 0 [--expect 200]
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--expect", type=int, default=0,
                        help="Expected episode count (0 = just report coverage)")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.expanduser(args.data_folder), "replayed_torque", f"task-{args.task_id:04d}")
    seen: dict[int, str] = {}
    dupes, bad = [], []

    for path in sorted(glob.glob(os.path.join(out_dir, "*.hdf5"))):
        name = os.path.basename(path)
        try:
            with h5py.File(path, "r") as f:
                attrs = f["data"].attrs
                if "manifest" in attrs:
                    pairs = [(e["group"], int(e["demo_id"])) for e in json.loads(attrs["manifest"])]
                elif "demo_id" in attrs:
                    pairs = [("demo_0", int(attrs["demo_id"]))]
                else:
                    bad.append((name, "no identity attrs (partial output)"))
                    continue
                if int(attrs.get("task_id", args.task_id)) != args.task_id:
                    bad.append((name, f"task_id mismatch: {attrs.get('task_id')}"))
                    continue
                layout = json.loads(attrs["proprio_layout"])
                s0, s1 = layout["joint_qeffort"]
                for group, demo_id in pairs:
                    if demo_id in seen:
                        dupes.append((demo_id, seen[demo_id], name))
                        continue
                    p = f[f"data/{group}/obs/robot_r1::proprio"]
                    if p.shape[0] < 100 or p.shape[1] != 229:
                        bad.append((f"{name}:{group}", f"shape {p.shape}"))
                        continue
                    eff = p[:, s0:s1]
                    norms = np.linalg.norm(eff, axis=1)
                    active = norms > 0.5
                    med = float(np.median(norms[active])) if active.mean() > 0.05 else -1.0
                    if not (5.0 < med < 500.0):
                        bad.append((f"{name}:{group}", f"active-median effort {med:.1f} N·m out of [5, 500]"))
                        continue
                    seen[demo_id] = name
        except OSError as e:
            bad.append((name, f"unreadable: {e}"))

    print(f"task {args.task_id}: {len(seen)} unique episodes OK across "
          f"{len(set(seen.values()))} files")
    for d, a, b in dupes:
        print(f"DUPLICATE demo {d}: {a} and {b}")
    for name, why in bad:
        print(f"BAD {name}: {why}")
    if args.expect and len(seen) != args.expect:
        missing = args.expect - len(seen)
        print(f"COVERAGE: {len(seen)}/{args.expect} ({missing} missing)")
    fail = bool(dupes or bad or (args.expect and len(seen) != args.expect))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
