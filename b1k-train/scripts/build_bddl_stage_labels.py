"""Turn the BDDL sidecar archive into per-frame stage labels for training.

The 2026 demos ship no record of task progress, so `ComputeSubtaskStateFromMeta` splits each
episode into equal time slices. This script produces the alternative: the *symbolic* progress,
read from the BDDL sidecars (`b1k-bddl-sidecars-archive`, one HDF5 per episode).

    stage(frame) = max over solution options of (# goal literals of that option that hold)

which is the counting `q_score` uses, so stage 0 = nothing done and stage N = task complete.

Two things the sidecars do not give us directly and this script handles:

* **Rate.** A sidecar samples every `stride`-th simulator step (30 by default), so it has
  ceil(T/stride) rows while the demo has T frames. We expand with the archive README's recipe:
  hold each value until the next sample (correct for booleans; sub-stride transitions are
  invisible, i.e. 1 s of timing resolution).
* **Join.** The sidecar's key is `demo_id`, the demos' key is `episode_index`. Sorting both and
  pairing them positionally lines up, and we *verify* it: every pair must agree on episode
  length, otherwise the script fails rather than training on misaligned labels.

Run it with a python that has h5py (the eval venv does, b1k-train's does not):

    b1k-evaluation/baselines/openpi/.venv/bin/python scripts/build_bddl_stage_labels.py --tasks 1

Output: one .npz holding `stage/<episode_index>` (int8, length = episode length) plus a JSON
sidecar with the per-task stage count, both under outputs/assets/bddl_stage_labels/.
"""

import argparse
import json
import pathlib

import h5py
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO.parent


def _attr(group, key):
    """Sidecar attributes are sometimes JSON strings, sometimes arrays."""
    value = group.attrs[key]
    if isinstance(value, (str, bytes)):
        return np.asarray(json.loads(value))
    return np.asarray(value)


def read_sidecar(path: pathlib.Path) -> dict:
    """One episode: its demo id, length, and the per-sample stage (satisfied count, best option)."""
    with h5py.File(path, "r") as f:
        data = f["data"]
        atom_status = data["atom_status"][:]  # [N, A] uint8
        frame = data["frame"][:]  # [N] int64, the simulator step of each row
        demo_id = int(data.attrs["demo_id"])
        num_steps = int(data.attrs["n_steps_total"])
        atom_option = _attr(data, "atom_option").reshape(-1)  # [A] which option each literal belongs to

    options = sorted(set(atom_option.tolist()))
    per_option = np.stack([atom_status[:, atom_option == o].sum(axis=1) for o in options], axis=1)
    stage = per_option.max(axis=1).astype(np.int16)  # [N]
    option_sizes = [int((atom_option == o).sum()) for o in options]
    return {
        "demo_id": demo_id,
        "num_steps": num_steps,
        "frame": frame,
        "stage": stage,
        "max_stage": max(option_sizes),
    }


def expand_to_full_rate(frame: np.ndarray, stage: np.ndarray, num_steps: int) -> np.ndarray:
    """Hold each sampled value until the next sample (archive README's recipe)."""
    idx = np.searchsorted(frame, np.arange(num_steps), side="right") - 1
    return stage[np.clip(idx, 0, len(stage) - 1)]


def episode_index_map(demos_root: pathlib.Path, task_index: int) -> pd.DataFrame:
    """The demos' episodes for one task: episode_index + length, ordered by episode_index."""
    files = sorted((demos_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no meta/episodes/**/*.parquet under {demos_root}")
    meta = pd.concat(pd.read_parquet(p) for p in files)
    task = meta[meta["task_index"] == task_index].sort_values("episode_index")
    if task.empty:
        raise ValueError(f"no episodes with task_index={task_index} in {demos_root}")
    return task[["episode_index", "length"]].reset_index(drop=True)


def build_task(archive_dir: pathlib.Path, demos_root: pathlib.Path, task_index: int) -> tuple[dict, int]:
    paths = sorted((archive_dir / f"task-{task_index:04d}").glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no sidecars in {archive_dir}/task-{task_index:04d}")
    sidecars = sorted((read_sidecar(p) for p in paths), key=lambda s: s["demo_id"])
    episodes = episode_index_map(demos_root, task_index)

    if len(sidecars) != len(episodes):
        raise ValueError(
            f"task {task_index}: {len(sidecars)} sidecars but {len(episodes)} episodes in the demos. "
            "The positional join is only valid when both cover the same set."
        )

    labels: dict[int, np.ndarray] = {}
    max_stage = 0
    non_monotonic = 0
    for sidecar, (episode_index, length) in zip(sidecars, episodes.itertuples(index=False), strict=True):
        if sidecar["num_steps"] != int(length):
            raise ValueError(
                f"task {task_index}: sidecar demo_id={sidecar['demo_id']} has {sidecar['num_steps']} steps but "
                f"episode {int(episode_index)} has {int(length)} frames -- the demo_id -> episode_index join is wrong."
            )
        full = expand_to_full_rate(sidecar["frame"], sidecar["stage"], sidecar["num_steps"])
        if (np.diff(full) < 0).any():
            non_monotonic += 1
        labels[int(episode_index)] = full.astype(np.int8)
        max_stage = max(max_stage, sidecar["max_stage"])

    num_stages = max_stage + 1  # stage 0 (nothing done) .. max_stage (all literals of one option)
    reached = max(int(v.max()) for v in labels.values())
    print(
        f"task {task_index}: {len(labels)} episodes, stages 0..{max_stage} ({num_stages} levels), "
        f"highest reached in any demo = {reached}, non-monotonic episodes = {non_monotonic}"
    )
    return labels, num_stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="1", help="comma-separated task indices, e.g. 1 or 0,1,17")
    parser.add_argument("--archive-dir", type=pathlib.Path,
                        default=pathlib.Path.home() / "b1k-scrub" / "b1k-bddl-sidecars-archive")
    parser.add_argument("--demos-root", type=pathlib.Path,
                        default=EVAL_ROOT / "train_set" / "2026-challenge-demos")
    parser.add_argument("--out", type=pathlib.Path,
                        default=REPO / "outputs" / "assets" / "bddl_stage_labels" / "2026-challenge-demos.npz")
    args = parser.parse_args()

    task_indices = [int(t) for t in args.tasks.split(",") if t.strip()]
    arrays: dict[str, np.ndarray] = {}
    num_stages: dict[str, int] = {}
    for task_index in task_indices:
        labels, stages = build_task(args.archive_dir, args.demos_root, task_index)
        num_stages[str(task_index)] = stages
        for episode_index, values in labels.items():
            arrays[f"stage/{episode_index}"] = values

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps({
        "source": str(args.archive_dir),
        "demos_root": str(args.demos_root),
        "definition": "max over solution options of the number of satisfied goal literals (q_score counting)",
        "stride_note": "sidecars sample every 30th step; values are held until the next sample (1 s resolution)",
        "tasks": task_indices,
        "num_stages": num_stages,
        "num_episodes": len(arrays),
    }, indent=2) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) and {meta_path.name}")


if __name__ == "__main__":
    main()
