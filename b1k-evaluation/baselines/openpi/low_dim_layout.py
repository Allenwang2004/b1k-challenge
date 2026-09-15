"""Recover the column->key map for the ``task::low_dim`` observation channel.

The sidecars stamp a ``proprio_layout`` attr for the 229-dim proprio vector but
nothing for ``task::low_dim``, so that channel has been unusable: it is a flat
float array whose column meaning depends on ``BehaviorTask.object_scope``
iteration order, which only exists inside the recording process.

That order is worse than undocumented -- it is *not stable across runs*.
``bddl.activity.get_object_scope`` builds the scope from a Python ``set``
(``condition_evaluation.py:673``), so with hash randomisation the column order
of the same task differs between two recordings. The layout is therefore a
per-file property and must be recovered per file, never assumed.

Recovery works because ``update_bddl_scope_metadata`` writes ``inst_to_name``
into the scene metadata using exactly the object_scope iteration order
(``behavior_task.py:570``), and the sidecar stores the whole scene json in
``data.attrs["scene_file"]``. Replaying the emission rules of
``BehaviorTask._get_obs`` (``behavior_task.py:573-617``) over that ordered list
reproduces the layout.

Per non-system instance the emission is::

    {inst}_real      1
    {inst}_pos       3   world xyz
    {inst}_ori_cos   3   cos of world rpy
    {inst}_ori_sin   3   sin of world rpy
    {inst}_in_gripper_{arm}   1 per arm   -- SKIPPED for the agent itself

so every instance costs 12 columns except the agent (always slot 0), which
costs 10. Total width is ``12*n - 2``, which pins ``n`` exactly.

One wrinkle: ``inst_to_name`` only lists instances whose entity is not None, so
instances that do not exist yet -- sliced products such as
``half__brussels_sprouts.n.01_*`` -- are absent from it while still occupying
12 columns each. Those slots are found from the data instead: a non-existent
instance emits ``real=0``, an existing one ``real=1``, so the frame-0 real bits
mark exactly which slots the named instances go into. The residual slots are
named from the task's BDDL ``:objects`` declaration (vendored in
``bddl_instances.json``); when they all share one synset -- the usual case --
that naming is exact up to an index permutation that carries no information.

Substance synsets are vendored in ``substance_synsets.json``, so this module
needs neither Isaac nor bddl3: h5py and numpy only, i.e. it runs on the
analysis venv.

Usage::

    from low_dim_layout import recover_layout, grasp_flags
    layout = recover_layout(h5file)                 # dict key -> (start, end)
    flags, objs = grasp_flags(h5file, "demo_0")     # (T, n_obj, n_arm) bool

CLI::

    python low_dim_layout.py <sidecar.hdf5> [--demo demo_0]
    python low_dim_layout.py --audit <archive-root>   # width check over a tree
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache

import h5py
import numpy as np

ARMS = ("left", "right")  # verified against gripper qpos, see --validate-arms
BLOCK = ("real", "pos", "ori_cos", "ori_sin")
BLOCK_WIDTH = {"real": 1, "pos": 3, "ori_cos": 3, "ori_sin": 3}
AGENT_PREFIX = "agent.n.01"

_HERE = os.path.dirname(os.path.abspath(__file__))
_SUBSTANCES_PATH = os.path.join(_HERE, "substance_synsets.json")
_BDDL_INSTS_PATH = os.path.join(_HERE, "bddl_instances.json")

# Gripper finger joint columns inside the 229-dim proprio vector (challenge_61).
GRIPPER_QPOS_COLS = {"left": (24, 26), "right": (49, 51)}


@lru_cache(maxsize=1)
def _substance_synsets() -> frozenset[str]:
    with open(_SUBSTANCES_PATH) as fp:
        return frozenset(json.load(fp))


@lru_cache(maxsize=1)
def _bddl_instances() -> dict[str, list[str]]:
    try:
        with open(_BDDL_INSTS_PATH) as fp:
            return json.load(fp)
    except FileNotFoundError:
        return {}


def synset_of(inst: str) -> str:
    """'almond.n.01_1' -> 'almond.n.01' (mirrors bddl_utils.synset_from_bddl_inst)."""
    return "_".join(inst.split("_")[:-1])


def is_system_inst(inst: str) -> bool:
    return synset_of(inst) in _substance_synsets()


def _scene_metadata(h5: h5py.File) -> dict:
    raw = h5["data"].attrs.get("scene_file")
    if raw is None:
        raise KeyError("sidecar has no scene_file attr; cannot recover low_dim layout")
    meta = json.loads(raw).get("metadata", {}).get("task", {})
    # Some scene jsons nest the task metadata one or two levels deep (task/task/...).
    while isinstance(meta, dict) and "inst_to_name" not in meta and "task" in meta:
        meta = meta["task"]
    if not isinstance(meta, dict) or "inst_to_name" not in meta:
        raise KeyError("scene_file metadata has no inst_to_name")
    return meta


def _named_order(h5: h5py.File) -> list[str]:
    """Non-system instances that existed at record time, in object_scope order."""
    insts = [inst for inst in _scene_metadata(h5)["inst_to_name"] if not is_system_inst(inst)]
    if not insts or not insts[0].startswith(AGENT_PREFIX):
        # object_scope is built agent-first (behavior_task.py:303,321). If that
        # is not what we see, the ordering assumption is broken -- refuse.
        raise ValueError(f"inst_to_name does not start with the agent: {insts[:3]}")
    return insts


def _first_demo(h5: h5py.File) -> str:
    return next(k for k in h5["data"] if k.startswith("demo_"))


def _low_dim(h5: h5py.File, demo: str | None = None) -> h5py.Dataset:
    return h5["data"][demo or _first_demo(h5)]["obs/task::low_dim"]


def instance_order(h5: h5py.File, demo: str | None = None) -> list[str]:
    """Full slot->instance list, including instances that did not exist at t=0.

    Slots whose occupant cannot be named are returned as
    ``"<unknown>_<synset>_slot{i}"`` (or ``"<unknown>_slot{i}"`` when the BDDL
    residual is not a single synset), so callers can always index by slot even
    if a name is unavailable.
    """
    named = _named_order(h5)
    ld = _low_dim(h5, demo)
    width = int(ld.shape[1])
    if (width + 2) % 12 != 0:
        raise ValueError(f"task::low_dim width {width} is not of the form 12*n-2; layout rules changed?")
    n = (width + 2) // 12
    if n == len(named):
        return list(named)
    if n < len(named):
        raise ValueError(f"width implies {n} instances but inst_to_name lists {len(named)} -- metadata/obs mismatch")

    # Some slots are instances that did not exist at record start. Find them
    # from the frame-0 `real` bit: slot 0 is the agent (10 cols), the rest 12.
    row0 = np.asarray(ld[0])
    starts = [0] + [10 + 12 * i for i in range(n - 1)]
    present = [True] + [bool(row0[s] > 0.5) for s in starts[1:]]
    if sum(present) != len(named):
        raise ValueError(
            f"{sum(present)} slots report real=1 at frame 0 but inst_to_name lists "
            f"{len(named)} existing instances -- cannot align slots"
        )

    task = h5["data"].attrs.get("task_name")
    task = task.decode() if isinstance(task, bytes) else task
    residual = [i for i in _bddl_instances().get(task, []) if not is_system_inst(i) and i not in set(named)]
    residual_synsets = {synset_of(i) for i in residual}
    single = residual_synsets.pop() if len(residual_synsets) == 1 else None

    order, it_named, it_res = [], iter(named), iter(residual)
    for slot, is_present in enumerate(present):
        if is_present:
            order.append(next(it_named))
        elif single is not None:
            order.append(next(it_res, f"<unknown>_{single}_slot{slot}"))
        else:
            order.append(f"<unknown>_slot{slot}")
    return order


def recover_layout(h5: h5py.File, demo: str | None = None, arms: tuple[str, ...] = ARMS) -> dict[str, tuple[int, int]]:
    """key -> (start, end) column slice for ``task::low_dim``.

    Raises ValueError if the reconstructed width disagrees with the file, which
    is the guard against a silently wrong ordering.
    """
    insts = instance_order(h5, demo)
    layout: dict[str, tuple[int, int]] = {}
    cursor = 0
    for slot, inst in enumerate(insts):
        for field in BLOCK:
            w = BLOCK_WIDTH[field]
            layout[f"{inst}_{field}"] = (cursor, cursor + w)
            cursor += w
        if slot != 0:  # the agent (slot 0) emits no in_gripper flags for itself
            for arm in arms:
                layout[f"{inst}_in_gripper_{arm}"] = (cursor, cursor + 1)
                cursor += 1

    width = int(_low_dim(h5, demo).shape[1])
    if cursor != width:
        raise ValueError(f"low_dim layout mismatch: reconstructed {cursor} columns but the file has {width}")
    return layout


def grasp_flags(h5: h5py.File, demo: str, arms: tuple[str, ...] = ARMS):
    """Ground-truth in-gripper flags.

    Returns ``(flags, objects)`` where ``flags`` is a (T, n_obj, n_arm) bool
    array and ``objects`` the matching instance names in slot order (agent
    excluded). These come from ``agent.is_grasping(arm, candidate_obj)``, i.e.
    the simulator's own grasp assertion -- not a gripper-width heuristic.
    """
    layout = recover_layout(h5, demo, arms)
    ld = np.asarray(_low_dim(h5, demo))
    objs = instance_order(h5, demo)[1:]
    out = np.zeros((ld.shape[0], len(objs), len(arms)), dtype=bool)
    for oi, inst in enumerate(objs):
        for ai, arm in enumerate(arms):
            s, _ = layout[f"{inst}_in_gripper_{arm}"]
            out[:, oi, ai] = ld[:, s] > 0.5
    return out, objs


def object_positions(h5: h5py.File, demo: str, arms: tuple[str, ...] = ARMS):
    """World xyz per instance in slot order (agent first).

    Returns ``(pos, real, objects)`` with pos (T, n, 3) and real (T, n) bool.
    Positions of instances with ``real=0`` are zeros, not missing data -- always
    mask with ``real`` before using them.
    """
    layout = recover_layout(h5, demo, arms)
    ld = np.asarray(_low_dim(h5, demo))
    insts = instance_order(h5, demo)
    pos = np.zeros((ld.shape[0], len(insts), 3), dtype=np.float32)
    real = np.zeros((ld.shape[0], len(insts)), dtype=bool)
    for i, inst in enumerate(insts):
        s, e = layout[f"{inst}_pos"]
        pos[:, i] = ld[:, s:e]
        real[:, i] = ld[:, layout[f"{inst}_real"][0]] > 0.5
    return pos, real, insts


def layout_from_object_scope(object_scope, arm_names, is_system=None) -> dict[str, list[int]]:
    """Build the layout directly from a live ``BehaviorTask.object_scope``.

    This is the authoritative form and the one recorders should use: it sees
    instances whose entity is None, in their true position, so nothing has to be
    inferred from frame-0 ``real`` bits afterwards. Mirrors ``_get_obs``
    (``behavior_task.py:573-617``) exactly.

    ``is_system`` defaults to the vendored substance table; pass
    ``omnigibson.utils.bddl_utils.is_system_bddl_inst`` when running inside a
    live environment so the knowledge base is the authority rather than a
    snapshot of it.
    """
    is_system = is_system or is_system_inst
    layout: dict[str, list[int]] = {}
    cursor = 0
    for slot, inst in enumerate(i for i in object_scope if not is_system(i)):
        for field in BLOCK:
            w = BLOCK_WIDTH[field]
            layout[f"{inst}_{field}"] = [cursor, cursor + w]
            cursor += w
        if slot != 0:
            for arm in arm_names:
                layout[f"{inst}_in_gripper_{arm}"] = [cursor, cursor + 1]
                cursor += 1
    return layout


def stamp_layout(path: str, arms: tuple[str, ...] = ARMS) -> dict[str, tuple[int, int]]:
    """Write the recovered map into the file as a ``low_dim_layout`` attr so
    downstream readers never repeat this. Idempotent."""
    with h5py.File(path, "r+") as f:
        layout = recover_layout(f, arms=arms)
        f["data"].attrs["low_dim_layout"] = json.dumps({k: list(v) for k, v in layout.items()})
    return layout


# --------------------------------------------------------------------------- CLI


def _iter_hdf5(root: str):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if n.endswith(".hdf5"):
                yield os.path.join(dirpath, n)


def _audit(root: str, limit: int | None, stamp: bool) -> int:
    files = sorted(_iter_hdf5(root))
    if limit:
        files = files[:limit]
    ok = bad = skipped = partial = 0
    failures = []
    for p in files:
        try:
            with h5py.File(p, "r+" if stamp else "r") as f:
                demos = [k for k in f["data"] if k.startswith("demo_")]
                if not demos:
                    skipped += 1
                    continue
                layout = recover_layout(f)
                if any(k.startswith("<unknown>") for k in layout):
                    partial += 1
                if stamp:
                    f["data"].attrs["low_dim_layout"] = json.dumps({k: list(v) for k, v in layout.items()})
            ok += 1
        except Exception as exc:  # noqa: BLE001 - an audit reports, it never raises
            bad += 1
            failures.append((p, str(exc).splitlines()[0][:160]))
    verb = "stamped" if stamp else "checked"
    print(f"low_dim layout {verb} over {len(files)} files: {ok} ok ({partial} with unnamed slots), "
          f"{bad} failed, {skipped} no-demos")
    for p, msg in failures[:20]:
        print(f"  FAIL {os.path.relpath(p, root)}: {msg}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    return 0 if bad == 0 else 1


def _validate_arms(root: str, limit: int | None) -> int:
    """The (left, right) order is an assumption; this is its empirical test.

    During frames where the LEFT in_gripper flag is set the LEFT gripper should
    be narrower than when it is free. If the order were swapped the association
    would invert, so the sign of the delta is the discriminator.
    """
    files = sorted(_iter_hdf5(root))
    if limit:
        files = files[:limit]
    deltas = {arm: [] for arm in ARMS}
    for p in files:
        with h5py.File(p, "r") as f:
            for demo in [k for k in f["data"] if k.startswith("demo_")]:
                try:
                    flags, _ = grasp_flags(f, demo)
                except Exception:  # noqa: BLE001
                    continue
                obs = f["data"][demo]["obs"]
                pkey = next((k for k in obs if "proprio" in k), None)
                if pkey is None or obs[pkey].shape[1] < 229:
                    continue
                pr = np.asarray(obs[pkey])
                for ai, arm in enumerate(ARMS):
                    s, e = GRIPPER_QPOS_COLS[arm]
                    width = pr[:, s:e].mean(axis=1)
                    held = flags[:, :, ai].any(axis=1)
                    if held.sum() < 30 or (~held).sum() < 30:
                        continue
                    deltas[arm].append(float(width[held].mean() - width[~held].mean()))
    bad = False
    for arm in ARMS:
        d = np.array(deltas[arm])
        if d.size == 0:
            print(f"{arm:5s}: no usable episodes")
            continue
        neg = int((d < 0).sum())
        print(f"{arm:5s}: n={d.size}  mean(width_held - width_free)={d.mean():+.4f}  narrower in {neg}/{d.size}")
        if d.mean() >= 0:
            bad = True
    print("\nARMS order is confirmed when both deltas are negative (gripper narrower while grasping).")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="sidecar hdf5, or a tree with --audit/--validate-arms")
    ap.add_argument("--demo", default=None)
    ap.add_argument("--audit", action="store_true", help="walk a tree and check the layout reconstructs")
    ap.add_argument("--stamp", action="store_true", help="write the low_dim_layout attr (works with --audit)")
    ap.add_argument("--validate-arms", action="store_true", help="empirically test the (left, right) assumption")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.validate_arms:
        return _validate_arms(args.path, args.limit)
    if args.audit:
        return _audit(args.path, args.limit, args.stamp)
    if args.stamp:
        layout = stamp_layout(args.path)
        print(f"stamped low_dim_layout ({len(layout)} keys) into {args.path}")
        return 0

    with h5py.File(args.path, "r") as f:
        demo = args.demo or _first_demo(f)
        layout = recover_layout(f, demo)
        print(f"{args.path} [{demo}] task={f['data'].attrs.get('task_name')}")
        print(f"  {len(instance_order(f, demo))} instances -> {max(e for _, e in layout.values())} columns")
        for k, (s, e) in layout.items():
            print(f"  [{s:4d}:{e:4d}] {k}")
        flags, objs = grasp_flags(f, demo)
        print("\n  grasp summary (frames held / onsets):")
        for oi, inst in enumerate(objs):
            for ai, arm in enumerate(ARMS):
                v = flags[:, oi, ai]
                if not v.any():
                    continue
                onsets = int(np.sum(v[1:] & ~v[:-1])) + int(v[0])
                print(f"    {inst:38s} {arm:5s} held={int(v.sum()):6d}  onsets={onsets}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
