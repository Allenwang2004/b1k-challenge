"""Pure recording helpers for ``L1RolloutRecorder`` — no omnigibson or Isaac imports.

Kept apart from the wrapper so the file formats can be unit-tested without a simulator. Every rollout becomes
three HDF5 files, each in the schema of the demo archive it mirrors, so the stage-classifier pipeline reads a
rollout exactly as it reads a demo:

* ``sidecar/task-XXXX/<base>.hdf5`` — the torque-sidecar schema (``data/demo_0/obs/robot_r1::proprio``, 229
  columns, ``proprio_layout`` attr), read by ``sof_standard.adapters.b1k_sidecar.read_sidecar_file`` and hence by
  ``extract_sof_frames.py``. Column order is the one ``replay_torque_fixed.py`` recorded: the 61-dim challenge
  state, then joint positions, velocities, measured efforts, applied efforts, gravity and Coriolis (28 each).
* ``bddl/task-XXXX/bddl_<demo_id>.hdf5`` — the BDDL-sidecar archive schema written by ``log_goal_status.py``
  (``frame``, ``atom_status``, ``initial_atom_status``, ``q_score``, ``atom_labels``, ``atom_option`` ...), read by
  ``l1.join.load_archive_episode``.
* ``rgb/task-XXXX/<base>.hdf5`` — the three camera frames the policy sees, every ``RGB_STRIDE`` steps, JPEG-coded
  (the demo videos are lossy too), for pi05 vision features.

plus ``meta/<base>.json`` with identity, outcome and timing. The meta file is written last and atomically: it is
the completion marker. A rollout without one died before it ended and its HDF5 files may be incomplete.

Rollouts get a synthetic 8-digit demo id (``rollout_demo_id``) so they can never collide with a real demo id
(real ones are ``task * 10_000 + demo``, at most 6 digits) and the archive path convention still holds.
"""
from __future__ import annotations

import json
import os
import time

import h5py
import numpy as np

N_DOF = 28
CHALLENGE_DIM = 61
JOINT_KEYS = ("joint_qpos", "joint_qvel", "joint_qeffort")
DECOMP_KEYS = ("joint_qeffort_applied", "joint_gravity", "joint_coriolis")
PROPRIO_DIM = CHALLENGE_DIM + N_DOF * (len(JOINT_KEYS) + len(DECOMP_KEYS))  # 229
BDDL_STRIDE = 30   # the archive's stride (1 Hz at 30 Hz)
RGB_STRIDE = 10    # 3 Hz, the anchor grid the classifier uses
MAX_OPTIONS = 64   # the demo archive's cap (log_goal_status.py --max-options); labels must match the demos'
JPEG_QUALITY = 95


def proprio_layout() -> dict:
    """Same key -> [start, end) map ``replay_torque_fixed.py`` stamps on every demo sidecar."""
    layout, cursor = {"challenge_61": [0, CHALLENGE_DIM]}, CHALLENGE_DIM
    for key in (*JOINT_KEYS, *DECOMP_KEYS):
        layout[key] = [cursor, cursor + N_DOF]
        cursor += N_DOF
    assert cursor == PROPRIO_DIM
    return layout


def rollout_demo_id(task_id: int, instance_id: int, rollout_id: int) -> int:
    """90_000_000 + task*100_000 + instance*100 + rollout. Eight digits, disjoint from real demo ids."""
    if not (0 <= task_id < 100 and 0 <= instance_id < 1000 and 0 <= rollout_id < 100):
        raise ValueError(f"id out of range: task {task_id}, instance {instance_id}, rollout {rollout_id}")
    return 90_000_000 + task_id * 100_000 + instance_id * 100 + rollout_id


def rollout_basename(task_name: str, instance_id: int, rollout_id: int) -> str:
    """Matches the evaluator's own json/video names: ``<task>_<instance>_<rollout>``."""
    return f"{task_name}_{instance_id}_{rollout_id}"


def option_slices(option_sizes) -> list[tuple[int, int]]:
    out, c = [], 0
    for n in option_sizes:
        out.append((c, c + int(n)))
        c += int(n)
    return out


def credited_q(arow: np.ndarray, init: np.ndarray, slices) -> float:
    """metrics/task_metric.py: max over options of literals true now AND not true at the start, over |option|."""
    if not slices:
        return 0.0
    return max(float(np.sum(arow[s:e] & (1 - init[s:e]))) / max(1, e - s) for s, e in slices)


def raw_reward(arow: np.ndarray, slices) -> float:
    """bddl.activity.get_reward: no initially-true exclusion."""
    if not slices:
        return 0.0
    return max(float(arow[s:e].sum()) / max(1, e - s) for s, e in slices)


def _flatten_body(body) -> str:
    if isinstance(body, str):
        return body
    return " ".join(_flatten_body(b) for b in body)


def atom_label(node) -> str:
    """Verbatim twin of ``log_goal_status.atom_label`` — the demo archive's literal names."""
    cls = type(node).__name__
    if cls == "HEAD" and node.children:
        return atom_label(node.children[0])
    if cls == "Negation" and node.children:
        return "not " + atom_label(node.children[0])
    state = getattr(node, "STATE_NAME", None)
    body = _flatten_body(getattr(node, "body", "")).strip()
    if state:
        args = [str(getattr(node, a)) for a in ("input1", "input2", "input")
                if getattr(node, a, None) is not None]
        if not args and body:
            args = [t.lstrip("?") for t in body.split() if t.lower() != state.lower()]
        return f"{state}({', '.join(args)})"
    return f"{cls}({body[:90]})" if body else cls


def goal_atoms(all_options, max_options: int = MAX_OPTIONS):
    """(atoms, option_sizes, n_options_total) with atoms = [(option, position, label, node)], capped like the archive."""
    all_options = list(all_options)
    options = all_options[:max_options]
    atoms = [(oi, pi, atom_label(nd), nd) for oi, opt in enumerate(options) for pi, nd in enumerate(opt)]
    return atoms, [len(o) for o in options], len(all_options)


def encode_rgb(img: np.ndarray, quality: int = JPEG_QUALITY) -> tuple[np.ndarray, str]:
    """(bytes as uint8 array, codec). JPEG via OpenCV (in the sim image), else PIL, else raw."""
    img = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[..., :3])
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", img[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            return np.asarray(buf, np.uint8).reshape(-1), "jpeg"
    except ImportError:
        pass
    try:
        import io

        from PIL import Image

        b = io.BytesIO()
        Image.fromarray(img).save(b, format="JPEG", quality=int(quality))
        return np.frombuffer(b.getvalue(), np.uint8), "jpeg"
    except ImportError:
        return img.reshape(-1), "raw"


def decode_rgb(buf: np.ndarray, codec: str, shape) -> np.ndarray:
    """Inverse of ``encode_rgb``: an (H, W, 3) uint8 RGB image."""
    buf = np.asarray(buf, np.uint8)
    if codec == "raw":
        return buf.reshape(tuple(shape))
    try:
        import cv2

        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    except ImportError:
        import io

        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(buf.tobytes())).convert("RGB"))


class _Rows:
    """An appendable, chunked HDF5 dataset that grows one row at a time, so its size is always the row count."""

    def __init__(self, group, name, row_shape, dtype, chunk_rows=256, compression="lzf"):
        self.ds = group.create_dataset(name, shape=(0, *row_shape), maxshape=(None, *row_shape),
                                       dtype=dtype, chunks=(chunk_rows, *row_shape), compression=compression)
        self.n = 0

    def append(self, row):
        self.ds.resize(self.n + 1, axis=0)
        self.ds[self.n] = row
        self.n += 1


class RolloutFiles:
    """Open, fill and finalize the three files of one rollout."""

    def __init__(self, out_dir: str, task_id: int, task_name: str, instance_id: int, rollout_id: int,
                 robot_name: str, action_dim: int, atom_labels, atom_option, option_sizes,
                 n_options_total: int, n_goal_predicates: int, camera_keys, extra_meta: dict | None = None,
                 bddl_attrs: dict | None = None):
        self.task_id, self.task_name = int(task_id), str(task_name)
        self.instance_id, self.rollout_id = int(instance_id), int(rollout_id)
        self.demo_id = rollout_demo_id(self.task_id, self.instance_id, self.rollout_id)
        self.base = rollout_basename(self.task_name, self.instance_id, self.rollout_id)
        self.slices = option_slices(option_sizes)
        self.camera_keys = list(camera_keys)
        self.n_atoms = len(atom_labels)
        self.t0 = time.time()
        self.extra = dict(extra_meta or {})
        tdir = f"task-{self.task_id:04d}"
        self.paths = {
            "sidecar": os.path.join(out_dir, "sidecar", tdir, f"{self.base}.hdf5"),
            "bddl": os.path.join(out_dir, "bddl", tdir, f"bddl_{self.demo_id:08d}.hdf5"),
            "rgb": os.path.join(out_dir, "rgb", tdir, f"{self.base}.hdf5"),
            "meta": os.path.join(out_dir, "meta", f"{self.base}.json"),
        }
        for p in self.paths.values():
            os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.exists(self.paths["meta"]):
            os.remove(self.paths["meta"])  # a re-run of this rollout: the old marker must not vouch for new files

        # sidecar
        self.f_side = h5py.File(self.paths["sidecar"], "w")
        d = self.f_side.create_group("data")
        d.attrs.update({"task_id": self.task_id, "task_name": self.task_name, "demo_id": self.demo_id,
                        "proprio_layout": json.dumps(proprio_layout()),
                        "manifest": json.dumps([{"group": "demo_0", "demo_id": self.demo_id}]),
                        "source": "rollout", "instance_id": self.instance_id, "rollout_id": self.rollout_id})
        g = d.create_group("demo_0")
        g.attrs["demo_id"] = self.demo_id
        self.side_group = g
        obs = g.create_group("obs")
        self.proprio = _Rows(obs, f"{robot_name}::proprio", (PROPRIO_DIM,), np.float32)
        self.action = _Rows(g, "action", (int(action_dim),), np.float32)
        self.wall = _Rows(g, "wall_time", (), np.float64, chunk_rows=1024)

        # bddl
        self.f_bddl = h5py.File(self.paths["bddl"], "w")
        b = self.f_bddl.create_group("data")
        b.attrs.update({"task_id": self.task_id, "task_name": self.task_name, "demo_id": self.demo_id,
                        "stride": BDDL_STRIDE, "n_atoms": self.n_atoms, "n_options": len(option_sizes),
                        "n_options_total": int(n_options_total),
                        "option_sizes": json.dumps([int(x) for x in option_sizes]),
                        "atom_labels": json.dumps([str(x) for x in atom_labels]),
                        "atom_option": json.dumps([int(x) for x in atom_option]),
                        "n_goal_predicates": int(n_goal_predicates), "source": "rollout",
                        "instance_id": self.instance_id, "rollout_id": self.rollout_id,
                        **(bddl_attrs or {})})
        self.bddl_group = b
        self.frame = _Rows(b, "frame", (), np.int64, chunk_rows=64)
        self.atoms = _Rows(b, "atom_status", (self.n_atoms,), np.uint8, chunk_rows=64)
        self.goal = _Rows(b, "goal_status", (int(n_goal_predicates),), np.uint8, chunk_rows=64)
        self.q = _Rows(b, "q_score", (), np.float32, chunk_rows=64)
        self.raw = _Rows(b, "bddl_reward", (), np.float32, chunk_rows=64)
        self.init = None

        # rgb
        self.f_rgb = h5py.File(self.paths["rgb"], "w")
        r = self.f_rgb.create_group("data")
        r.attrs.update({"task_id": self.task_id, "demo_id": self.demo_id, "stride": RGB_STRIDE,
                        "camera_keys": json.dumps(self.camera_keys)})
        self.rgb_group = r
        self.rgb_frame = _Rows(r, "frame_idx", (), np.int64, chunk_rows=64)
        self.rgb = {}
        self.closed = False

    # --- writes -------------------------------------------------------------------------------------------
    def set_initial(self, init_row: np.ndarray):
        self.init = np.asarray(init_row, dtype=np.uint8)
        self.bddl_group.create_dataset("initial_atom_status", data=self.init)

    def add_step(self, proprio: np.ndarray, action: np.ndarray, wall: float):
        proprio = np.asarray(proprio, np.float32)
        if proprio.shape != (PROPRIO_DIM,):
            raise ValueError(f"proprio has shape {proprio.shape}, expected ({PROPRIO_DIM},)")
        a = np.zeros(self.action.ds.shape[1], np.float32)
        flat = np.asarray(action, dtype=np.float32).reshape(-1)[: len(a)]
        a[: len(flat)] = flat
        self.proprio.append(proprio)
        self.action.append(a)
        self.wall.append(float(wall))

    def add_atoms(self, step: int, arow: np.ndarray, goal_row: np.ndarray) -> float:
        if self.init is None:
            raise RuntimeError("add_atoms before set_initial")
        arow = np.asarray(arow, dtype=np.uint8)
        q = credited_q(arow, self.init, self.slices)
        self.frame.append(int(step))
        self.atoms.append(arow)
        self.goal.append(np.asarray(goal_row, dtype=np.uint8))
        self.q.append(q)
        self.raw.append(raw_reward(arow, self.slices))
        return q

    def add_rgb(self, step: int, images: dict):
        for key in self.camera_keys:
            img = np.asarray(images[key])
            buf, codec = encode_rgb(img)
            if key not in self.rgb:
                rows = _Rows(self.rgb_group, key, (), h5py.vlen_dtype(np.uint8), chunk_rows=16, compression=None)
                rows.ds.attrs.update({"codec": codec, "shape": json.dumps([int(x) for x in img.shape[:2]] + [3])})
                self.rgb[key] = rows
            self.rgb[key].append(buf)
        self.rgb_frame.append(int(step))

    def flush(self):
        for f in (self.f_side, self.f_bddl, self.f_rgb):
            f.flush()

    @property
    def last_q(self) -> float | None:
        if self.closed:
            return getattr(self, "_last_q", None)
        return float(self.q.ds[self.q.n - 1]) if self.q.n else None

    # --- close ------------------------------------------------------------------------------------------------
    def finalize(self, n_steps: int, terminated: bool, truncated: bool, success: bool | None,
                 status: str = "ok", note: str = "", extra: dict | None = None):
        if self.closed:
            return self.paths
        q_final = self.last_q
        self._last_q = q_final
        self.closed = True
        self.side_group.attrs["num_samples"] = int(n_steps)
        self.bddl_group.attrs["n_steps_total"] = int(n_steps)
        self.bddl_group.attrs["n_steps_evaluated"] = int(n_steps)
        if self.init is None:  # died before the first step: keep the file loadable
            self.bddl_group.create_dataset("initial_atom_status", data=np.zeros(self.n_atoms, np.uint8))
        n = self.frame.n
        self.bddl_group.create_dataset("reward", data=np.zeros(n, np.float32))
        term = np.zeros(n, bool)
        if n and (terminated or truncated):
            term[-1] = True
        self.bddl_group.create_dataset("terminated", data=term)
        for f in (self.f_side, self.f_bddl, self.f_rgb):
            f.close()
        meta = {
            "task_id": self.task_id, "task_name": self.task_name, "instance_id": self.instance_id,
            "rollout_id": self.rollout_id, "demo_id": self.demo_id, "steps": int(n_steps),
            "terminated": bool(terminated), "truncated": bool(truncated), "success": success,
            "q_final_recorded": q_final, "status": status, "note": note,
            "started_at": self.t0, "finished_at": time.time(), "files": self.paths,
            "bddl_rows": n, "rgb_rows": self.rgb_frame.n, "camera_keys": self.camera_keys,
            **self.extra, **(extra or {}),
        }
        tmp = self.paths["meta"] + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        os.replace(tmp, self.paths["meta"])
        return self.paths
