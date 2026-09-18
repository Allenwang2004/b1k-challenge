"""Eval wrapper that records a live policy rollout in the schemas of the demo archives.

Usage (Allen's evaluator, image ``b1k-sim``)::

    python -m omnigibson.eval.eval ... --env-wrapper omnigibson.eval.wrappers.l1_rollout_recorder.L1RolloutRecorder

The wrapper is ``DefaultWrapper`` (224 px RGB, the setting of the ``rlc_ckpt2`` reference rollouts) plus a recorder.
It never changes what the policy sees or does: it reads robot and task state after each ``env.step`` and writes it
to ``<--output-dir>/l1_record/`` (or ``$L1_RECORD_DIR``); see ``l1_record_io`` for the files. Per rollout it stores

* every step: the 229-column proprio vector of the demo torque sidecars (challenge state, joint position,
  velocity, measured effort, and the applied / gravity / Coriolis decomposition read exactly as
  ``replay_torque_fixed.py`` read it) and the action;
* every 10th step: the three camera frames;
* every 30th step and the last: the truth of every goal literal of the first 64 goal options, the per-condition
  goal status, and the credited q (``metrics/task_metric.py``) against the literal truth before the first step --
  the same snapshot ``TaskMetric.reset`` takes.

Recording can never break a rollout: any recorder error is logged once, recording stops for the rest of the
process, and the rollout continues. ``L1_RECORD_DISABLE=1`` turns the recorder off; ``L1_RECORD_RGB=0`` skips the
camera frames; ``L1_RECORD_META`` (a JSON object) is copied into every rollout's meta file.

Instance and rollout ids come from ``Evaluator.start_recording``'s video path (``<task>_<instance>_<rollout>.mp4``,
always set with ``--write-video``), falling back to ``Evaluator.load_task_instance`` and a per-instance counter.
"""
import atexit
import json
import logging
import os
import re
import sys
import time

import numpy as np
import torch as th

from omnigibson.envs import Environment
from omnigibson.eval.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
    get_robot_camera_names,
)
from omnigibson.eval.wrappers import l1_record_io as rio
from omnigibson.eval.wrappers.default_wrapper import DefaultWrapper
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)  # as evaluator.py does: the recorder's lines belong in the task log

C61_KEYS = tuple(PROPRIOCEPTION_INDICES["R1Pro"].keys())
_VIDEO_RE = re.compile(r"_(\d+)_(\d+)\.mp4$")
_IDS = {"instance_id": None, "rollout_id": None, "counter": 0, "source": None}


def _install_evaluator_hooks() -> bool:
    """Learn the instance and rollout ids from the evaluator. Class-level and idempotent."""
    try:
        from omnigibson.eval.evaluator import Evaluator
    except Exception as exc:  # not running under the evaluator
        logger.warning(f"L1 recorder: evaluator hooks not installed ({exc}); ids fall back to 999/counter")
        return False
    if getattr(Evaluator, "_l1_recorder_hooked", False):
        return True
    orig_load, orig_start = Evaluator.load_task_instance, Evaluator.start_recording

    def load_task_instance(self, instance_id, *args, **kwargs):
        out = orig_load(self, instance_id, *args, **kwargs)
        _IDS.update(instance_id=int(instance_id), rollout_id=None, counter=0, source="load_task_instance")
        return out

    def start_recording(self, fpath, *args, **kwargs):
        m = _VIDEO_RE.search(os.path.basename(str(fpath)))
        if m:
            _IDS.update(instance_id=int(m.group(1)), rollout_id=int(m.group(2)), source="video_path")
        return orig_start(self, fpath, *args, **kwargs)

    Evaluator.load_task_instance = load_task_instance
    Evaluator.start_recording = start_recording
    Evaluator._l1_recorder_hooked = True
    return True


def _output_dir() -> str:
    if os.environ.get("L1_RECORD_DIR"):
        return os.path.expanduser(os.environ["L1_RECORD_DIR"])
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--output-dir" and i + 1 < len(argv):
            return os.path.join(os.path.expanduser(argv[i + 1]), "l1_record")
        if a.startswith("--output-dir="):
            return os.path.join(os.path.expanduser(a.split("=", 1)[1]), "l1_record")
    return os.path.join(os.getcwd(), "l1_record")


def _env_meta() -> dict:
    raw = os.environ.get("L1_RECORD_META", "")
    if not raw:
        return {}
    try:
        meta = json.loads(raw)
        return meta if isinstance(meta, dict) else {"L1_RECORD_META": raw}
    except ValueError:
        return {"L1_RECORD_META": raw}


def _np(x) -> np.ndarray:
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class L1RolloutRecorder(DefaultWrapper):
    """DefaultWrapper plus a passive recorder of proprio, effort decomposition, RGB and BDDL literal truth."""

    def __init__(self, env: Environment):
        super().__init__(env=env)
        self._l1_enabled = os.environ.get("L1_RECORD_DISABLE", "").lower() not in ("1", "true", "yes")
        self._l1_record_rgb = os.environ.get("L1_RECORD_RGB", "1").lower() not in ("0", "false", "no")
        self._l1_out = _output_dir()
        self._l1_extra = _env_meta()
        self._l1_robot = env.robots[0]
        cams = get_robot_camera_names(self._l1_robot.name, getattr(env, "_eval_robot_config", {}))
        self._l1_camera_obs = {role: f"{name}::rgb" for role, name in sorted(cams.items())}
        self._l1_files = None
        self._l1_pending = False
        self._l1_nodes, self._l1_n_pred, self._l1_ids_source = [], 0, None
        self._l1_step, self._l1_c61_maxdiff = 0, None
        self._l1_timing = {"env_step_s": 0.0, "recorder_s": 0.0}
        self._l1_counts = {"decomp_fail": 0, "goal_fail": 0, "rgb_missing": 0}
        self._l1_hooked = _install_evaluator_hooks()
        if self._l1_enabled and int(self._l1_robot.n_dof) != rio.N_DOF:
            logger.error(f"L1 recorder: robot has {self._l1_robot.n_dof} DOF, the sidecar layout needs {rio.N_DOF}; off")
            self._l1_enabled = False
        atexit.register(self._l1_atexit)
        logger.info(f"L1 recorder {'on' if self._l1_enabled else 'OFF'} -> {self._l1_out} "
                    f"(cameras {list(self._l1_camera_obs)}, rgb {'on' if self._l1_record_rgb else 'off'})")

    # --- gym API -----------------------------------------------------------------------------------------
    def reset(self):
        if self._l1_files is not None:  # the previous rollout never reached terminated/truncated
            self._l1_guard(self._l1_finalize, False, False, None, "aborted", "reset before the episode ended")
        out = super().reset()
        self._l1_pending = self._l1_enabled
        return out

    def step(self, action, n_render_iterations=1):
        if not self._l1_enabled:
            return super().step(action, n_render_iterations=n_render_iterations)
        t0 = time.perf_counter()
        if self._l1_pending:
            self._l1_pending = False
            self._l1_guard(self._l1_open)  # literal truth BEFORE the first step, as TaskMetric.reset snapshots it
        t1 = time.perf_counter()
        obs, reward, terminated, truncated, info = super().step(action, n_render_iterations=n_render_iterations)
        t2 = time.perf_counter()
        if self._l1_files is not None:
            self._l1_timing["env_step_s"] += t2 - t1
            self._l1_timing["recorder_s"] += t1 - t0
            self._l1_guard(self._l1_record, obs, action, terminated, truncated, info)
            self._l1_timing["recorder_s"] += time.perf_counter() - t2  # after the final step: not in its meta
        return obs, reward, terminated, truncated, info

    # --- recording ---------------------------------------------------------------------------------------
    def _l1_guard(self, fn, *args):
        try:
            return fn(*args)
        except Exception:
            logger.exception("L1 recorder failed; recording is off for the rest of this process")
            self._l1_enabled = False
            self._l1_pending = False
            files, self._l1_files = self._l1_files, None
            if files is not None:
                try:
                    files.finalize(self._l1_step, False, False, None, status="recorder_error",
                                   note=repr(sys.exc_info()[1]), extra=self._l1_run_stats())
                except Exception:
                    logger.exception("L1 recorder: could not close the files of the failed rollout")
            return None

    def _l1_open(self):
        task = self.env.task
        atoms, sizes, n_total = rio.goal_atoms(task.ground_goal_state_options)
        self._l1_nodes = [nd for *_, nd in atoms]
        self._l1_n_pred = len(task.activity_goal_conditions)
        instance_id = _IDS["instance_id"]
        rollout_id = _IDS["rollout_id"] if _IDS["rollout_id"] is not None else _IDS["counter"]
        self._l1_ids_source = _IDS["source"] if _IDS["rollout_id"] is not None else "counter"
        if instance_id is None:
            instance_id, self._l1_ids_source = 999, "unknown"
        self._l1_step = 0
        self._l1_timing = {"env_step_s": 0.0, "recorder_s": 0.0}
        self._l1_counts = {"decomp_fail": 0, "goal_fail": 0, "rgb_missing": 0}
        self._l1_c61_maxdiff = None
        self._l1_files = rio.RolloutFiles(
            self._l1_out, TASK_NAMES_TO_INDICES[task.activity_name], task.activity_name, instance_id, rollout_id,
            self._l1_robot.name, int(self._l1_robot.action_dim), [a[2] for a in atoms], [a[0] for a in atoms], sizes,
            n_total, self._l1_n_pred, list(self._l1_camera_obs) if self._l1_record_rgb else [],
            extra_meta={**self._l1_extra, "wrapper": type(self).__name__, "ids_source": self._l1_ids_source,
                        "output_dir": self._l1_out, "pid": os.getpid()},
            bddl_attrs={"condition_strs": json.dumps([str(c) for c in task.activity_goal_conditions]),
                        "natural_language_goal_conditions": json.dumps(
                            list(getattr(task, "activity_natural_language_goal_conditions", None) or []))},
        )
        self._l1_files.set_initial(self._l1_literals())
        logger.info(f"L1 recorder: rollout {self._l1_files.base} ({len(atoms)} literals in {len(sizes)} of "
                    f"{n_total} options, ids from {self._l1_ids_source}) -> {self._l1_out}")

    def _l1_literals(self) -> np.ndarray:
        ev = self.env.task._evaluate_predicate
        return np.fromiter((bool(nd.evaluate(ev)) for nd in self._l1_nodes), dtype=np.uint8, count=len(self._l1_nodes))

    def _l1_goal_status(self) -> np.ndarray:
        row = np.zeros(self._l1_n_pred, np.uint8)
        try:
            task = self.env.task
            _, sat = task.compiled_task.check_goal(task._evaluate_predicate)
            row[np.asarray(sat["satisfied"], dtype=int)] = 1
        except Exception:
            self._l1_counts["goal_fail"] += 1
        return row

    def _l1_proprio(self) -> np.ndarray:
        """The demo sidecar's 229 columns, from the same calls ``replay_torque_fixed.py`` made."""
        robot = self._l1_robot
        d = robot._get_proprioception_dict()
        parts = [_np(d[k]).reshape(-1) for k in C61_KEYS] + [_np(d[k]).reshape(-1) for k in rio.JOINT_KEYS]
        try:
            av = robot._articulation_view
            dec = [_np(x).reshape(-1) for x in (av.get_applied_joint_efforts(), av.get_generalized_gravity_forces(),
                                                av.get_coriolis_and_centrifugal_forces())]
            if any(x.size != rio.N_DOF for x in dec):
                raise ValueError(f"decomposition sizes {[x.size for x in dec]}")
        except Exception:
            self._l1_counts["decomp_fail"] += 1
            dec = [np.zeros(rio.N_DOF, np.float32)] * 3
        return np.concatenate(parts + dec).astype(np.float32)

    def _l1_record(self, obs, action, terminated, truncated, info):
        k = self._l1_step
        vec = self._l1_proprio()
        self._l1_files.add_step(vec, _np(action), time.time())
        end = bool(terminated or truncated)
        flat = None
        if k == 0 or (self._l1_record_rgb and k % rio.RGB_STRIDE == 0):
            name = self._l1_robot.name
            flat = flatten_obs_dict(obs[name], parent_key=name) if isinstance(obs, dict) and name in obs else {}
        if k == 0 and f"{self._l1_robot.name}::proprio" in flat:  # our challenge columns == what the policy saw
            seen = _np(flat[f"{self._l1_robot.name}::proprio"]).reshape(-1)
            if seen.size == rio.CHALLENGE_DIM:
                self._l1_c61_maxdiff = float(np.abs(seen - vec[: rio.CHALLENGE_DIM]).max())
        if self._l1_record_rgb and k % rio.RGB_STRIDE == 0:
            if all(key in flat for key in self._l1_camera_obs.values()):
                self._l1_files.add_rgb(k, {role: _np(flat[key]) for role, key in self._l1_camera_obs.items()})
            else:
                self._l1_counts["rgb_missing"] += 1
        if k % rio.BDDL_STRIDE == 0 or end:
            self._l1_files.add_atoms(k, self._l1_literals(), self._l1_goal_status())
            self._l1_files.flush()
        self._l1_step = k + 1
        if end:
            done = info.get("done", {}) if isinstance(info, dict) else {}
            success = bool(done["success"]) if "success" in done else None
            self._l1_finalize(terminated, truncated, success, "ok", "")

    def _l1_run_stats(self) -> dict:
        n = max(self._l1_step, 1)
        return {"env_step_ms_mean": 1e3 * self._l1_timing["env_step_s"] / n,
                "recorder_ms_mean": 1e3 * self._l1_timing["recorder_s"] / n,
                "c61_first_step_maxdiff": self._l1_c61_maxdiff, **self._l1_counts}

    def _l1_finalize(self, terminated, truncated, success, status, note):
        files, self._l1_files = self._l1_files, None
        if files is None:
            return
        files.finalize(self._l1_step, bool(terminated), bool(truncated), success, status, note, self._l1_run_stats())
        _IDS["counter"] += 1
        _IDS["rollout_id"] = None
        logger.info(f"L1 recorder: closed {files.base} ({status}, {self._l1_step} steps, "
                    f"q {files.last_q}, success {success})")

    def _l1_atexit(self):
        if self._l1_files is not None:
            try:
                self._l1_finalize(False, False, None, "process_exit", "the process exited mid-rollout")
            except Exception:
                pass
