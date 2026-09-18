"""``serve_ilia.py`` plus a JSONL log of System 2's stage decisions (L1 pilot, see L1_PILOT_GUIDE.md).

Starts exactly the server ``serve_ilia.py`` starts -- same alias shim, same ``serve_b1k.main``, same flags -- after
patching ``B1KPolicyWrapper`` (in this process only) so that every stage decision is appended to
``$L1_STAGE_LOG_DIR/stage_log_<host>_<pid>_<start>.jsonl``, one JSON object per line:

    {"event": "vote", "t": <unix time>, "task_id": 0, "step": 40, "prediction": 3, "argmax": 1,
     "logits": [...], "stage_before": 0, "stage_after": 1, "history": [1, 1]}
    {"event": "correction", ..., "stage_from": 2, "stage_to": 1}   a correction rule moved the stage inside act()
    {"event": "task_change", ..., "task_from": null, "task_to": 0}
    {"event": "reset", ...}                                         the evaluator reset the policy (new rollout)
    {"event": "server_start", ...}

``step`` is the wrapper's ``step_count``: the index, within the rollout, of the env step whose action this call
returns (the evaluator resets the server's policy on every rollout). A vote at step k saw the observation from
before action k, which is the recorder's row k-1 (the initial state when k = 0). ``argmax`` is the raw argmax of
the logits; ``history`` holds the wrapper's votes after it clamps them to the task's last stage.

Nothing about the policy changes: the patches call the original methods and only read the wrapper's state.

It also resolves the checkpoint's **asset id**. ``create_trained_policy`` loads the norm stats and the FAST
tokenizer from ``<checkpoint>/assets/<data config asset id>``, and ``pi_behavior_b1k_fast`` now names the 2026
training dataset (``behavior-1k/2026-challenge-demos``), while Ilia's released checkpoints carry theirs under
``IliaLarchenko/behavior_224_rgb``. With ``--l1-asset-id auto`` (the default) the server uses the one asset
directory the checkpoint actually contains, and says so in the log and in the ``server_start`` event; ``--l1-asset-id
<id>`` forces one and ``--l1-asset-id keep`` leaves the config alone. Only the lookup key changes -- the stats
loaded are the ones that shipped with the checkpoint.

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false L1_STAGE_LOG_DIR=<dir> \\
    <policy venv>/bin/python -P serve_ilia_logged.py --solution-repo . --port 8010 \\
        policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir <checkpoint>
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import importlib.util
import json
import logging
import os
import socket
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_serve_ilia():
    """serve_ilia.py by path: the server runs with ``python -P``, so b1k-train is not on sys.path."""
    spec = importlib.util.spec_from_file_location("serve_ilia", os.path.join(_HERE, "serve_ilia.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return str(x)


class StageLog:
    """Append-only JSONL, one line per event, flushed immediately."""

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        name = f"stage_log_{socket.gethostname()}_{os.getpid()}_{int(time.time())}.jsonl"
        self.path = os.path.join(directory, name)
        self._fh = open(self.path, "a", buffering=1)
        self._lock = threading.Lock()

    def write(self, **record) -> None:
        record = {"t": time.time(), **record}
        line = json.dumps(record, default=_jsonable)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()


def _ids(w) -> dict:
    return {"task_id": w.task_id, "step": int(w.step_count), "prediction": int(w.prediction_count)}


def install_stage_logging(wrapper_cls, log: StageLog) -> None:
    """Patch the class in place. Every patch calls the original and only reads state; logging errors are
    reported once and never reach the policy."""
    if getattr(wrapper_cls, "_l1_stage_logging", False):
        return
    orig_update = wrapper_cls.update_current_stage
    orig_reset = wrapper_cls.reset
    orig_act = wrapper_cls.act
    orig_task = wrapper_cls._handle_task_change
    failed = []

    def safe_write(**record):
        try:
            log.write(**record)
        except Exception:
            if not failed:
                failed.append(True)
                logging.exception("L1 stage log: write failed (further failures are silent)")

    def update_current_stage(self, predicted_subtask_logits):
        before = self.current_stage
        entered = getattr(self, "_l1_stage_at_act", before)
        if before != entered:
            safe_write(event="correction", **_ids(self), stage_from=entered, stage_to=before)
        out = orig_update(self, predicted_subtask_logits)
        self._l1_voted = True
        try:
            arr = np.asarray(predicted_subtask_logits, dtype=np.float64).reshape(-1)
            # the model emits -inf for stages a task does not have; JSON has no infinity, so those are null
            logits = [round(float(v), 4) if np.isfinite(v) else None for v in arr]
            safe_write(event="vote", **_ids(self), argmax=int(arr.argmax()) if arr.size else None,
                       logits=logits, stage_before=before,
                       stage_after=self.current_stage, history=[int(v) for v in self.prediction_history])
        except Exception:
            safe_write(event="vote", **_ids(self), stage_before=before, stage_after=self.current_stage,
                       note="logits not loggable")
        return out

    def act(self, obs):
        self._l1_stage_at_act = self.current_stage
        self._l1_voted = False
        out = orig_act(self, obs)
        if not self._l1_voted and self.current_stage != self._l1_stage_at_act:
            ids = _ids(self)
            ids["step"] -= 1  # act() has already counted this step
            safe_write(event="correction", **ids, stage_from=self._l1_stage_at_act, stage_to=self.current_stage,
                       note="no vote in this call")
        return out

    def _handle_task_change(self, new_task_id):
        old = self.task_id
        out = orig_task(self, new_task_id)
        if old != self.task_id:
            safe_write(event="task_change", **_ids(self), task_from=old, task_to=self.task_id)
        self._l1_stage_at_act = self.current_stage  # a task change resets the stage; that is not a correction
        return out

    def reset(self):
        out = orig_reset(self)
        safe_write(event="reset", **_ids(self))
        return out

    wrapper_cls.update_current_stage = update_current_stage
    wrapper_cls.act = act
    wrapper_cls._handle_task_change = _handle_task_change
    wrapper_cls.reset = reset
    wrapper_cls._l1_stage_logging = True


def checkpoint_asset_ids(checkpoint_dir: str) -> list[str]:
    """The asset ids a checkpoint actually carries: ``assets/<a>/<b>/norm_stats.json`` -> ``a/b``."""
    root = os.path.join(os.path.expanduser(checkpoint_dir), "assets")
    found = glob.glob(os.path.join(root, "*", "*", "norm_stats.json"))
    return sorted(os.path.relpath(os.path.dirname(p), root) for p in found)


def resolve_asset_id(requested: str, checkpoint_dir: str, config_asset_id: str | None) -> str | None:
    """The asset id to serve with, or None to leave the config alone.

    ``auto``: keep the config's id when the checkpoint has it, otherwise take the checkpoint's own -- but only
    when there is exactly one, so an ambiguous checkpoint fails loudly instead of being served with guessed
    normalization.
    """
    if requested == "keep":
        return None
    if requested != "auto":
        return requested
    have = checkpoint_asset_ids(checkpoint_dir)
    if config_asset_id and config_asset_id in have:
        return None
    if len(have) == 1:
        return have[0]
    raise SystemExit(
        f"cannot resolve the asset id: the config asks for {config_asset_id!r} and {checkpoint_dir} carries "
        f"{have or 'none'}. Pass --l1-asset-id <id> (or keep).")


def patch_asset_id(config_module, asset_id: str) -> None:
    """Make every get_config() return a config whose data assets point at `asset_id`."""
    orig = config_module.get_config

    def get_config(name):
        cfg = orig(name)
        data = dataclasses.replace(cfg.data, assets=dataclasses.replace(cfg.data.assets, asset_id=asset_id))
        return dataclasses.replace(cfg, data=data)

    config_module.get_config = get_config


def main() -> int:
    log_dir = os.environ.get("L1_STAGE_LOG_DIR")
    if not log_dir:
        raise SystemExit("L1_STAGE_LOG_DIR is not set")
    serve_ilia = _load_serve_ilia()

    # The same path setup serve_ilia.main() does, so the wrapper class can be patched before it serves.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--solution-repo", default=".")
    pre.add_argument("--l1-asset-id", default="auto", help="auto (default), keep, or an explicit asset id")
    pre.add_argument("--policy.config", dest="policy_config", default=None)
    pre.add_argument("--policy.dir", dest="policy_dir", default=None)
    known, _ = pre.parse_known_args()
    repo = os.path.abspath(os.path.expanduser(known.solution_repo))
    for path in (os.path.join(repo, "src"), os.path.join(repo, "scripts")):
        if not os.path.isdir(path):
            raise SystemExit(f"{path} not found -- is --solution-repo pointing at the solution checkout?")
        if path not in sys.path:
            sys.path.insert(0, path)
    serve_ilia.install_learning_alias()

    from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from b1k.training import config as _config

    asset_id = None
    if known.policy_dir and known.policy_config:
        cfg = _config.get_config(known.policy_config)
        current = cfg.data.assets.asset_id or getattr(cfg.data, "repo_id", None)
        asset_id = resolve_asset_id(known.l1_asset_id, known.policy_dir, current)
        if asset_id:
            patch_asset_id(_config, asset_id)
            logging.info(f"asset id: {current!r} (config) -> {asset_id!r} (in {known.policy_dir})")

    # serve_ilia.main() parses sys.argv again with tyro, which does not know this flag.
    argv, skip = [], False
    for a in sys.argv:
        if skip:
            skip = False
            continue
        if a == "--l1-asset-id":
            skip = True
            continue
        if a.startswith("--l1-asset-id="):
            continue
        argv.append(a)
    sys.argv = argv

    log = StageLog(log_dir)
    install_stage_logging(B1KPolicyWrapper, log)
    log.write(event="server_start", argv=sys.argv, pid=os.getpid(), host=socket.gethostname(),
              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), asset_id=asset_id,
              checkpoint=known.policy_dir, policy_config=known.policy_config)
    logging.info(f"L1 stage log -> {log.path}")
    return serve_ilia.main()  # unchanged from here: parses the same argv and serves


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    sys.exit(main())
