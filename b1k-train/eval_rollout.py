"""Run the 50-task challenge rollout evaluation against a served policy.

One command to reproduce a leaderboard number for any websocket policy -- ours
or a published one -- and emit it in the *same JSON schema* as
``docs/challenge_submissions/*.json``, so the result drops straight into the
existing analysis (``challenge50_extract.py``, ``challenge50_report.py``)
alongside Comet and RLC without any conversion step.

``omnigibson.eval.eval`` already evaluates **one** task per process. This script
is the layer above it: task selection, checkpoint-aware ordering, resume,
per-task isolation, and aggregation.

Why a subprocess per task rather than one long-lived process:

* Isaac has to tear down and rebuild the scene between tasks anyway, so nothing
  is saved by keeping one process alive.
* A crash in task 23 must not lose tasks 0-22. Each task is isolated and its
  log kept; the sweep continues and reports what failed.
* The policy server keys its state on ``task_id``. A fresh client process
  reconnects and sends the new id, which is exactly what triggers a clean
  ``reset()`` (and, for multi-checkpoint servers, a checkpoint switch).

**Checkpoint-aware ordering matters a lot.** A server like the RLC one holds a
single checkpoint in GPU memory and reloads when the task maps to a different
one. Sweeping tasks in numeric order makes it thrash: their mapping alternates
between checkpoints almost every task, so 0..49 costs ~40 reloads of a 12 GB
checkpoint. Pass ``--checkpoint-mapping`` and tasks are grouped so each
checkpoint is loaded once -- 4 loads instead of ~40.

## Running against the RLC (1st place) policy

Their model is a modified Pi0.5 -- task embeddings instead of text, a System 2
stage predictor whose output is fed back as input, rolling soft inpainting, and
correction rules -- so it *must* be served by their own code. See
``ILIA_EVAL.md`` next to this file for the full setup, the patches their repo
needs against our 2026 fork, and the checkpoint mapping.

Verified compatibility: their task embeddings are indexed 0-49, and their 50
training tasks are exactly the tasks with ``Task ID`` 0-49 in
``B100_task_misc.csv`` -- the same table behind our ``TASK_NAMES_TO_INDICES``.
So the ``task_id`` our evaluator sends (``evaluator.py:349``) is directly
compatible with both their embeddings and their checkpoint mapping. No
remapping is needed, and none should be introduced.

## Which split you can actually run

``asset_utils.get_task_instance_path`` maps modes to directories under
``2026-challenge-task-instances``:

    train        scenes/               instances 0-299     present
    public_test  scene_test/public/    instances 301-320   present
    hidden_test  scene_test/private/   instances 321-340   NOT DISTRIBUTED

The private split is held by the organisers, so **`--mode hidden_test` cannot
run locally** -- it fails with "Could not find 2026 hidden_test task instance
321". The leaderboard's headline numbers are on that split, so a local run is
never bit-comparable to them; it is comparable *between models*, provided every
model is evaluated on the same split and instances. Default here is
``public_test``, which is the closest available analogue (and the split several
published submissions also report).

## Examples

Smoke test with the zero-action policy (no server, no GPU needed for the
policy) on two short tasks::

    python eval_rollout.py --tasks 0,40 --instances 0 --policy local \\
        --output-dir /tmp/smoke

Full hidden-set sweep against a served policy, matching the leaderboard
protocol (50 tasks x 10 instances)::

    python eval_rollout.py --tasks all --mode public_test --instances 0-9 \\
        --host 127.0.0.1 --port 8010 \\
        --checkpoint-mapping ~/evaluation/behavior-1k-solution/task_checkpoint_mapping.json \\
        --output-dir ~/eval_runs/rlc_hidden \\
        --submission-out ~/eval_runs/rlc_hidden/submission.json \\
        --team "RLC-reproduction" --track standard

Re-aggregate without re-running anything::

    python eval_rollout.py --aggregate-only --output-dir ~/eval_runs/rlc_hidden \\
        --submission-out ~/eval_runs/rlc_hidden/submission.json --team "RLC-reproduction"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

# Evaluation constants, mirroring omnigibson/eval/utils/score_utils.py. They are
# restated (not imported) so aggregation runs anywhere -- importing omnigibson
# pulls in Isaac and requires OMNIGIBSON_DATA_PATH, which a laptop doing the
# write-up will not have.
EVAL_TIMEOUT_MULTIPLIER = 1.5
DISTANCE_KEYS = ("base", "left", "right")


def _task_table(data_path: str | None) -> dict[str, int]:
    """{task_name: task_id} from B100_task_misc.csv -- the same source as
    ``eval_utils.TASK_NAMES_TO_INDICES``."""
    root = data_path or os.environ.get("OMNIGIBSON_DATA_PATH", "")
    path = os.path.join(root, "2026-challenge-task-instances", "metadata", "B100_task_misc.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"Cannot find {path}.\nSet OMNIGIBSON_DATA_PATH or pass --data-path so task ids can be resolved."
        )
    with open(path, newline="", encoding="utf-8") as f:
        return {row["Task"]: int(row["Task ID"]) for row in csv.DictReader(f)}


def _parse_ints(spec: str, limit: int) -> list[int]:
    """'0-9', '0,3,7', 'all' -> list of ints."""
    if spec.strip().lower() == "all":
        return list(range(limit))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def resolve_tasks(spec: str, table: dict[str, int]) -> list[tuple[int, str]]:
    """Task spec -> ordered [(task_id, task_name)], restricted to the challenge 50."""
    by_id = {v: k for k, v in table.items() if v < 50}
    if spec.strip().lower() == "all":
        ids = sorted(by_id)
    else:
        ids = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if part in table:  # a task name
                ids.append(table[part])
            else:
                ids.extend(_parse_ints(part, 50))
    bad = [i for i in ids if i not in by_id]
    if bad:
        raise SystemExit(f"Task ids outside the challenge 50: {bad}")
    seen, ordered = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            ordered.append((i, by_id[i]))
    return ordered


def order_by_checkpoint(tasks: list[tuple[int, str]], mapping_path: str | None) -> list[tuple[int, str]]:
    """Group tasks so a switching server loads each checkpoint exactly once."""
    if not mapping_path:
        return tasks
    with open(os.path.expanduser(mapping_path)) as f:
        mapping = json.load(f)["checkpoints"]
    task_to_ckpt = {int(t): name for name, info in mapping.items() for t in info["tasks"]}
    missing = [t for t, _ in tasks if t not in task_to_ckpt]
    if missing:
        raise SystemExit(f"--checkpoint-mapping does not cover tasks {missing}")
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for item in tasks:
        groups[task_to_ckpt[item[0]]].append(item)
    ordered = []
    for name in sorted(groups, key=lambda n: -len(groups[n])):  # biggest group first
        ordered.extend(sorted(groups[name]))
    return ordered


def _fix_ownership(path: str) -> None:
    """Hand root-created output back to the account that owns the run directory.

    The sim container runs as root, so everything it writes lands ``root:root``
    and nobody on a shared account can clean it up. Running the container with
    ``--user`` is *not* an alternative: Isaac writes into its own installation
    (``site-packages/isaacsim/apps/omnigibson_5_1_0.kit``) and dies with
    ``PermissionError`` as a non-root user -- verified, not assumed.

    So instead we run as root and give the files back afterwards, to whichever
    account owns the nearest ancestor that root did not create (i.e. the
    directory the host mounted in). No-op when not running as root.
    """
    if os.geteuid() != 0:
        return
    owner = os.path.abspath(path)
    while owner != "/" and os.stat(owner).st_uid == 0:
        owner = os.path.dirname(owner)
    st = os.stat(owner)
    if st.st_uid == 0:
        return  # nothing sensible to hand back to
    for root, dirs, files in os.walk(path):
        for name in (*dirs, *files):
            try:
                os.chown(os.path.join(root, name), st.st_uid, st.st_gid)
            except OSError:
                pass
    try:
        os.chown(path, st.st_uid, st.st_gid)
    except OSError:
        pass


def _free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def _rollout_paths(json_dir: str, task_name: str) -> list[str]:
    if not os.path.isdir(json_dir):
        return []
    return sorted(os.path.join(json_dir, n) for n in os.listdir(json_dir)
                  if n.startswith(f"{task_name}_") and n.endswith(".json"))


def run_task(task_id: int, task_name: str, args, env: dict, expected: int) -> tuple[bool, str]:
    """One `omnigibson.eval.eval` subprocess. Returns (ok, log_path).

    Success is judged on **rollout files produced**, not on the exit code.
    Isaac's shutdown path can return 0 after an unhandled exception -- a task
    that died on `load_task_instance` still exited 0 here -- so trusting the
    return code would let a 50-task sweep report "50/50 ok" and then aggregate
    an empty submission.
    """
    out_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "logs", f"{task_id:02d}_{task_name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [
        sys.executable, "-m", "omnigibson.eval.eval",
        "--task-name", task_name,
        "--mode", args.mode,
        "--policy", args.policy,
        "--instance-indices", *[str(i) for i in args.instance_list],
        "--num-rollouts", str(args.num_rollouts),
        "--output-dir", out_dir,
        "--env-wrapper", args.env_wrapper,
    ]
    if args.policy == "websocket":
        cmd += ["--host", args.host, "--port", str(args.port)]
    if args.robot_config:
        cmd += ["--robot-config", args.robot_config]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    cmd += ["--write-video"] if args.write_video else ["--no-write-video"]
    cmd += ["--headless"] if args.headless else ["--no-headless"]

    if args.dry_run:
        print("  " + " ".join(cmd))
        return True, log_path

    started = time.time()
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                              timeout=args.task_timeout or None)
    _fix_ownership(out_dir)
    mins = (time.time() - started) / 60
    n = len(_rollout_paths(os.path.join(out_dir, "json"), task_name))
    ok = n >= expected and proc.returncode == 0
    status = "ok  " if ok else ("PARTIAL" if n else "FAIL")
    print(f"  {status} rc={proc.returncode} {mins:6.1f} min  {n}/{expected} rollouts  log={log_path}")
    if not ok:
        for line in _tail_error(log_path):
            print(f"       {line}")
    return ok, log_path


def _tail_error(log_path: str, keep: int = 3) -> list[str]:
    """The last exception line(s) from a task log, for an at-a-glance reason."""
    try:
        with open(log_path, errors="replace") as f:
            lines = [ln.rstrip() for ln in f]
    except OSError:
        return []
    hits = [ln.strip() for ln in lines
            if ("Error" in ln or "Exception" in ln) and "omni.kit.app" not in ln]
    return hits[-keep:] if hits else lines[-keep:]


# ------------------------------------------------------------------ aggregation


def _time_score(normalized_time: float) -> float:
    """Leaderboard time score from normalized_time (metrics/task_metric.py).

    time_score = M/(M-1) - 1/((M-1)*nt), clipped to [0, 1]; M = 1.5.
    """
    m = EVAL_TIMEOUT_MULTIPLIER
    if not normalized_time or normalized_time <= 0:
        return 0.0
    return max(0.0, min(1.0, m / (m - 1) - 1.0 / ((m - 1) * normalized_time)))


def _collect(output_dir: str) -> dict[str, dict[str, dict]]:
    """{task_name: {rollout_key: result_dict}} from the per-rollout JSONs."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for task_name in sorted(os.listdir(output_dir)):
        json_dir = os.path.join(output_dir, task_name, "json")
        for path in _rollout_paths(json_dir, task_name):
            try:
                with open(path) as f:
                    res = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            key = f"{res.get('instance_id', 0)}_{res.get('rollout_id', 0)}"
            out[task_name][key] = res
    return out


def build_submission(output_dir: str, team: str, track: str, testset: str,
                     affiliation: str, expected_rollouts: int) -> dict:
    """Assemble the leaderboard-schema submission JSON.

    Per-task means divide by ``expected_rollouts``, not by however many rollouts
    happened to finish. That is what the official scorer does
    (``score_utils.compute_final_q_score`` divides by a fixed count), and it is
    why a partially-complete run scores low rather than scoring high on a
    subset. `missing_rollouts` records the shortfall explicitly so a partial run
    can never be mistaken for a complete one.
    """
    collected = _collect(output_dir)
    per_rollout: dict[str, dict[str, dict[str, float]]] = {
        k: {} for k in ("q_score", "time_score", *(f"{d}_distance_score" for d in DISTANCE_KEYS))
    }
    per_task: dict[str, dict[str, float]] = {
        k: {} for k in ("q_score", "task_sr", "time_score", *(f"{d}_distance_score" for d in DISTANCE_KEYS))
    }
    missing: dict[str, int] = {}

    for task_name, rollouts in sorted(collected.items()):
        sums = defaultdict(float)
        sr = 0.0
        for key, res in rollouts.items():
            q = float(res.get("q_score", {}).get("final", 0.0) or 0.0)
            nt = float(res.get("time", {}).get("normalized_time", 0.0) or 0.0)
            ts = _time_score(nt)
            per_rollout["q_score"].setdefault(task_name, {})[key] = q
            per_rollout["time_score"].setdefault(task_name, {})[key] = ts
            sums["q_score"] += q
            sums["time_score"] += ts
            sr += 1.0 if res.get("success") else 0.0
            nad = res.get("normalized_agent_distance", {}) or {}
            for d in DISTANCE_KEYS:
                v = float(nad.get(d, 0.0) or 0.0)
                v = max(0.0, min(1.0, v))
                per_rollout[f"{d}_distance_score"].setdefault(task_name, {})[key] = v
                sums[f"{d}_distance_score"] += v
        n = max(expected_rollouts, 1)
        for k in per_task:
            per_task[k][task_name] = (sr / n) if k == "task_sr" else (sums[k] / n)
        got = len(rollouts)
        if got < expected_rollouts:
            missing[task_name] = expected_rollouts - got

    n_tasks = max(len(collected), 1)
    overall = {k: sum(v.values()) / n_tasks for k, v in per_task.items()}
    return {
        "team": team,
        "affiliation": affiliation,
        "date": time.strftime("%Y%m%d"),
        "track": track,
        "testset": testset,
        "overall_scores": overall,
        "per_task_scores": per_task,
        "per_rollout_scores": per_rollout,
        "_provenance": {
            "generated_by": "OmniGibson/scripts/learning/eval_rollout.py",
            "output_dir": os.path.abspath(output_dir),
            "tasks_present": len(collected),
            "expected_rollouts_per_task": expected_rollouts,
            "missing_rollouts": missing,
            "complete": len(collected) == 50 and not missing,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="all", help="'all', ids ('0-9', '0,3,7'), or task names")
    # public_test, not hidden_test: the hidden split is not distributed. See the
    # note in the module docstring.
    p.add_argument("--mode", choices=("train", "public_test", "hidden_test"), default="public_test")
    p.add_argument("--instances", default="0-9", help="instance indices within the split, e.g. '0-9'")
    p.add_argument("--num-rollouts", type=int, default=1, help="rollouts per instance")
    p.add_argument("--policy", choices=("websocket", "local"), default="websocket")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--robot-config", default=None)
    p.add_argument("--env-wrapper", default="omnigibson.eval.wrappers.DefaultWrapper")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--write-video", action="store_true")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-path", default=None, help="overrides OMNIGIBSON_DATA_PATH for the task table")
    p.add_argument("--checkpoint-mapping", default=None,
                   help="task_checkpoint_mapping.json; groups tasks so each checkpoint loads once")
    p.add_argument("--task-timeout", type=int, default=0, help="seconds per task, 0 = no limit")
    p.add_argument("--min-free-gb", type=float, default=20.0,
                   help="stop before starting a task if free space is below this (shared box guard)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="skip tasks that already have all expected rollouts (default: on)")
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    p.add_argument("--aggregate-only", action="store_true", help="skip rollouts, just build the submission")
    p.add_argument("--submission-out", default=None)
    p.add_argument("--team", default="ours")
    p.add_argument("--affiliation", default="")
    p.add_argument("--track", default="standard")
    args = p.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    args.instance_list = _parse_ints(args.instances, 20)
    expected = len(args.instance_list) * args.num_rollouts

    if not args.aggregate_only:
        table = _task_table(args.data_path)
        tasks = order_by_checkpoint(resolve_tasks(args.tasks, table), args.checkpoint_mapping)
        print(f"{len(tasks)} tasks x {len(args.instance_list)} instances x {args.num_rollouts} rollouts "
              f"= {len(tasks) * expected} rollouts, mode={args.mode}")
        if args.checkpoint_mapping:
            print(f"ordered by checkpoint group: {[t for t, _ in tasks]}")

        env = dict(os.environ)
        env.setdefault("OMNIGIBSON_HEADLESS", "1" if args.headless else "0")
        failures = []
        for n, (task_id, task_name) in enumerate(tasks, 1):
            done = len(_rollout_paths(os.path.join(args.output_dir, task_name, "json"), task_name))
            if args.resume and done >= expected:
                print(f"[{n}/{len(tasks)}] {task_id:2d} {task_name}: complete ({done}), skipping")
                continue
            free = _free_gb(args.output_dir)
            if free < args.min_free_gb:
                print(f"\nSTOPPING: {free:.1f} GB free < --min-free-gb {args.min_free_gb}. "
                      f"{len(tasks) - n + 1} tasks not started.\n"
                      "This box is shared and the replay fleet writes to the same filesystem; "
                      "filling it would take out both. Free space, then re-run -- --resume picks up here.")
                failures.extend(t for _, t in tasks[n - 1:])
                break
            print(f"[{n}/{len(tasks)}] {task_id:2d} {task_name}  ({free:.0f} GB free)"
                  + (f" (resuming, {done}/{expected} present)" if done else ""))
            try:
                ok, _ = run_task(task_id, task_name, args, env, expected)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT after {args.task_timeout}s")
                ok = False
            if not ok:
                failures.append(task_name)
        if args.dry_run:
            return 0
        print(f"\n{len(tasks) - len(failures)}/{len(tasks)} tasks produced all "
              f"{expected} rollouts" + (f"; incomplete: {failures}" if failures else ""))

    if args.submission_out:
        testset = {"hidden_test": "hidden", "public_test": "public", "train": "train"}[args.mode]
        sub = build_submission(args.output_dir, args.team, args.track, testset, args.affiliation, expected)
        os.makedirs(os.path.dirname(os.path.abspath(args.submission_out)), exist_ok=True)
        with open(args.submission_out, "w") as f:
            json.dump(sub, f, indent=1)
        prov = sub["_provenance"]
        print(f"\nsubmission -> {args.submission_out}")
        print(f"  tasks {prov['tasks_present']}/50, complete={prov['complete']}")
        if prov["missing_rollouts"]:
            print(f"  INCOMPLETE -- missing rollouts: {prov['missing_rollouts']}")
            print("  Per-task means still divide by the expected count, so these scores are")
            print("  depressed by absence, not by failure. Do not compare them to the leaderboard.")
        for k, v in sub["overall_scores"].items():
            print(f"  {k:22s} {v:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
