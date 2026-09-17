# Running the RLC (1st place) checkpoint — verified runbook

How to evaluate the winning 2025 BEHAVIOR Challenge policy (Robot Learning
Collective / Ilia Larchenko, 26% q-score) inside our 2026 fork, and get numbers
comparable with our own models.

Everything below was **run end to end on 2026-08-13**, not written from the
upstream README. Where their documented command does not work against our fork,
the working one is given and the reason explained.

- Code: <https://github.com/IliaLarchenko/behavior-1k-solution>
- Weights: <https://huggingface.co/IliaLarchenko/behavior_submission>
- Report: [arXiv 2512.06951](https://arxiv.org/abs/2512.06951)

> **Paths moved 2026-09-17:** the checkpoint is now `~/evaluation/behavior_checkpoints/ilia/checkpoint_2` (was `~/models_ilia/checkpoint_2`) and run outputs are under `~/evaluation/eval_runs/` (was `~/eval_runs/`). `scripts/wait_for_gpu_and_rollout.sh` automates §4–§5 with the new paths; the commands below still show the old ones.

---

## 0. Where everything already lives (shared `b1k-challenge` account)

All of it is under the shared account, so Charles and Allen can use it directly.

**`b1k-challenge@140.109.21.51` (idlab_server1 / "monster")**

| path | what |
|---|---|
| `~/evaluation/behavior-1k-solution/` | their repo + our `serve_ilia.py`, `eval_rollout.py`, this doc |
| `~/evaluation/behavior-1k-solution/task_checkpoint_mapping.json` | task→checkpoint map, paths already pointed at `~/models_ilia` |
| `~/models_ilia/checkpoint_2/` | RLC checkpoint 2 (12.6 GB) — **only ckpt 2 is here so far** |
| `~/evaluation/b1k-baselines/baselines/openpi/.venv/` | the Python env that runs the server |
| `~/evaluation/BEHAVIOR-1K/datasets/` | sim assets (mounted as `/data`) |
| `~/eval_runs/rlc_ckpt2/` | sweep outputs: `<task>/json/*.json`, `<task>/videos/*.mp4`, `logs/` |
| `~/ilia_serve.log`, `~/ilia_serve.pid` | policy server log / pid |
| `~/eval_sweep.log` | sweep driver log |
| `~/b1k-analysis/` | the demo-side analysis scripts + results (see §8) |

**`b1k-challenge@172.25.166.50` (idlab1)** — all four checkpoints:
`/mnt/train-data-1-hdd/b1k-challenge/checkpoints/Ilia/checkpoint_{1..4}`

Transfers: `b1k-challenge@idlab1 → b1k-challenge@monster` works (~8.8 MB/s;
12.6 GB took 23 min). The reverse does not, and `ecappiell@idlab1` cannot reach
the `b1k-challenge` account on the monster at all.

```bash
ssh b1k-challenge@172.25.166.50 \
  'rsync -a --info=progress2 --partial \
     /mnt/train-data-1-hdd/b1k-challenge/checkpoints/Ilia/checkpoint_3 \
     b1k-challenge@140.109.21.51:~/models_ilia/'
```

⚠️ **Check `df -h /` first.** The monster root was at 98% / ~41 GB free and
drifting down from other users. Each checkpoint is ~12 GB.

---

## 1. Why their code is required

This is not stock Pi0.5 with different weights. The parts that change inference:

- **No language model** — 50 trainable task embeddings. The model is
  conditioned on an integer `task_id`; prompt plumbing is unused.
- **System 2 stage predictor** — the VLM predicts the current task stage (5–15
  per task, `TASK_NUM_STAGES`) as an auxiliary head. The wrapper votes over a
  3-prediction history before advancing, then feeds the stage **back in** as
  `subtask_state`. Stage is model *input*, not just output.
- **Rolling soft inpainting** — predict 30 actions, execute 26, keep 4 to
  condition the next prediction.
- **Cubic compression** — 26 actions executed in 20 steps (1.3×), base velocity
  rescaled by the factor, auto-disabled when the gripper command is changing.
- **Correction rules**, e.g. open the gripper after a failed grasp.
- **Four task-specific checkpoints**, switched on `task_id`.

All of these were observed working in our run (§5).

Consequence: the wrapper is **stateful across steps**, so an episode boundary
must reset it. The server resets on client reconnect and the wrapper resets when
`task_id` changes — one eval process per task satisfies both, which is what
`eval_rollout.py` does.

## 2. Task-index compatibility — verified, do not "fix" it

Their task embeddings are indexed 0–49. Their 50 training tasks are **exactly**
the tasks with `Task ID` 0–49 in
`2026-challenge-task-instances/metadata/B100_task_misc.csv` — checked as sets,
no difference either way. That CSV is the source of our
`eval_utils.TASK_NAMES_TO_INDICES`, and `evaluator.py:349` sends
`obs["task_id"]` from it.

Their embeddings, their `task_checkpoint_mapping.json`, and our evaluator all
speak the same index. **No remapping is needed; adding one would silently
mis-condition the model.**

Their `src/b1k/training/data_loader.py` contains a 50-name list in a *different*
order — that is a dataset filter, not an index. Do not read task ids off it.

| ckpt | tasks |
|---|---|
| 1 | 2, 3, 5, 6, 10, 11, 13, 14, 15, 19, 23, 24, 25, 28, 29, 34, 42, 44, 47, 48 |
| 2 | 0, 1, 7, 8, 9, 12, 16, 17, 18, 20, 21, 22, 26, 30, 43, 45 |
| 3 | 4, 27, 31, 32, 33, 35, 36, 37, 38, 39, 41, 46, 49 |
| 4 | 40 |

## 3. The 2025 → 2026 API gap

Upstream renamed `omnigibson/learning/` to `omnigibson/eval/`. Three of their
serving modules import the old path (`scripts/serve_b1k.py`,
`src/b1k/policies/b1k_policy.py`, `src/b1k/shared/eval_b1k_wrapper.py`).

`serve_ilia.py` aliases `omnigibson.learning → omnigibson.eval` in `sys.modules`
and stubs the one unused `datas` symbol, then calls their `serve_b1k.main`
untouched. Their tree stays byte-identical, so a `git pull` cannot drop a patch
and no behavioural difference can be blamed on our edits.

**No `setup_remote.sh` and no submodule checkout are needed.** Verified: with the
shim, all of `b1k.{shared.eval_b1k_wrapper, policies.b1k_policy,
policies.checkpoint_switcher, models.pi_behavior_config, training.config}`
import, and `get_config("pi_behavior_b1k_fast")` returns a `PiBehaviorConfig`,
using the openpi venv already at `~/evaluation/b1k-baselines/baselines/openpi/.venv`.

## 4. Start the policy server

```bash
ssh b1k-challenge@140.109.21.51
cd ~/evaluation/behavior-1k-solution

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
nohup ~/evaluation/b1k-baselines/baselines/openpi/.venv/bin/python serve_ilia.py \
    --solution-repo . \
    --port 8010 \
    policy:checkpoint \
      --policy.config pi_behavior_b1k_fast \
      --policy.dir /home/b1k-challenge/models_ilia/checkpoint_2 \
    > ~/ilia_serve.log 2>&1 & echo $! > ~/ilia_serve.pid
```

Check it: `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8010/healthz`
→ `200`. Startup takes ~50 s (checkpoint restore ~6 s of it).

Four things that cost time to discover:

1. **Argument order is significant.** `policy:checkpoint` is a *tyro subcommand*.
   Top-level flags (`--port`, `--task-checkpoint-mapping`) go **before** it;
   `--policy.*` go **after**. Any other order fails with "Unrecognized or
   misplaced options".
2. **Always set the JAX memory flags.** JAX preallocates ~75% of the device and
   will evict other people's jobs. Their script sets 0.5 via `setdefault`, so an
   explicit env var still wins.
3. **Port 8000 is taken on the monster** by another user's process. Use 8010.
4. **`--task-checkpoint-mapping` requires all four checkpoints on disk** — the
   switcher validates all 50 tasks are mapped *and* loads from the mapped paths.
   With only ckpt 2 present, omit it and restrict the sweep to ckpt 2's 16 tasks.

Multi-checkpoint mode (once all four are local) adds
`--task-checkpoint-mapping ./task_checkpoint_mapping.json` before
`policy:checkpoint`. Only one checkpoint is resident at a time; a switch costs an
unload + `jax.clear_caches()` + reload.

Stop it with the recorded pid — never by pattern:

```bash
P=$(cat ~/ilia_serve.pid); [ -d /proc/$P ] && kill $P
```

## 5. Run the sweep

The sim side uses the **existing** `b1k-sim:latest` container — nothing to build.

```bash
docker run --rm --runtime=nvidia --network host --name b1k-eval-rlc \
  -e OMNIGIBSON_HEADLESS=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /home/b1k-challenge/evaluation/BEHAVIOR-1K/datasets:/data \
  -v /home/b1k-challenge/eval_runs:/scratch \
  b1k-sim:latest /opt/conda/envs/behavior/bin/python -u /scratch/eval_rollout.py \
    --tasks 0,1,7,8,9,12,16,17,18,20,21,22,26,30,43,45 \
    --mode public_test --instances 0-1 \
    --host 127.0.0.1 --port 8010 \
    --output-dir /scratch/rlc_ckpt2 \
    --robot-config /behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml \
    --write-video --min-free-gb 25 \
    --submission-out /scratch/rlc_ckpt2/submission.json \
    --team "RLC-checkpoint2"
```

- `--network host` lets the container reach the policy server on the host.
- Put the server and the sim on the **same** GPU only if it has room; here both
  are on GPU 0 because the replay fleet owns GPU 1.
- `--resume` is on by default — an interrupted sweep restarts cheaply.
- `--min-free-gb` stops cleanly before starting a task if the shared disk is
  running out, rather than being the job that fills it.
- `--policy local` runs a zero-action policy for plumbing smoke tests.
- `--dry-run` prints the commands and exits.

**Output ownership — handled automatically; do not use `--user`.**

The container runs as root, so anything it writes lands `root:root` and nobody
on a shared account can clean it up. `eval_rollout.py` therefore chowns each
task's output back to the account that owns the mounted run directory, after
every task (so an interrupted sweep still leaves usable files). No flags needed.

**`--user $(id -u):$(id -g)` does not work here** — this was tried and it fails,
in three stages, each masking the next:

1. `KeyError: getpwuid(): uid not found` — fixed by
   `-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro`.
2. `cannot cache function '_quat_multiply': no locator available` — fixed by
   `-e NUMBA_CACHE_DIR=…`, `-e OMNIGIBSON_APPDATA_PATH=…`, `-e HOME=…`.
3. **Fatal:** `PermissionError: '/opt/conda/envs/behavior/lib/python3.11/
   site-packages/isaacsim/apps/omnigibson_5_1_0.kit'` — Isaac writes *into its
   own installation* at startup, which a non-root user cannot do. There is no
   flag that fixes this short of making site-packages writable.

The first two stages pass far enough to look like success (`import omnigibson`
works, files land correctly owned), which is exactly why this needs a full
rollout to falsify rather than a smoke test.

To repair output from an older run made before this was added, use the
container's own root context — the host account has no sudo:

```bash
docker run --rm --entrypoint /bin/chown \
  -v /home/b1k-challenge/eval_runs:/scratch b1k-sim:latest \
  -R "$(id -u):$(id -g)" /scratch
```

## 6. Which split you can actually run

`asset_utils.get_task_instance_path` maps modes to directories under
`2026-challenge-task-instances`:

| mode | directory | instances | present? |
|---|---|---|---|
| `train` | `scenes/` | 0–299 | yes |
| `public_test` | `scene_test/public/` | 301–320 | yes |
| `hidden_test` | `scene_test/private/` | 321–340 | **no** |

The private split is held by the organisers, so **`--mode hidden_test` cannot
run locally** — it fails with "Could not find 2026 hidden_test task instance
321". The leaderboard's headline numbers are on that split, so a local run is
never bit-comparable to them. It *is* comparable between models, provided each
model is evaluated on the same split and instances. Default is `public_test`.

## 7. Results, and reading them honestly

`--submission-out` writes the **same schema** as
`docs/challenge_submissions/*.json`, so a run drops straight into
`challenge50_extract.py` / `challenge50_report.py` beside the published Comet
and RLC entries with no conversion.

Re-aggregate at any time without re-running:

```bash
python eval_rollout.py --aggregate-only --output-dir ~/eval_runs/rlc_ckpt2 \
    --mode public_test --instances 0-1 \
    --submission-out ~/eval_runs/rlc_ckpt2/submission.json --team "RLC-checkpoint2"
```

Two things that decide whether the number is honest:

- **Per-task means divide by the expected rollout count, not by how many
  finished** — same as the official scorer. A partial run therefore scores low
  rather than scoring high on an easy subset. This is exactly why 31 of Comet's
  *public* per-task scores are zeros produced by absence rather than failure.
- A `_provenance` block records `tasks_present`, `missing_rollouts` and a
  `complete` flag, and the CLI warns when short. **Do not quote an incomplete
  run against the leaderboard.**

**Never trust the exit code.** Isaac returns 0 even after an unhandled
exception — a task that died on `load_task_instance` still exited 0 with zero
rollout files. `eval_rollout.py` therefore judges success on *rollout files
produced* and prints the causing exception; anything else reports "50/50 ok" and
then aggregates an empty submission.

### First verified result

`turning_on_radio` (task 0), public_test, ckpt 2:

| instance | q | success | steps | norm. time | base | left | right |
|---|---|---|---|---|---|---|---|
| 301 | 1.000 | yes | 1794 | 1.198 | 2.07 | 0.63 | 0.67 |
| 302 | 0.000 | no | 3225 | 0.667 | 0.87 | 0.29 | 0.33 |

Server-side, all of the inference machinery was observed live: task change
detection, stage advance *and* regression via the voting logic, inpainting on
every prediction, 26→20 compression, and the gripper-variation guard disabling
compression. Note the policy is **not deterministic** — a repeat of instance 301
gave the same q but 2130 vs 1794 steps.

### Throughput — measure, don't assume

**1.28 steps/s** while sharing the box with the 16-worker replay fleet (load avg
~187), plus **~8.4 min Isaac boot per task** (amortised across that task's
instances). That is ~5× slower than an idle-box assumption of 6 steps/s.

| scope | sharing (1.28 steps/s) | fleet idle (6 steps/s) |
|---|---|---|
| 1 rollout × 50 tasks | ~194 h | ~41 h |
| 10 instances × 50 tasks | ~1940 h | ~414 h |

`turning_on_radio` is the shortest task (~1800–3200 steps) against a
~16,800-step median, so do not extrapolate the whole sweep from it.

## 8. The demo-side analysis (separate rig)

`~/b1k-analysis/` on the monster holds the scripts and results for the
demonstration-side work, also under `b1k-challenge`:

- `low_dim_layout.py` — recovers the `task::low_dim` column map (per-file: the
  order comes from a Python `set`, so it differs between recordings) and gives
  ground-truth grasp flags.
- `demo_grasp_index.py`, `grasp_report.py`, `grasp_out/` — grasp events, base
  approach geometry, and the per-task reference distribution.
- `lerobot_demo_source.py`, `offpolicy_agreement.py`, `serve_b1k_offpolicy.py` —
  off-policy action agreement (feed demo frames to a policy, compare to the
  human) using `behavior-1k/2026-challenge-demos`, which has RGB, camera poses
  and official skill segmentation.
- `REPORT.md`, `master_table.csv` — the 50-task difficulty study.

These were originally produced under `ecappiell@idlab1` and copied here; the
originals remain at `ecappiell@idlab1:~/b1k-analysis/`.
