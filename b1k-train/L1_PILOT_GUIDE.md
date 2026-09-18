# L1 pilot rollouts — guide for Allen

> **Status: ready.** The smoke rollout passed on 2026-09-18: task 0,
> instance 301 on GPU 1 -- 1,511 steps, success, q_score 1.0, against 1,794 steps and the same outcome in your
> `rlc_ckpt2` run without the recorder. Recording cost 3.6 ms per 94 ms simulation step. You can start the
> pilot whenever a GPU is free. This guide is updated in place if anything below changes.

## What this is

Eduardo's stage-classifier sprint needs Ilia's rollouts as a **test set**. While his policy runs, we record
extra signals: robot sensor channels, which BDDL goal conditions are true over time, the camera frames at
3 Hz, and Ilia's own System-2 stage predictions.

**The policy sees exactly what it sees today** — same cameras, same resolution, same timing. We never
train on, tune on, or pick checkpoints with these rollouts.

## What we add to your workspace

New files only. We don't edit any existing file, we stay on `main`, and we commit locally — you push when
you're happy with them. Before each change we check `git status` so we never work on the same files at the
same time.

| new file | what it does |
|---|---|
| `BEHAVIOR-1K/OmniGibson/omnigibson/eval/wrappers/l1_rollout_recorder.py` | the recorder (an env wrapper). Not exported from `wrappers/__init__.py`; it's loaded by its full path |
| `BEHAVIOR-1K/OmniGibson/omnigibson/eval/wrappers/l1_record_io.py` | the file formats the recorder writes (no simulator imports) |
| `b1k-train/serve_ilia_logged.py` | `serve_ilia.py` unchanged, plus a log of Ilia's stage prediction on every policy call |
| `b1k-train/l1_pilot_rollouts.sh` | one command for the whole pilot: server, sim container, sweep |
| `b1k-train/L1_PILOT_GUIDE.md` | this guide |

**The `b1k-sim` image is not rebuilt.** The runner mounts the recorder file into the container at launch.

## The pilot

| | |
|---|---|
| policy | Ilia checkpoint 2 — `~/evaluation/behavior_checkpoints/ilia/checkpoint_2` |
| tasks | 0 `turning_on_radio` · 1 `picking_up_trash` · 7 `picking_up_toys` · 8 `rearranging_kitchen_furniture` · 9 `putting_up_Christmas_decorations_inside` · 12 `preparing_lunch_box` · 16 `moving_boxes_to_storage` · 17 `bringing_water` · 18 `tidying_bedroom` · 20 `sorting_vegetables` |
| instances | `public_test` 0–1 (instances 301 and 302), one rollout each — **20 rollouts** |
| output | `~/evaluation/eval_runs/l1_pilot/` |
| expected time | 1–3 days, depending on how busy the GPU is |

These are the same tasks and instances as `eval_runs/rlc_ckpt2`. That's deliberate: those 20 runs without
the recorder let us check that recording doesn't change the policy's outcomes.

## Before you start

1. You have Eduardo's **"smoke passed"**.
2. `nvidia-smi`: pick a GPU with **at least 26 GB free that nobody else is using** (sim ≈ 16 GB, server
   ≈ 7.5 GB). If only a card someone else is using has room, ask Eduardo before going ahead. The runner
   checks this itself and refuses to start below it.
3. `df -h /`: **at least 20 GB free**. The runner refuses to start below that. The recordings are small --
   about 5 MB per 1,000 steps, so roughly 1.5 GB for the whole pilot -- but the videos are not.
4. `cd ~/evaluation && git status`: nothing of yours in progress in the files listed above.

## Run it

```bash
cd ~/evaluation/b1k-train
bash l1_pilot_rollouts.sh --gpu <N> --dry-run   # prints every command it would run, starts nothing
bash l1_pilot_rollouts.sh --gpu <N>             # starts the server and the sweep, detached
```

One card carries both by default. To split them: `--server-gpu <A> --sim-gpu <B>`.

What the runner does, in order:

1. Starts `serve_ilia_logged.py` on port **8010** (checkpoint 2, JAX preallocation off and its memory capped
   at 12% of the card, which is about 7.5 GB in practice), with its pid in `eval_runs/l1_pilot/server.pid`.
   The server takes the norm stats and tokenizer that ship inside the checkpoint: `pi_behavior_b1k_fast` now
   names your 2026 training dataset, while Ilia's checkpoints carry theirs under `IliaLarchenko/behavior_224_rgb`,
   so it resolves the id from the checkpoint and logs which one it used. Nothing else about the config changes.
2. Waits for `http://127.0.0.1:8010/healthz` to answer `200` (about a minute).
3. Starts the sim container **non-root** — uid 1011 with the `sim_eval_kit` overlays, the same setup as the
   `sim` service in your `docker-compose.yml` — named `l1-pilot-sim`, with `--network host` and the recorder
   mounted.
4. Inside it, runs `eval_rollout.py` on the 10 tasks above, with the recorder as `--env-wrapper`,
   `--min-free-gb 20`, `--write-video` and a 12-hour-per-task timeout (`picking_up_toys` runs to its 28,336-step
   limit on both instances, about 7.5 hours).
5. Stops its own server when the sweep ends.

## Watch it

```bash
bash l1_pilot_rollouts.sh --status        # tasks done and running, disk, GPU, last log lines
tail -f ~/evaluation/eval_runs/l1_pilot/logs/driver.log
```

Healthy looks like: every 1–3 hours a task directory appears with `json/`, `videos/` and `l1_record/` in
it, and `stage_logs/` keeps growing. The task log also shows `L1 recorder: rollout <name> ...` when a rollout
starts and `closed ...` when it ends.

Running the sim non-root makes it write its caches into `sim_eval_kit/` (`.nv/`, `.nvidia-omniverse/`,
`.triton/`, and two files under `isaacsim_apps/`). They show up as untracked in `git status`; they are caches,
so please just leave them there.

## Stop it, resume it

```bash
bash l1_pilot_rollouts.sh --stop          # stops l1-pilot-sim and our server (by pid file) — nothing else
bash l1_pilot_rollouts.sh --gpu <N>       # resumes: finished tasks are skipped
```

Please never stop other people's containers or processes, and never stop anything by name pattern
(`pkill -f …`) — the shared account runs other people's work too.

## If something goes wrong

| you see | do |
|---|---|
| a task log shows an error | leave the outputs where they are, run `--stop`, send Eduardo the task log path |
| `STOPPING: … N tasks not started` | not a crash: the disk guard tripped. Tell Eduardo; we'll move data off the machine |
| someone else starts using the GPU | `--stop`, wait, then resume with the same command |
| the server isn't healthy after 10 minutes | look at `eval_runs/l1_pilot/logs/server.log`; don't retry in a loop — tell Eduardo |

## Please don't

- Use these rollouts to train, tune, or choose checkpoints for our classifier. They are its test set.
- Delete or move anything under `eval_runs/l1_pilot/`. We copy it to idlab1 and clean up ourselves after a
  checksum match.
- Run a second sweep against the same server. Ilia's wrapper is stateful and resets when the task changes;
  two sweeps would corrupt each other.
- Run the sim container as root (the old recipe in `ILIA_EVAL.md` §5). The runner already uses the non-root
  setup.

## What happens next

We pull the recordings to idlab1, extract features, and freeze our analysis plan before looking at any
result. If the pilot works, the next batch — the 12 tasks where Ilia's policy most often makes partial
progress, 4 instances each — comes to you with an updated guide. Some of those tasks need checkpoints 1 or 3
copied to the monster first (~12 GB each).

Questions: Eduardo.
