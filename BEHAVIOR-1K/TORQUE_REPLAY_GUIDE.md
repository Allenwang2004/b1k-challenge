# Torque Replay Production Guide (Milestone 1)

**Audience**: Charles (and future us) scaling the physics-on torque replay of the
2026-challenge raw demos to rented H100 nodes. Everything here was validated on
idlab2 (2× RTX 5090) on 2026-07-17/18; every pitfall listed was actually hit.

## What this produces

For each raw teleop demo (HF `behavior-1k/2026-challenge-rawdata`, ~1 TB total,
100 tasks × 200 episodes), a small **sidecar HDF5** with physically-meaningful
force channels, replayed teacher-forced (trajectory divergence vs the demo:
~4e-4 rad — effectively none):

- `data/demo_i/obs/robot_r1::proprio` — **229-dim** per step:
  61 challenge layout + `joint_qpos`(28) + `joint_qvel`(28) +
  `joint_qeffort`(28, PhysX **measured** net joint torque) +
  `joint_qeffort_applied`(28, ≡0 under position drives, kept for effort-mode compat) +
  `joint_gravity`(28, g(q)) + `joint_coriolis`(28, c(q,q̇))
- contact+friction+inertial torque = `measured − gravity − coriolis` residual
  (q̈ via finite-difference of q̇ at 30 Hz; there is **no acceleration API** in
  the stack)
- Join safety: `data`-group attrs `task_id`, `task_name`, `replay_cell`
  (`B-natural-frequency`), `proprio_layout` (JSON name→slice), and `manifest`
  (JSON group→demo_id) — a renamed file can never be mismatched.
- Size: ~2–3 MB per episode (~0.1% of the visual replay). The 1.4 TB visual
  replay is **never duplicated**; sidecars join on `(task_id, demo_id, step)`.

Physical sanity reference (task 0, 20 episodes): base_x/y ~20/17 N·m median,
torso chain 10.7→1.7, shoulders ~3.6, wrists ~0.1, grippers ~0.05, passive
wheels 0. Locomotion plateau ~250 N·m effort-norm, contact spikes ≤~480 N·m.

## Why these exact settings (do not change casually)

- `--preserve-frequencies` (**cell B**) is mandatory: the wrapper's default
  1 kHz micro-stepping produces ~300× inflated snap-transient torques
  (E04 calibration matrix). Natural frequency = physically plausible efforts
  AND still teacher-forced.
- Effort caps are pointless (measured torque is net transmitted, not drive
  output) — don't bother.
- Replays are **CPU-bound**: an RTX 5090 sits at 0–38% GPU util while each
  worker eats ~6 CPU cores (PhysX CPU solver + python). `gm.USE_GPU_DYNAMICS`
  is False by default; a GPU-dynamics variant is under evaluation — do not
  enable it without re-running the effort-equivalence check.

## Sizing rule (per node)

```
workers = min( VRAM_total / 14 GB,  CPU_cores / 6,  (RAM_GB − 10) / 12 )
```
CPU cores are almost always the binding constraint on GPU-dense nodes.
Examples: idlab2 (2×32 GB, 24 cores, 62 GB RAM) → 4 workers (cores-bound, and
at 4 it already touches swap — don't exceed).
A 4×H100 node with 96 vCPU / 512 GB RAM → ~16 workers.
**When renting H100s, prioritize CPU core count per GPU** — a 26-vCPU-per-GPU
shape wastes most of the H100.

## H100 / Zettabyte setup (container path — cross-user safe)

H100 has no RT cores: camera rendering is impossible, but this workload is
camera-less by design. Host driver must NOT be 595.x (Isaac 5.1 crashes;
580.65+ validated). The toolkit injects the HOST driver into containers —
a container cannot fix a bad host driver.

1. **Image**: `b1k-sim` (fork overlay on `stanfordvl/behavior:3.9.0`).
   Registries are blocked on Zettabyte, HF is whitelisted:
   ```bash
   # any machine with the image + normal egress (or take the tarball from
   # /media/.../b1k-artifacts/images/b1k-sim.tar.zst on Eduardo's workstation):
   ./docker/transfer_image_hf.sh save b1k-sim ecappiell/b1k-artifacts
   # on the H100 host:
   ./docker/transfer_image_hf.sh load b1k-sim ecappiell/b1k-artifacts
   ```
2. **Repo** (the image predates the replay scripts — mount a current checkout
   over the baked sources; editable installs resolve because the mount path
   matches):
   ```bash
   git clone git@github.com:EduCappiello/BEHAVIOR-1K.git && cd BEHAVIOR-1K
   git checkout feat/force-replay
   ```
3. **Datasets** on the host, mounted at `/data` (HF direct download):
   ```bash
   mkdir -p datasets && cd datasets
   for z in behavior-1k-assets-3.9.0 omnigibson-robot-assets 2026-challenge-task-instances; do
     curl -LO "https://huggingface.co/datasets/behavior-1k/zipped-datasets/resolve/main/$z.zip"
     unzip -q $z.zip -d ${z%-3.9.0} && rm $z.zip
   done
   # + the decryption key (ask Eduardo) -> datasets/omnigibson.key
   ```
   **All three datasets AND the key are hard requirements.** Missing
   `omnigibson-robot-assets` fails as `Scene must have exactly one robot,
   found 0` — the robot registry is populated by globbing
   `$DATA_PATH/*/models/*/*.yaml`, not by code.
4. **Run** (one container per node; raw demos auto-download per task):
   ```bash
   docker run -d --name b1k-replay --gpus all \
     -v $PWD:/behavior-src \
     -v $PWD/datasets:/data \
     -v $PWD/b1k-data:/b1k-data \
     b1k-sim python /behavior-src/OmniGibson/scripts/learning/batch_replay_torque.py \
       --data_folder /b1k-data --task-id 3 --gpus 0,1,2,3 --workers 4 --chunk-size 20
   docker logs -f b1k-replay
   ```
   `--workers` is per GPU; a 4-GPU/96-core node → `--workers 4` = 16 processes.

## Task sharding across machines

The driver is per-task and fully independent across tasks — shard by task id:

| Machine | Tasks |
|---|---|
| idlab2 (Eduardo) | 0–9 |
| H100 node 1 | 10–32 |
| H100 node 2 | 33–55 |
| H100 node 3 | 56–78 |
| H100 node 4 | 79–99 |

Long tasks dominate (mean raw size varies 7→109 MB/episode across tasks) —
if a node finishes early, take tasks from the busiest neighbor's tail.
Run tasks sequentially per node (`for t in $(seq 10 32); do … --task-id $t; done`);
the driver exits nonzero if any episode failed.

## Resume, failure handling, QC

- **Resume is automatic and safe**: done demo_ids are read from output attrs;
  outputs without attrs are partial (crash mid-chunk) and are auto-deleted and
  redone. Just re-run the same command.
- **Never trust exit codes of Isaac processes** — Kit's shutdown handler can
  exit 0 after a crash. The driver already verifies chunks by manifest
  content; if you script around it, do the same.
- Per-chunk logs: `<data_folder>/replay_logs/task-XXXX/chunk_*.log`.
- QC per task (run after each task completes):
  ```bash
  python /behavior-src/OmniGibson/scripts/learning/verify_task_sidecars.py \
      --data_folder /b1k-data --task-id 3
  ```
  Checks: every episode present exactly once, 229-dim proprio, >100 steps,
  active-phase median effort in [5, 500] N·m.
- Upload results per finished task (tiny):
  ```bash
  huggingface-cli upload ecappiell/b1k-torque-sidecars \
      /b1k-data/replayed_torque/task-0003 task-0003 --repo-type dataset --private
  ```

## Troubleshooting (all of these actually happened)

| Symptom | Cause / fix |
|---|---|
| `Scene must have exactly one robot, found 0` | `omnigibson-robot-assets` missing from `/data` (registry is a yaml glob) |
| Asset decryption errors | `datasets/omnigibson.key` missing |
| Isaac crashes at startup on host with driver 595.x | unsupported driver; need 580.65+ (containers inherit host driver) |
| `module 'warp.types' has no attribute 'array'` (native envs) | warp-lang got upgraded ≥1.13; pin `warp-lang<1.13` (container already correct) |
| Chunk "OK" but file has 1 episode | you're on code before `5eeb93917` — three chained-replay bugs (scene baseline / exit-0 / incremental flush) |
| Efforts in the thousands of N·m | running without `--preserve-frequencies` (1 kHz micro-step transients) |
| Workers slow, host swapping | too many workers for RAM/cores — apply the sizing rule |
| GPU util ~0% | normal — the workload is CPU-bound; check CPU instead |
| Episodes with ~50% zero effort tail | normal — teleop idle tail, articulation asleep; trim in post |

## Shared-machine etiquette

Only use GPUs explicitly allocated to you. A momentarily idle GPU on a shared
box is not a free GPU — ask first (this is a standing rule from Eduardo).

## Native path (lab boxes, no docker group) — what runs on idlab2

Identical results without containers: byte-copy of a validated conda env
rsynced to the **same absolute path** (`~/miniforge3/envs/behavior` — envs are
prefix-dependent; this only works user→same-username), repo + datasets rsynced,
then the same driver invocation with `~/miniforge3/envs/behavior/bin/python`.
See DEPLOYMENT.md § idlab2 for the exact bring-up record.

## Threading (critical for multi-worker throughput)

Two per-instance thread pools default to ALL cores and destroy multi-worker
scaling (observed: load 69 on 24 cores, 8× slowdown at 4 workers):

1. **torch/OpenMP intra-op pools** — the dominant one. The driver caps them
   automatically (`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` =
   `cores / total_workers − 1`, override via `REPLAY_PHYSICS_THREADS`).
2. **PhysX CPU dispatcher** — capped via the same value
   (`--/physics/numThreads` kit override; `--physics-threads` on
   replay_torque.py).

With caps: near-linear worker scaling (idlab2: 4 workers ≈ 4 × single-worker
rate, load ~45). If you script workers yourself, set all four knobs.
