# Training on the 2026 challenge demos (LeRobot v3.0)

Status as of 2026-09-15: the whole pipeline runs on `behavior-1k/2026-challenge-demos`
(env → data → norm stats → FAST tokenizer → data loader → model init → weight restore) except the
train step itself, which does not fit in the GPU memory that was free at the time (see "GPU memory").

## What changed vs. the upstream ILIA repo

- `openpi/` is a vendored copy of `wensi-ai/openpi@01177e0` (ILIA's pinned commit, no `.git`). Its
  lerobot pin was switched from `huggingface/lerobot@577cd10` (0.3.4, v2.1 datasets) to
  `wensi-ai/lerobot@release/b1k` (0.5.2, v3.0 datasets). `pyproject.toml` overrides
  `huggingface-hub<1.0` (transformers 4.53.2 needs it) and pins `torchcodec==0.4.0` (torch 2.7.1).
- `src/b1k/training/behavior_dataset.py`: new loader replacing
  `omnigibson.learning.datas.lerobot_dataset.BehaviorLeRobotDataset` (removed in BEHAVIOR-1K v3.9.0).
  RGB only, local partial downloads, 2026 camera keys renamed to the 2025 keys the transforms expect.
- `openpi/src/openpi/policies/b1k_proprio.py`: vendored 2026 R1Pro `PROPRIOCEPTION_INDICES`
  (61-dim state) so the training venv does not need OmniGibson. `b1k_policy.py` /
  `eval_b1k_wrapper.py` import it from there.
- `scripts/compute_norm_stats.py`, `scripts/train_fast_tokenizer.py`: enumerate episodes from
  `data/chunk-*/file-*.parquet` (v3) as well as the old `data/task-*/episode_*.parquet`.
- `scripts/train.py`: first-batch image logging gathers via `jax.device_get` (integer-indexing a
  sharded array crashes with `CUDA_ERROR_ILLEGAL_ADDRESS` on this box with jax 0.5.3 + 2 GPUs).
- `src/b1k/training/config.py`: `pi_behavior_b1k_fast` points at
  `/home/b1k-challenge/evaluation/train_set/2026-challenge-demos`.

Task indices 0–49 of the 2026 dataset equal the 2025 ones (`turning_on_radio`=0,
`picking_up_trash`=1, ...), so the model's 50-task embedding / `TASK_NUM_STAGES` tables line up
without remapping. Tasks 50–99 are outside the model's tables.

## Commands (from `b1k-train/`)

```bash
# env
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11

# data: one task, RGB only (~5.4 GB) + the full episode table (needed: it is positional)
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="behavior-1k/2026-challenge-demos", repo_type="dataset",
    local_dir="/home/b1k-challenge/evaluation/train_set/2026-challenge-demos", max_workers=8,
    allow_patterns=["meta/info.json","meta/stats.json","meta/tasks.jsonl","meta/tasks.parquet","meta/episodes/**",
                    "annotations/task-0001/*","data/chunk-001/*","videos/observation.rgb.*/chunk-001/*"])
PY

# pre-training assets (both verified, ~30 s / ~60 s on 200 episodes)
.venv/bin/python scripts/compute_norm_stats.py --config-name pi_behavior_b1k_fast --correlation --num-workers 16
.venv/bin/python scripts/train_fast_tokenizer.py --config-name pi_behavior_b1k_fast --encoded-dims="0:6,7:23" --vocab-size=1024 --num-workers 16

# smoke training (pi05_base is downloaded to ~/.cache/openpi on first run, 11.6 GB)
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
.venv/bin/python scripts/train.py pi_behavior_b1k_fast --exp_name smoke_task1 \
    --batch_size 2 --fsdp_devices 2 --num_train_steps 30 --save_interval 20 --keep_period 20 \
    --log_interval 5 --num_workers 4 --no-wandb_enabled --overwrite
```

## GPU memory

`init_train_state` needs ~50 GB on one GPU (fp32 params + AdamW + EMA of a 3.3B model) and the
train step additionally gathers the full fp32 params (~13.4 GB) under FSDP. With ~34 GB free per GPU
(other users' jobs held ~62 GB on each), every variant failed with `RESOURCE_EXHAUSTED`:
single GPU; 2×FSDP; 2×FSDP `--ema_decay None`; 2×FSDP `--ema_decay None --num_flow_samples 1`.
Re-run the smoke command above when a GPU has ≥ ~60 GB free (or both have ≥ ~40 GB).
