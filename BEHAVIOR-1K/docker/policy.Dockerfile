# syntax=docker/dockerfile:1
# Policy-side containers — NO Isaac Sim anywhere in this image family.
# Multi-stage; training and evaluation share the same base:
#   base : python 3.10 + torch cu124 + bddl3 + OmniGibson[eval]
#          (websocket server/policy classes only — importing omnigibson works
#          without Isaac Sim as long as the sim is never launched)
#   prod : slim serving image = the challenge submission artifact
#          docker build -f docker/policy.Dockerfile --target prod -t b1k-policy-prod .
#   dev  : research image (training tooling, notebooks)
#          docker build -f docker/policy.Dockerfile --target dev -t b1k-policy-dev .
#
# Runs on any CUDA-capable GPU/driver (pure compute — unaffected by the
# Isaac-Sim 595.x driver issue; fine on H100 and RTX PRO 6000 Blackwell).
# Challenge constraint: submission policies must fit a single 24 GB VRAM GPU.
#
# This replaces upstream's docker/submission.Dockerfile, which is broken on
# v3.9.0 (CMD imports the removed omnigibson.learning package and installs
# `-e bddl` — the directory is bddl3).
# NOTE: continuumio/miniconda3 is deprecated (successor: anaconda/miniconda)
# and its current base is Debian 13, where libgl1-mesa-glx was removed —
# hence libgl1 below (upstream's submission.Dockerfile still lists the old name).
FROM continuumio/miniconda3 AS base

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libudev-dev \
    libinput-dev \
    linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

# python 3.11: the pinned lerobot (wensi-ai/lerobot@release/b1k, pulled in by
# OmniGibson[eval]) requires >=3.11 — upstream's submission.Dockerfile still
# says 3.10 and cannot build against v3.9.0.
RUN conda create -n behavior python=3.11 -y -c conda-forge
SHELL ["conda", "run", "-n", "behavior", "/bin/bash", "-c"]

RUN pip install "numpy<2" "setuptools<=79"
# torch 2.7.0+cu128: parity with setup.sh's behavior env (--cuda-version 12.8).
# OmniGibson's setup.py has no torch pin, so the version MUST be repeated in
# the eval install below or pip upgrades to latest (observed: 2.11+cu130).
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

COPY bddl3 /b1k-src/bddl3
COPY OmniGibson /b1k-src/OmniGibson
WORKDIR /b1k-src

RUN pip install -e bddl3
RUN pip install -e "OmniGibson[eval]" "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" --extra-index-url https://download.pytorch.org/whl/cu128

ENV PATH=/opt/conda/envs/behavior/bin:$PATH
ENV CONDA_DEFAULT_ENV=behavior

# omnigibson.macros asserts the data path exists at import time; the policy
# containers never load sim assets, so an empty stub satisfies it. Mount real
# data here only if a policy needs task metadata.
RUN mkdir -p /data
ENV OMNIGIBSON_DATA_PATH=/data

RUN pip install uv

COPY b1k-baselines/baselines/openpi /openpi
WORKDIR /openpi
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

RUN uv pip install "huggingface_hub<1.0,>=0.30.0"

RUN sed -i \
    's/from omnigibson.learning.utils.eval_utils/from omnigibson.eval.utils.eval_utils/' \
    src/openpi/policies/b1k_policy.py

ENV TORCHDYNAMO_DISABLE=1

# All task checkpoints ship in one image (challenge rule: single submission
# image for all 100 tasks). docker/checkpoint_map.json maps task_name ->
# checkpoint dir name under behavior_checkpoints/; add a line there whenever
# a new task's checkpoint is dropped into behavior_checkpoints/.
COPY behavior_checkpoints/49999_radio /checkpoints/49999_radio
COPY docker/checkpoint_map.json /checkpoint_map.json
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /
ENV B1K_TASK_DATA_JSON=/behavior-src/docs/challenge/task_data.json
# TASK_NAME selects the checkpoint at container start (see entrypoint.sh) —
# the evaluator is expected to restart this container per task with
# `-e TASK_NAME=<task>`, since the sim<->policy wire protocol carries no
# per-request task/prompt field (prompt is fixed for the server's lifetime).
ENTRYPOINT ["/entrypoint.sh"]

# ── prod: challenge submission / evaluation serving ─────────────────────────
FROM base AS prod

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=30 \
    CMD curl -sf http://localhost:8000/healthz || exit 1

# Default: zero-action smoke server (validates the wire contract end-to-end).
# For a real submission, replace CMD with your checkpoint server exposing the
# same WebsocketPolicyServer interface on port 8000.
CMD ["python", "-u", "-c", "from omnigibson.eval.utils.network_utils import WebsocketPolicyServer; from omnigibson.eval.policies import LocalPolicy; server = WebsocketPolicyServer(LocalPolicy(action_dim=23)); server.serve_forever()"]

# ── dev: research/training image ─────────────────────────────────────────────
FROM base AS dev

RUN pip install jupyterlab wandb matplotlib pandas scikit-learn einops uv

# b1k-baselines (openpi serving venv, il_lib ACT) stays a separate repo,
# mounted at runtime:  -v $HOME/Documents/dev/b1k-baselines:/b1k-baselines
# openpi keeps its own uv venv INSIDE the mounted repo (the two-env pattern:
# incompatible JAX/openpi deps never enter this conda env).
CMD ["/bin/bash"]
