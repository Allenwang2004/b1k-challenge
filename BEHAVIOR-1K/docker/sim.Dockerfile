# b1k-sim — simulation container (Isaac Sim 5.1 + OmniGibson, fork overlay)
#
# Base: upstream prebuilt image (built from docker/Dockerfile; Isaac Sim 5.1,
# full BEHAVIOR conda env, CUDA 12.8, curobo precompiled for RTX 20/30/40/50,
# A100, GB-series, RTX PRO 6000). Our fork's sources overlay /behavior-src so
# the image's editable installs resolve to fork code.
#
# HOST DRIVER REALITY (containers do NOT isolate the GPU driver — the NVIDIA
# container toolkit injects the HOST driver user-space into the container):
#   * Isaac Sim 5.1 requires driver 580.65.06+ (Linux); the 595.x branch
#     crashes at startup (librtx.scenedb.plugin.so) on bare metal AND inside
#     containers. Do not run this image on a 595.x host.
#   * H100/A100 (no RT cores): camera rendering is unsupported; CAMERA-LESS
#     workloads (force/BDDL replay, physics-only) work headless.
#
# Build (repo root):  docker build -f docker/sim.Dockerfile -t b1k-sim .
# If the upstream base is unavailable, build it first:
#   ./docker/build_docker.sh   (produces stanfordvl/behavior from docker/Dockerfile)
ARG BASE_IMAGE=stanfordvl/behavior:3.9.0
FROM ${BASE_IMAGE}

# Overlay fork sources over the baked editable installs
COPY bddl3 /behavior-src/bddl3
COPY OmniGibson /behavior-src/OmniGibson
COPY scripts /behavior-src/scripts

ENV OMNIGIBSON_HEADLESS=1 \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ACCEPT_EULA=YES

# The base image declares VOLUME /data (OMNIGIBSON_DATA_PATH) and /cache.
# Mount the repo's datasets/ at /data (must contain behavior-1k-assets,
# omnigibson-robot-assets, 2026-challenge-task-instances).
WORKDIR /behavior-src

# Inherited ENTRYPOINT activates the behavior conda env. Default: shell.
# Common commands:
#   Eval rollout against a policy server:
#     python -m omnigibson.eval.eval --task-name turning_on_radio \
#       --host <policy-host> --port 8000 --instance-indices 0 --output-dir /data/outputs
#   Torque-augmented live collection:
#     python scripts/collect_torque_dataset.py --task turning_on_radio \
#       --policy-host <policy-host> --policy-port 8000
CMD ["/bin/bash"]
