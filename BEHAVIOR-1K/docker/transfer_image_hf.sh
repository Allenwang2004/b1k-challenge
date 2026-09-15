#!/usr/bin/env bash
# Transfer docker images to/from an SSL-whitelisted server (e.g. Zettabyte,
# where docker.io/nvcr.io are blocked but huggingface.co is allowed) using a
# private HF dataset repo as the artifact channel.
#
#   Save side (any machine with registry access):
#     ./docker/transfer_image_hf.sh save b1k-sim ecappiell/b1k-artifacts
#   Load side (whitelisted server):
#     ./docker/transfer_image_hf.sh load b1k-sim ecappiell/b1k-artifacts
#
# Requires: huggingface-cli (pip install -U huggingface_hub), `hf auth login`
# done once on each machine, zstd.
set -euo pipefail

MODE=${1:?save|load}
IMAGE=${2:?image name, e.g. b1k-sim}
HF_REPO=${3:?private HF dataset repo, e.g. ecappiell/b1k-artifacts}
SAFE_NAME=$(echo "${IMAGE}" | tr '/:' '__')
TARBALL="${SAFE_NAME}.tar.zst"

case "${MODE}" in
save)
    echo ">>> Saving ${IMAGE} -> ${TARBALL}"
    docker save "${IMAGE}" | zstd -T0 -8 -o "${TARBALL}"
    echo ">>> Uploading to hf://datasets/${HF_REPO}/images/${TARBALL}"
    huggingface-cli upload --repo-type dataset "${HF_REPO}" "${TARBALL}" "images/${TARBALL}"
    rm -f "${TARBALL}"
    ;;
load)
    echo ">>> Downloading hf://datasets/${HF_REPO}/images/${TARBALL}"
    huggingface-cli download --repo-type dataset "${HF_REPO}" "images/${TARBALL}" --local-dir .
    echo ">>> Loading into docker"
    zstd -dc "images/${TARBALL}" | docker load
    rm -rf images
    ;;
*)
    echo "unknown mode: ${MODE} (use save|load)" >&2
    exit 1
    ;;
esac
echo ">>> Done."
