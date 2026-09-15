# Serving a real π0/π0.5 checkpoint for `turning_on_radio` eval

What was done this session to go from "sim ↔ zero-action stub" to "sim ↔ real
π0 checkpoint, driving a rollout on video", on `idlab_server1`
(`feat/policy-eval`, 2× RTX PRO 6000 Blackwell). Every command below was
actually run and verified, in order. Kept here instead of only in chat
history so the next person (or session) doesn't have to re-derive it.

---

## 1. Why two containers, and why openpi isn't baked into `b1k-policy-*`

`docker/sim.Dockerfile` (→ `b1k-sim`) and `docker/policy.Dockerfile`
(→ `b1k-policy-prod`/`-dev`) are separate because of two independent hard
conflicts, not preference:

- **Driver**: Isaac Sim needs a specific NVIDIA driver branch; containers
  inherit the *host* driver (the NVIDIA container toolkit injects it, it
  isn't isolated), so this is a host-level constraint no amount of
  containerizing fixes.
- **Python stack**: the sim side is PyTorch + `OmniGibson[eval]` (which pulls
  `lerobot` from `wensi-ai/lerobot@release/b1k`); the real policy is JAX +
  `openpi` (which wants `lerobot` from official `huggingface/lerobot` at a
  pinned commit, reporting `0.3.4`). Same import name, two different forks —
  can't both live in one `site-packages`.

`docker/policy.Dockerfile`'s `base`/`prod`/`dev` targets only install
`OmniGibson[eval]` (for the websocket protocol code) + PyTorch. They were
never meant to contain openpi/JAX directly — per the file's own comments,
openpi is supposed to arrive via a **separate, isolated environment**
mounted in at runtime. This doc is the concrete version of that.

---

## 2. Building the openpi environment (done once, on the host)

Not inside a container — built directly on `idlab_server1`'s filesystem at
`~/b1k-baselines/baselines/openpi/.venv`, using `uv`. This turned out to be
mountable into a container later (§4) without any ABI problems, so there was
no need to rebuild it a second time inside a container image.

```bash
git clone https://github.com/StanfordVL/b1k-baselines.git ~/b1k-baselines
cd ~/b1k-baselines
git submodule update --init baselines/openpi

cd baselines/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Also needed for the websocket protocol code (network_utils.py) and for
# openpi's own PROPRIOCEPTION_INDICES import (see fix #3 below):
uv pip install -e /home/allen19/BEHAVIOR-1K/bddl3
uv pip install -e "/home/allen19/BEHAVIOR-1K/OmniGibson[eval]"
```

### Three bugs this install surfaces, and the fixes

1. **Import-time crash: `TORCHDYNAMO_DISABLE=1` required.**
   `OmniGibson/omnigibson/object_states/attached_to.py` calls a
   `@torch_compile`-decorated `euler2quat` *eagerly at module import time*.
   That triggers a Triton→gcc→`-lcuda` compile, which fails on this host
   (has `libcuda.so.1`, no unversioned `libcuda.so` symlink for the linker,
   and no sudo to add one). Fix: always export
   `TORCHDYNAMO_DISABLE=1` before running anything that imports `omnigibson`
   in this venv. Forces eager execution, sidesteps the whole Triton/CUDA
   compile toolchain requirement (not needed for serving anyway).

2. **`huggingface-hub` version conflict.** `OmniGibson[eval]`'s install
   pulls `huggingface-hub==1.24.0`; `transformers` (an openpi dependency)
   requires `<1.0`. Fix:
   ```bash
   uv pip install "huggingface_hub<1.0,>=0.30.0"   # → resolved to 0.36.2
   ```

3. **A second, separate `omnigibson.learning` import**, not in our own
   script — inside **openpi's own package**,
   `baselines/openpi/src/openpi/policies/b1k_policy.py:8`:
   ```python
   from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
   ```
   `omnigibson.learning` was deleted and renamed to `omnigibson.eval` in the
   fork's v3.9.0 upstream sync. Pulled in transitively via `config.py`'s
   import of `b1k_policy` for `B1kOutputs`. Fixed in place, one line:
   ```python
   from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES
   ```

4. **`lerobot` fork got silently overwritten, but it doesn't matter for
   serving.** After step 2's `OmniGibson[eval]` install, `lerobot` in this
   venv is `wensi-ai/lerobot@release/b1k` (reports `0.5.2`), not the
   official fork openpi originally wanted (`0.3.4`). Confirmed via:
   ```bash
   .venv/bin/python -c "import lerobot; print(lerobot.__file__)"
   cat .venv/lib/python3.11/site-packages/lerobot-*.dist-info/direct_url.json
   ```
   This is a real, silent "last install wins" conflict — but the serving
   path (`create_trained_policy` → `Policy.infer`) never touches lerobot's
   dataset-loading API, only the (unaffected) model/checkpoint code, so it
   never actually raises. **Would likely break if this venv were used for
   training** (`scripts/train_val.py`), which does exercise
   `LeRobotB1KDataConfig`. Not fixed — just noted, since fixing it isn't
   needed for serving and would cost more `uv pip install --no-deps`
   surgery than it's worth right now.

### `serve_b1k_patched.py` — our own copy, not an in-place edit

`scripts/serve_b1k.py` (upstream) imports the deleted `omnigibson.learning`
module and resolves task prompts via a `BehaviorLerobotDatasetMetadata`
stub that no longer exists either. Rather than patch the original in place,
a parallel copy lives at
`~/b1k-baselines/baselines/openpi/scripts/serve_b1k_patched.py`:

- `from omnigibson.eval.utils.network_utils import WebsocketPolicyServer`
  (fixed import path)
- Task-instruction lookup reads `BEHAVIOR-1K/docs/challenge/task_data.json`
  (ships in-repo, needs no dataset download) instead of the dead
  `BehaviorLerobotDatasetMetadata` stub. Falls back to the wrapper's
  built-in default prompt if that file isn't mounted/found.
- Everything else (`Args`/`Checkpoint`/`Default` dataclasses, the server
  loop) is untouched from the original.

Verified the obs *schema* itself didn't need any fix: `r1pro.yaml`'s
`eval.camera_sensor_names` and the flattened `proprio` key
(`omnigibson/eval/utils/eval_utils.py`) produce exactly the same wire keys
(`{robot}::proprio`, `{robot}::{robot}:zed_link:Camera:0::rgb`, etc.)
`B1KPolicyWrapper.process_obs()` (in openpi's `eval_b1k_wrapper.py`) already
expects — only the import paths had rotted, not the protocol.

---

## 3. Getting a checkpoint that actually loads

### First attempt (failed): `IliaLarchenko/behavior_submission`

The 2025-Challenge 1st-place π0.5 solution, on HuggingFace:

```bash
hf download IliaLarchenko/behavior_submission --include "checkpoint_2/*" \
  --local-dir ~/behavior_checkpoints/
```

Loads fine at the Orbax level, but `train_config.model.load(...)` then
throws a PyTree structure mismatch: openpi's `pi0_b1k`/`pi05_b1k` configs
(`config.py:695-747`) both use `Pi0Config(paligemma_variant="gemma_2b_lora")`
— i.e. they expect LoRA adapter weights (`lora_a`/`lora_b`) at every
attention/MLP layer. IliaLarchenko's checkpoint doesn't have them (it's
from an independent training codebase,
`github.com/IliaLarchenko/behavior-1k-solution`, not openpi's LoRA recipe).
Not a `--policy.config` selection mistake — both registered configs use the
same LoRA variant, so there was no existing config that matched this
checkpoint's architecture. Deleted (12GB, not needed):

```bash
rm -rf ~/behavior_checkpoints/checkpoint_2
```

### What actually worked: b1k-baselines' own `turning_on_radio` checkpoint

Per `BEHAVIOR-1K/scripts/SETUP.md` ("Option A") and
`b1k-baselines/tutorials/openpi.md`: task-specific checkpoints, trained by
running openpi's *own* `pi0_b1k` recipe
(`uv run scripts/train_val.py pi0_b1k ... --weight_loader.params_path=gs://openpi-assets/checkpoints/pi0_base/params`,
which matches `pi0_b1k`'s own `weight_loader` in `config.py:708` exactly) —
so its PyTree structurally matches `pi0_b1k`, and its `assets/` folder
resolves under `pi0_b1k`'s default `asset_id`
(`"behavior-1k/2025-challenge-demos"`) with **no symlink workaround needed**
(confirmed after unzip — unlike IliaLarchenko's checkpoint, which needed one
before it was deleted).

Only two such checkpoints exist (`turning_on_radio`, `picking_up_trash`) —
not general-purpose, but an exact match for the task under test. Hosted on
Google Drive, not HF — needed `gdown` (not curl/wget; Drive serves an
interstitial HTML page instead of the binary past ~100MB):

```bash
cd ~/b1k-baselines/baselines/openpi
uv pip install gdown

.venv/bin/gdown 'https://drive.google.com/file/d/1YU7evHBj7vfjmE-tNK-Rbie8ytholQTc/view?usp=sharing' \
  -O ~/behavior_checkpoints/openpi_turning_on_radio.zip
# → 12.2GB. gdown auto-resolves the share URL; the --fuzzy flag some gdown
#   versions want doesn't exist in gdown 6.1.0, just pass the URL directly.

cd ~/behavior_checkpoints
python3 -c "import zipfile; zipfile.ZipFile('openpi_turning_on_radio.zip').extractall('.')"
rm openpi_turning_on_radio.zip
```

Unpacks to `~/behavior_checkpoints/49999_radio/{params/, train_state/,
assets/behavior-1k/2025-challenge-demos/norm_stats.json, _CHECKPOINT_METADATA}`
— exactly the layout `scripts/SETUP.md` documents, and (verified) the
`assets/` path pi0_b1k expects natively.

---

## 4. Serving it — native process, then containerized

### Native (host) — what actually got the first successful rollout

```bash
cd ~/b1k-baselines/baselines/openpi
B1K_TASK_DATA_JSON=/home/allen19/BEHAVIOR-1K/docs/challenge/task_data.json \
TORCHDYNAMO_DISABLE=1 \
.venv/bin/python scripts/serve_b1k_patched.py \
  --task_name=turning_on_radio \
  policy:checkpoint \
  --policy.config=pi0_b1k \
  --policy.dir=/home/allen19/behavior_checkpoints/49999_radio
```
Checkpoint restore: ~6s (local disk, warm). `curl -sf
http://localhost:8000/healthz` → `OK`.

### Containerized via mount — the "real two containers" version

Rather than rebuilding the whole `uv sync` inside a `b1k-policy-*` image
(same result, ~10 more minutes and several more GB for no functional
difference), the already-built host venv gets **mounted in** at runtime.
Confirmed this has no ABI/glibc issues (JAX finds both GPUs fine despite the
venv being built on the host and the container's base being a different
Debian than the host OS) — the earlier worry about this was real but didn't
materialize in practice.

**Everything that gets `-e python`'d or imports `omnigibson` needs to be
mounted at the *same absolute path* inside the container as on the host** —
these venvs/installs use editable installs and symlinks with absolute host
paths baked in (`uv`'s python symlink points at
`~/.local/share/uv/python/...`; `OmniGibson[eval]`'s editable install points
at `/home/allen19/BEHAVIOR-1K/OmniGibson`). Remounting at a different
container-side path breaks those references.

```bash
docker run --rm --gpus all -d --name policy-mounted-test -p 8000:8000 \
  -v /home/allen19/b1k-baselines:/home/allen19/b1k-baselines \
  -v /home/allen19/.local/share/uv:/home/allen19/.local/share/uv \
  -v /home/allen19/behavior_checkpoints:/home/allen19/behavior_checkpoints \
  -v /home/allen19/BEHAVIOR-1K:/home/allen19/BEHAVIOR-1K \
  -v /home/allen19/BEHAVIOR-1K/datasets:/data \
  b1k-policy-prod bash -c 'sleep infinity'

docker exec -d \
  -e TORCHDYNAMO_DISABLE=1 \
  -e B1K_TASK_DATA_JSON=/home/allen19/BEHAVIOR-1K/docs/challenge/task_data.json \
  policy-mounted-test bash -c \
  'cd /home/allen19/b1k-baselines/baselines/openpi && \
   .venv/bin/python scripts/serve_b1k_patched.py \
     --task_name=turning_on_radio policy:checkpoint \
     --policy.config=pi0_b1k \
     --policy.dir=/home/allen19/behavior_checkpoints/49999_radio \
     > /tmp/serve.log 2>&1'
```
(`/data` mount is needed because `omnigibson.eval.utils.eval_utils`'s
`TASK_NAMES_TO_INDICES` reads `$OMNIGIBSON_DATA_PATH/2026-challenge-task-instances/metadata/B100_task_misc.csv`,
and the `b1k-policy-*` images default `OMNIGIBSON_DATA_PATH=/data`.)

Verify: `curl -sf http://localhost:8000/healthz` → `OK`; `docker exec
policy-mounted-test tail -f /tmp/serve.log` shows the same restore/prompt
lines as the native run.

---

## 5. Running the eval

Same `b1k-sim` invocation used for the earlier zero-action wiring test, just
pointed at the real server and given a real step budget instead of the
artificially short one used for pure wire-checking:

```bash
docker run --rm --gpus all --network host \
  -v /home/allen19/BEHAVIOR-1K/datasets:/data \
  -v /home/allen19/BEHAVIOR-1K/outputs:/data/outputs \
  -e OMNIGIBSON_HEADLESS=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  b1k-sim python -m omnigibson.eval.eval \
  --task-name turning_on_radio --host 127.0.0.1 --port 8000 \
  --instance-indices 0 --num-rollouts 1 --write-video --max-steps 1500 \
  --output-dir /data/outputs/eval
```
Output: `outputs/eval/videos/turning_on_radio_301_0.mp4`,
`outputs/eval/json/turning_on_radio_301_0.json`.

`--network host` is required here specifically because the policy is
reachable at `127.0.0.1:8000` on the *host* (whether native or the
container above, which publishes `-p 8000:8000`) — a plain
`docker compose run sim` (bridge network) can't reach that without
`--add-host=host.docker.internal:host-gateway` and `--host
host.docker.internal` instead.

---

## 6. A system-level detour, unrelated to any of the above

Mid-session, `nvidia-smi` and all `--gpus all` containers started failing
with `Driver/library version mismatch`. Root-caused via `/var/log/apt/history.log`
and `/var/log/dpkg.log`: an admin manually churned the driver on this shared
box (purge → install `nvidia-driver-595-open` → purge → reinstall
`nvidia-driver-580-open` → in-place upgrade `580.159.03` → `580.173.02`,
timestamped `2026-07-24 07:00:34`) without reloading the kernel module or
rebooting. Not caused by anything in this doc — no `apt`/`dpkg`/driver
command was ever run as part of this pipeline (no sudo access the whole
session). Resolved itself once the module was reloaded (out of our control);
`nvidia-smi` confirmed healthy before re-running §4-5. Mentioned here only
so a future "GPU containers suddenly can't start" doesn't get mis-attributed
to anything above.

---

## 7. If you want a single self-contained image (checkpoint + env baked in)

Nothing above requires this — the mount-based container in §4 already gives
a "real" two-container split without paying to duplicate ~9GB of venv +
12GB of checkpoint into an image layer. Build a standalone image only if you
need to *ship* the policy somewhere without the host's `~/b1k-baselines` and
`~/behavior_checkpoints` alongside it (e.g. the actual challenge submission
artifact, or a different machine).

Extend `docker/policy.Dockerfile`'s `base` stage — after the existing
`OmniGibson[eval]` install, before the `prod`/`dev` split:

```dockerfile
# --- openpi + checkpoint, baked in ---
RUN pip install uv

# Build context must include b1k-baselines/ (as a sibling checkout, or a
# `COPY --from=` from a separate build stage that clones it) and the
# checkpoint directory. For a real build, add both to `.dockerignore`
# exceptions and the `docker build` context, or COPY from a pre-staged
# local path:
COPY b1k-baselines/baselines/openpi /openpi
WORKDIR /openpi
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Fix #2 from §2 (huggingface-hub/transformers conflict)
RUN uv pip install "huggingface_hub<1.0,>=0.30.0"

# Fix #3 from §2 (omnigibson.learning -> omnigibson.eval) — either patch at
# build time, or just make sure the b1k-baselines checkout you COPY in
# already has scripts/serve_b1k_patched.py and the b1k_policy.py fix, i.e.
# build from the same local checkout this session already patched rather
# than a fresh `git clone` of upstream.
RUN sed -i \
    's/from omnigibson.learning.utils.eval_utils/from omnigibson.eval.utils.eval_utils/' \
    src/openpi/policies/b1k_policy.py

# Fix #1 from §2 (Triton/-lcuda import-time crash) — set globally so it
# applies no matter how the container's CMD invokes python:
ENV TORCHDYNAMO_DISABLE=1

# Checkpoint (12GB — this is what makes the image heavy; consider a
# separate `checkpoint`-only image/volume instead of baking it in if you'll
# swap checkpoints often):
COPY behavior_checkpoints/49999_radio /checkpoints/49999_radio

WORKDIR /
ENV B1K_TASK_DATA_JSON=/behavior-src/docs/challenge/task_data.json
CMD ["/openpi/.venv/bin/python", "/openpi/scripts/serve_b1k_patched.py", \
     "--task_name=turning_on_radio", "policy:checkpoint", \
     "--policy.config=pi0_b1k", "--policy.dir=/checkpoints/49999_radio"]
```

Build (from the repo root, with `b1k-baselines/` and
`behavior_checkpoints/49999_radio/` staged into the build context first —
Docker can't `COPY` from outside the context):

```bash
cp -r ~/b1k-baselines .
cp -r ~/behavior_checkpoints/49999_radio behavior_checkpoints/49999_radio
docker build -f docker/policy.Dockerfile -t b1k-policy-pi0-radio .
docker run --rm --gpus all -p 8000:8000 b1k-policy-pi0-radio
```

Expect the image to land somewhere around 8GB (current `b1k-policy-prod`
base) + ~9GB (openpi venv, mostly JAX/torch wheels) + 12GB (checkpoint) ≈
**~29GB** — same order of magnitude as `b1k-sim` (14.6GB) or
`behavior-torque-collection` (51.5GB) already on this box, so budget disk
accordingly before building (this box has repeatedly dropped to single-digit
GB free from other users' activity during this session — check `df -BG /`
first).

The `lerobot` fork conflict noted in §2 fix #4 still applies here and is
still fine to leave unfixed for a pure serving image — only relevant if this
image is later repurposed for training.
