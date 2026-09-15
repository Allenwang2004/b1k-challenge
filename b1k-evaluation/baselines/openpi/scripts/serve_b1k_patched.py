"""
Patched variant of serve_b1k.py for the current (2026-era) omnigibson.eval module layout.

The original scripts/serve_b1k.py imports `omnigibson.learning`, which was removed in the
v3.9.0 upstream merge on this fork (renamed to `omnigibson.eval`), so it no longer runs.

Verified against OmniGibson/omnigibson/eval/r1pro.yaml + eval_utils.py: the wire protocol and
the camera/proprio observation keys B1KPolicyWrapper.process_obs() reads
(`{robot}::proprio`, `{robot}::{robot}:zed_link:Camera:0::rgb`, etc.) are UNCHANGED from the
2025-era layout, so no obs-schema fix is needed -- only two import-path issues:

  1. WebsocketPolicyServer now lives under omnigibson.eval.utils.network_utils.
  2. BehaviorLerobotDatasetMetadata (only ever used here to resolve a task's text prompt) no
     longer exists. Replaced with a lookup into BEHAVIOR-1K/docs/challenge/task_data.json,
     which ships in this repo and needs no dataset download. Falls back to --default-prompt /
     the wrapper's built-in default if that file isn't mounted.

Everything else (Args/Checkpoint/Default, create_policy, the server loop) is unchanged from
scripts/serve_b1k.py.
"""
import dataclasses
import enum
import json
import logging
import os
import socket

import tyro

from omnigibson.eval.utils.network_utils import WebsocketPolicyServer

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config

# Mount BEHAVIOR-1K/docs/challenge/task_data.json here (or set this env var) to resolve
# instructions for tasks other than turning_on_radio.
TASK_DATA_JSON = os.environ.get("B1K_TASK_DATA_JSON", "/behavior-src/docs/challenge/task_data.json")


def _load_task_instruction(task_name: str | None) -> str:
    default = "Turn on the radio receiver that's on the table in the living room."
    if not task_name:
        return default
    if not os.path.exists(TASK_DATA_JSON):
        logging.warning(f"{TASK_DATA_JSON} not mounted; using default prompt for task {task_name!r}.")
        return default
    with open(TASK_DATA_JSON) as f:
        tasks = json.load(f)["tasks"]
    for t in tasks:
        if t["id"] == task_name:
            return t["instruction"]
    logging.warning(f"No instruction found for task {task_name!r} in {TASK_DATA_JSON}; using default prompt.")
    return default


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    dataset_root: str | None = None
    task_name: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
    )


def main(args: Args) -> None:
    prompt = args.default_prompt or _load_task_instruction(args.task_name)
    logging.info(f"Using prompt: {prompt}")

    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    policy = B1KPolicyWrapper(policy, text_prompt=prompt)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
