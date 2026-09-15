"""Serve a B1K openpi checkpoint **statelessly**, for off-policy agreement tests.

Why not ``serve_b1k_patched.py``: its ``B1KPolicyWrapper`` defaults to
``control_mode="temporal_ensemble"``, which keeps a queue of past action chunks
and blends each new prediction with them. That is correct for a rollout, where
queries arrive at consecutive timesteps -- and wrong here, where the harness
samples frames every N steps from a demo. Ensembling across frames that are a
second apart would mix unrelated predictions and the "disagreement" measured
would be an artifact of the queue rather than of the policy.

So this server holds no state between queries. Each ``act`` is an independent
``policy.infer`` on one frame, and it returns the model's **whole action chunk**
(H, 23) rather than a single blended step, which is what lets the client measure
how fast the prediction decays over the chunk horizon.

It reuses the checkpoint loading, prompt lookup and wire protocol of
``serve_b1k_patched.py`` unchanged, so the weights and preprocessing are
identical to what the eval harness would use.

Run it from the openpi venv on a machine that has the checkpoint::

    cd ~/evaluation/b1k-baselines/baselines/openpi
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \\
    .venv/bin/python serve_b1k_offpolicy.py \\
        --policy.config pi05_b1k \\
        --policy.dir ~/evaluation/behavior_checkpoints/30ep \\
        --task-name freeze_fruit --port 8000

The JAX memory flags matter on a shared box: JAX preallocates ~75% of the
device by default, which would evict other people's work off the GPU.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import socket

import numpy as np
import torch
import tyro
from openpi.policies import policy_config as _policy_config
from openpi.shared.eval_b1k_wrapper import RESIZE_SIZE
from openpi.training import config as _config
from openpi_client.image_tools import resize_with_pad

from omnigibson.eval.utils.network_utils import WebsocketPolicyServer

TASK_DATA_JSON = os.environ.get("B1K_TASK_DATA_JSON",
                                os.path.expanduser("~/evaluation/BEHAVIOR-1K/docs/challenge/task_data.json"))
DEFAULT_PROMPT = "Turn on the radio receiver that's on the table in the living room."

CAM_KEYS = [
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",          # egocentric
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
]


def _load_task_instruction(task_name: str | None) -> str:
    """The prompt for a task, from the demo dataset's own task table.

    ``docs/challenge/task_data.json`` is not usable for this: most of its 100
    entries carry no ``instruction`` field at all (only a handful do), so a
    lookup there silently falls through to the radio prompt. The demos repo's
    ``meta/tasks.jsonl`` has all 100 and is the text the policies were trained
    against, which is the text they must be prompted with.
    """
    if not task_name:
        return DEFAULT_PROMPT
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("behavior-1k/2026-challenge-demos", "meta/tasks.jsonl", repo_type="dataset")
    with open(path) as fp:
        for line in fp:
            rec = json.loads(line)
            if rec.get("task_name") == task_name:
                return rec["task"]
    raise SystemExit(
        f"task {task_name!r} not found in meta/tasks.jsonl. Serving the wrong prompt would "
        "silently invalidate the comparison, so refusing to fall back to the radio default."
    )


class StatelessB1KPolicy:
    """One frame in, one full action chunk out. No queue, no ensembling."""

    def __init__(self, policy, text_prompt: str):
        self.policy = policy
        self.text_prompt = text_prompt
        self.n_calls = 0

    def reset(self) -> None:  # the server calls this; there is nothing to reset
        pass

    def _batch(self, obs: dict) -> dict:
        imgs = [resize_with_pad(np.asarray(obs[k])[None, ..., :3], RESIZE_SIZE, RESIZE_SIZE)[0]
                for k in CAM_KEYS]
        state = np.asarray(obs["robot_r1::proprio"], dtype=np.float64).reshape(-1)
        return {
            "observation/egocentric_camera": imgs[0],
            "observation/wrist_image_left": imgs[1],
            "observation/wrist_image_right": imgs[2],
            "observation/state": state,
            "prompt": self.text_prompt,
        }

    def act(self, obs: dict) -> torch.Tensor:
        actions = self.policy.infer(self._batch(obs))["actions"]
        self.n_calls += 1
        if self.n_calls % 50 == 0:
            logging.info("served %d queries", self.n_calls)
        return torch.from_numpy(np.asarray(actions, dtype=np.float32))


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    policy: Checkpoint
    task_name: str | None = None
    default_prompt: str | None = None
    port: int = 8000


def main(args: Args) -> None:
    prompt = args.default_prompt or _load_task_instruction(args.task_name)
    logging.info("prompt: %s", prompt)
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), os.path.expanduser(args.policy.dir)
    )
    wrapped = StatelessB1KPolicy(policy, prompt)
    logging.info("serving stateless policy on %s:%d", socket.gethostname(), args.port)
    WebsocketPolicyServer(policy=wrapped, host="0.0.0.0", port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
