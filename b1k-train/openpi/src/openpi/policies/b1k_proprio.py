"""Proprioception layout of the BEHAVIOR-1K 2026 challenge demos (R1Pro, 61-dim ``observation.state``).

Vendored from ``omnigibson.eval.utils.eval_utils.PROPRIOCEPTION_INDICES`` (BEHAVIOR-1K v3.9.0) so the
training environment does not need OmniGibson installed. The 2025 layout (256-dim, from
``omnigibson.learning``) is *not* compatible with this table; ``check_state_dim`` guards against
feeding the wrong dataset through it.
"""

from collections import OrderedDict

import numpy as np

STATE_DIM = 61

PROPRIOCEPTION_INDICES = {
    "R1Pro": OrderedDict(
        {
            "base_qvel": np.s_[0:3],
            "arm_left_qpos": np.s_[3:10],
            "arm_left_qvel": np.s_[10:17],
            "eef_left_pos": np.s_[17:20],
            "eef_left_quat": np.s_[20:24],
            "gripper_left_qpos": np.s_[24:26],
            "gripper_left_qvel": np.s_[26:28],
            "arm_right_qpos": np.s_[28:35],
            "arm_right_qvel": np.s_[35:42],
            "eef_right_pos": np.s_[42:45],
            "eef_right_quat": np.s_[45:49],
            "gripper_right_qpos": np.s_[49:51],
            "gripper_right_qvel": np.s_[51:53],
            "trunk_qpos": np.s_[53:57],
            "trunk_qvel": np.s_[57:61],
        }
    ),
}


def check_state_dim(state) -> None:
    dim = np.shape(state)[-1]
    if dim != STATE_DIM:
        raise ValueError(
            f"observation.state has {dim} dims but PROPRIOCEPTION_INDICES expects the 2026 R1Pro layout "
            f"({STATE_DIM} dims). The 2025 demos (256-dim) need the omnigibson.learning table instead."
        )
