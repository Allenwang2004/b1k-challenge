"""2026 R1Pro proprioception layout (61-dim ``observation.state``).

Training venv: the vendored table in openpi (``openpi/src/openpi/policies/b1k_proprio.py``).
Eval venv (b1k-evaluation openpi + OmniGibson 3.9.0, via serve_ilia.py): the identical table from OmniGibson.
"""

try:
    from openpi.policies.b1k_proprio import PROPRIOCEPTION_INDICES, STATE_DIM, check_state_dim
except ImportError:
    from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES

    STATE_DIM = 61

    def check_state_dim(state) -> None:
        import numpy as np

        if np.shape(state)[-1] != STATE_DIM:
            raise ValueError(f"observation.state has {np.shape(state)[-1]} dims, expected {STATE_DIM} (2026 R1Pro)")

__all__ = ["PROPRIOCEPTION_INDICES", "STATE_DIM", "check_state_dim"]
