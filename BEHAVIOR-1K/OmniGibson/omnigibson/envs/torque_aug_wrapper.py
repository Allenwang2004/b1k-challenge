"""
TorqueAugDataWrapper — extends LeRobotDataWrapper with per-step joint
efforts (N·m) and per-arm gripper contact flags, emitting them as named
channels alongside the standard RGB + proprio observations.

Extra features added to every dataset frame:
    observation.joint_efforts    float32  (n_dof,)   measured torques (N·m) for all joints
    observation.contact_{arm}    float32  (1,)        1.0 if that gripper has ≥1 non-self contact
"""
import torch as th
from omnigibson.envs.env_base import Environment
from omnigibson.envs.lerobot_data_wrapper import LeRobotDataWrapper


class TorqueAugDataWrapper(LeRobotDataWrapper):
    """
    Subclass of LeRobotDataWrapper that augments every recorded frame with:
      - observation.joint_efforts  : raw measured joint torques (N·m), shape (n_dof,)
      - observation.contact_{arm}  : gripper contact flag per arm,    shape (1,)

    These extra channels are appended after the standard obs/action fields
    so the dataset is a strict superset of the baseline challenge format.
    Both channels are also present in the first (reset) frame so every
    (obs_t, action_t+1) pair carries the full signal.
    """

    @classmethod
    def get_lerobot_obs_mapping(cls, env: Environment, depth_output_unit: str = "m") -> tuple[dict, dict]:
        obs_mapping, obs_features = super().get_lerobot_obs_mapping(env, depth_output_unit=depth_output_unit)
        robot = env.robots[0]

        obs_features["observation.joint_efforts"] = {
            "dtype": "float32",
            "shape": (robot.n_dof,),
            "names": None,
        }
        for arm in robot.arm_names:
            obs_features[f"observation.contact_{arm}"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            }

        return obs_mapping, obs_features

    def _process_obs(self, obs: dict, info: dict) -> dict:
        frame = super()._process_obs(obs=obs, info=info)
        robot = self.env.robots[0]

        # Joint efforts — measured torques from the articulation view (N·m)
        frame["observation.joint_efforts"] = robot.get_joint_efforts().float()

        # Per-arm gripper contact flags (non-self contacts only)
        for arm in robot.arm_names:
            contact_paths, _ = robot._find_gripper_contacts(arm=arm)
            frame[f"observation.contact_{arm}"] = th.tensor([float(len(contact_paths) > 0)])

        return frame
