from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES, HEAD_RESOLUTION, WRIST_RESOLUTION
from omnigibson.sensors import VisionSensor
from omnigibson.utils.ui_utils import create_module_logger


# Create module logger
log = create_module_logger("WBVIMAWrapper")


class WBVIMAWrapper(EnvironmentWrapper):
    """
    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        # Here, we modify the robot observation to include depth_linear
        # For a complete list of available modalities, see VisionSensor.ALL_MODALITIES
        robot = env.robots[0]
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            robot.sensors[sensor_name].add_modality("depth_linear")
            if camera_id == "head":
                robot.sensors[sensor_name].horizontal_aperture = 40.0
                robot.sensors[sensor_name].image_height = HEAD_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = HEAD_RESOLUTION[1]
            else:
                robot.sensors[sensor_name].image_height = WRIST_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = WRIST_RESOLUTION[1]
        # reload observation space
        env.load_observation_space()
        # we also set task to include obs
        env.task._include_obs = True

    def step(self, action, n_render_iterations=1):
        """
        By default, run the normal environment step() function

        Args:
            action (th.tensor): action to take in environment
            n_render_iterations (int): Number of rendering iterations to use before returning observations

        Returns:
            4-tuple:
                - (dict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is terminated
                - (bool) whether the current episode is truncated
                - (dict) misc information
        """
        obs, reward, terminated, truncated, info = self.env.step(action, n_render_iterations=n_render_iterations)
        # Now, query for some additional privileged task info
        obs["task"] = self.env.task.get_obs(self.env)
        return obs, reward, terminated, truncated, info

    def reset(self):
        # Note that we need to also add additional observations in reset() because the returned observation will be passed into policy
        ret = self.env.reset()
        ret[0]["task"] = self.env.task.get_obs(self.env)
        return ret
