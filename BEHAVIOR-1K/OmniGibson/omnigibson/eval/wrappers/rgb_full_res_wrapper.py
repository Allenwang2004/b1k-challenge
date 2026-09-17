from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.eval.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    get_robot_camera_names,
    set_sensor_modalities,
)

logger = create_module_logger(module_name=__name__)


class RGBFullResWrapper(EnvironmentWrapper):
    """
    RGB-only cameras at data-collection resolution (head 720x720, wrists 480x480).

    Same as RGBDFullResWrapper without the depth modality: the policy receives the same RGB frames the
    demos were recorded at (and resizes them itself), and rollout videos are written at native resolution
    instead of the upscaled 224px frames DefaultWrapper produces.

    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        robot = env.robots[0]
        robot_eval_config = getattr(env, "_eval_robot_config", {})
        camera_roles_by_sensor_name = {
            camera_name.split("::")[1]: camera_id
            for camera_id, camera_name in get_robot_camera_names(robot.name, robot_eval_config).items()
        }
        for sensor_name, sensor in robot.sensors.items():
            if not hasattr(sensor, "image_height") or not hasattr(sensor, "image_width"):
                continue
            set_sensor_modalities(sensor, {"rgb"})
            camera_id = camera_roles_by_sensor_name.get(sensor_name)
            if camera_id == "head":
                sensor.image_height = HEAD_RESOLUTION[0]
                sensor.image_width = HEAD_RESOLUTION[1]
            else:
                sensor.image_height = WRIST_RESOLUTION[0]
                sensor.image_width = WRIST_RESOLUTION[1]
            # Update only this sensor's space (as DefaultWrapper does): env.load_observation_space() would
            # also recompute the robot proprioception space, which needs the articulation view that does not
            # exist yet when the wrapper is instantiated by omnigibson.eval.eval.
            sensor_space = sensor.load_observation_space()
            if env.observation_space is not None:
                env.observation_space.spaces[robot.name].spaces[sensor_name] = sensor_space
        logger.info("Reloaded camera observation spaces (RGB, full resolution)!")
