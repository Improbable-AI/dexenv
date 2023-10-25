from loguru import logger

from dexenv.envs import env_name_to_path
from dexenv.utils.common import load_class_from_path


def create_task_env(cfg, **kwargs):
    """
    Creates the task from configurations
    """
    sim_device = cfg.task.sim.device
    sim_rl_device = cfg.task.sim.rl_device

    split_device = sim_device.split(":")
    device_id = int(split_device[1]) if len(split_device) > 1 else 0
    graphics_device_id = device_id

    env_class = load_class_from_path(cfg.task.env.name, env_name_to_path[cfg.task.env.name])
    logger.info(f'Creating environment {cfg.task.env.name}')
    env = env_class(cfg.task,
                    sim_device=sim_device,
                    rl_device=sim_rl_device,
                    graphics_device_id=graphics_device_id,
                    **kwargs)
    return env
