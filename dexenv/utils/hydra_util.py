import os
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from dexenv.utils.common import pathlib_file


def get_hydra_run_dir(pathlib=True):
    run_dir = os.getcwd()
    if pathlib:
        run_dir = pathlib_file(run_dir)
    return run_dir


def override_hydra_cfg(cfg, overrides):
    config_name = HydraConfig.get().job.config_name
    cfg = compose(config_name, overrides=overrides, return_hydra_config=True)
    return cfg
