import isaacgym
import os
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pathlib import Path

from .utils.common import generate_id
from .utils.common import make_dir
from .utils.os_utils import get_env

LIB_PATH = Path(__file__).resolve().parent

imp_str = '/abcdefg'

PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT", imp_str))
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = LIB_PATH
    os.environ['PROJECT_ROOT'] = PROJECT_ROOT.as_posix()

HYDRA_OUT_ROOT: Path = Path(get_env("HYDRA_OUT_ROOT", imp_str))

if HYDRA_OUT_ROOT.as_posix() == imp_str:
    HYDRA_OUT_ROOT = LIB_PATH
    os.environ['HYDRA_OUT_ROOT'] = HYDRA_OUT_ROOT.as_posix()
if not HYDRA_OUT_ROOT.exists():
    make_dir(HYDRA_OUT_ROOT)

import hydra

try:
    OmegaConf.register_new_resolver('uuid', lambda: generate_id())
except:
    pass
