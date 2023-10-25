import numpy as np
import torch
from collections import deque
from dataclasses import dataclass
from omegaconf.dictconfig import DictConfig
from typing import Any

from dexenv.utils.common import stack_data
from dexenv.utils.data import TorchTrajectory
from dexenv.utils.info_util import TIMEOUT_KEY
from dexenv.utils.info_util import aggregate_traj_info
from dexenv.utils.info_util import info_has_key
from dexenv.utils.torch_utils import torch_to_np


@dataclass
class BasicRunner:
    agent: Any
    env: Any
    cfg: DictConfig
    eval_env: Any = None
    store_next_ob: bool = True

    def __post_init__(self):
        self.train_env = self.env
        self.num_train_envs = self.env.num_envs
        self.obs = None
        if self.eval_env is None:
            self.eval_env = self.env
        self.train_ep_return = deque(maxlen=self.cfg.alg.deque_size)
        self.train_ep_len = deque(maxlen=self.cfg.alg.deque_size)
        self.train_success = deque(maxlen=self.cfg.alg.deque_size)
        self.save_ob_in_eval = self.cfg.save_ob_in_eval
        self.disable_tqdm = not self.cfg.alg.tqdm
        self.reset_record()

    def __call__(self, **kwargs):
        raise NotImplementedError

    def reset(self, env=None, *args, **kwargs):
        if env is None:
            env = self.train_env
        self.obs = env.reset(*args, **kwargs)
        self.reset_record()

    def reset_record(self):
        self.cur_ep_len = np.zeros(self.num_train_envs)
        self.cur_ep_return = np.zeros(self.num_train_envs)

    def create_traj(self, evaluation=False):
        if evaluation:
            capacity = self.cfg.alg.eval_rollout_steps
        else:
            capacity = self.cfg.alg.train_rollout_steps
        return TorchTrajectory(capacity=capacity,
                               traj_keys_in_cpu=self.cfg.alg.traj_keys_in_cpu)

    def handle_timeout(self, next_ob, done, reward, info, skip_record=False):
        if info_has_key(info, TIMEOUT_KEY, single_info=True):
            time_out = aggregate_traj_info(info, TIMEOUT_KEY, single_info=True)
            episode_done = time_out | done
        else:
            episode_done = done
        if isinstance(episode_done, torch.Tensor):
            done_idx = episode_done.flatten().nonzero(as_tuple=True)[0].cpu().numpy()
        else:
            done_idx = np.argwhere(episode_done).flatten()
        self.cur_ep_len += 1
        if isinstance(info, list) and 'raw_reward' in info[0]:
            self.cur_ep_return += torch_to_np(stack_data([x['raw_reward'] for x in info]))
        else:
            self.cur_ep_return += torch_to_np(reward)
        if len(done_idx) > 0:
            true_next_ob = next_ob
            true_done = done
            if not skip_record:
                self.train_ep_return.extend([self.cur_ep_return[dix] for dix in done_idx])
                self.train_ep_len.extend([self.cur_ep_len[dix] for dix in done_idx])
                success_key = 'success'
                if isinstance(info, dict) and success_key in info:
                    self.train_success.extend([info[success_key][i].cpu() for i in done_idx])
                elif isinstance(info, list) and success_key in info[0]:
                    self.train_success.extend([info[i][success_key].cpu() for i in done_idx])
            self.cur_ep_return[done_idx] = 0
            self.cur_ep_len[done_idx] = 0
        else:
            true_next_ob = next_ob
            true_done = done
        return true_next_ob, true_done, done_idx, reward
