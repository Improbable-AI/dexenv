import numpy as np
import platform
import time
import torch
import wandb
from collections.abc import Sequence
from dataclasses import dataclass
from loguru import logger
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import BatchSampler
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from typing import Any

from dexenv.utils.common import get_list_stats
from dexenv.utils.common import smooth_value
from dexenv.utils.dataset import DictDataset
from dexenv.utils.dataset import TrajDatasetSplitByDone
from dexenv.utils.hydra_util import get_hydra_run_dir
from dexenv.utils.info_util import aggregate_traj_info
from dexenv.utils.info_util import get_element_from_traj_infos
from dexenv.utils.info_util import get_element_from_traj_infos_row_ids
from dexenv.utils.torch_utils import torch_to_np


@dataclass
class BaseEngine:
    agent: Any
    runner: Any
    cfg: DictConfig

    def __post_init__(self):
        self.cur_step = 0
        self._best_eval_ret = -np.inf
        self._best_train_ret = -np.inf
        self._best_eval_success_rate = -np.inf
        self._best_train_success_rate = -np.inf
        self._eval_is_best = False
        self._train_is_best = False

        self.smooth_eval_return = None
        self.smooth_train_return = None
        self.smooth_tau = self.cfg.alg.smooth_eval_tau
        self.optim_stime = None

        wandb_cfg = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
        if self.cfg.test or self.cfg.resume:
            if self.cfg.resume_id is None:
                raise ValueError('Please specify the run ID to be resumed!')
            run_id = self.cfg.resume_id.split(':')[0]
            if '/' in run_id and not self.cfg.resume_to_diff_id:
                run_id_split = run_id.split('/')
                run_id = run_id_split[-1]
                self.cfg.logging.wandb.project = run_id_split[-2]

            if self.cfg.resume_to_diff_id:
                run_id = get_hydra_run_dir().name
            if self.cfg.resume:
                wandb_cfg['hostname'] = platform.node()
            self.wandb_runs = wandb.init(**self.cfg.logging.wandb,
                                         resume='allow', id=run_id,
                                         config=wandb_cfg if self.cfg.resume else None,
                                         )
        else:
            wandb_cfg['hostname'] = platform.node()
            wandb_kwargs = self.cfg.logging.wandb
            wandb_tags = wandb_kwargs.get('tags', None)
            if wandb_tags is not None and isinstance(wandb_tags, str):
                wandb_kwargs['tags'] = [wandb_tags]
            self.wandb_runs = wandb.init(**wandb_kwargs, config=wandb_cfg,
                                         # id=wandb_id, name=wandb_id,
                                         # settings=wandb.Settings(start_method="thread"),
                                         )
        logger.warning(f'Wandb run dir:{self.wandb_runs.dir}')
        logger.warning(f'      Project name:{self.wandb_runs.project_name()}')

        if (self.cfg.test or self.cfg.resume) and not self.cfg.test_pretrain:
            self.cur_step = self.agent.load_model(eval=self.cfg.test and self.cfg.test_eval_best,
                                                  pretrain_model=self.cfg.alg.pretrain_model)
            if self.cfg.resume_to_diff_id:
                self.cur_step = 0
        else:
            if self.cfg.alg.pretrain_model is not None:
                self.agent.load_model(pretrain_model=self.cfg.alg.pretrain_model)

    def train(self, **kwargs):
        raise NotImplementedError

    def rollout_once(self, *args, **kwargs):
        t0 = time.perf_counter()
        self.agent.eval_mode()
        traj = self.runner(**kwargs)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        return traj, elapsed_time

    def do_eval(self, det=False, sto=False):
        eval_log_info = dict()
        det = det or self.cfg.alg.det_eval
        sto = sto or self.cfg.alg.sto_eval
        if det:
            det_log_info, _ = self.eval(eval_num=self.cfg.test_num,
                                        sample=False, smooth=True)
            det_log_info = {f'det/{k}': v for k, v in det_log_info.items()}
            eval_log_info.update(det_log_info)
        if sto:
            sto_log_info, _ = self.eval(eval_num=self.cfg.test_num,
                                        sample=True, smooth=False)

            sto_log_info = {f'sto/{k}': v for k, v in sto_log_info.items()}
            eval_log_info.update(sto_log_info)
        if len(eval_log_info) > 0:
            wandb.log(eval_log_info, step=self.cur_step)
        if self._eval_is_best and self.cfg.save_eval_ckpt:
            self.agent.save_model(is_best=True,
                                  step=self.cur_step,
                                  eval=True,
                                  wandb_run=self.wandb_runs)

    @torch.no_grad()
    def eval(self, render=False, eval_num=1,
             sleep_time=0, sample=True, no_tqdm=None):
        t0 = time.perf_counter()
        time_steps = []
        rets = []
        lst_step_infos = []
        successes = []
        check_dist = None
        dists = []
        if no_tqdm is not None:
            disable_tqdm = bool(no_tqdm)
        else:
            disable_tqdm = not self.cfg.test
        for idx in tqdm(range(eval_num), disable=disable_tqdm):
            traj, _ = self.rollout_once(time_steps=self.cfg.alg.eval_rollout_steps,
                                        return_on_done=False,
                                        sample=self.cfg.alg.sample_action and sample,
                                        render=render,
                                        sleep_time=sleep_time,
                                        evaluation=True)

            done_indices = traj.done_indices
            if hasattr(traj, 'raw_rewards'):
                rewards = traj.raw_rewards
            else:
                rewards = traj.rewards
            infos = traj.infos

            for eid in range(len(done_indices)):
                done_ids = done_indices[eid]
                start_id = 0
                for ejd in range(len(done_ids)):
                    end_id = done_ids[ejd] + 1
                    ret = np.sum(torch_to_np(rewards[start_id:end_id, eid]))
                    start_id = end_id
                    rets.append(ret)
                    lst_step_infos.append(get_element_from_traj_infos(infos=infos, key=None,
                                                                      i=end_id - 1, j=eid))
            episode_steps = traj.episode_steps
            time_steps.extend(episode_steps)

            successes.extend(get_element_from_traj_infos_row_ids(infos=infos, key='success',
                                                                 row_ids=done_indices))
        raw_traj_info = {'return': rets,
                         'lst_step_info': lst_step_infos}
        if isinstance(time_steps[0], list):
            raw_traj_info['episode_length'] = np.concatenate(time_steps)
        else:
            raw_traj_info['episode_length'] = np.asarray(time_steps)
        if check_dist:
            raw_traj_info['abs_dist'] = np.concatenate(dists)
        log_info = dict()
        for key, val in raw_traj_info.items():
            if 'info' in key:
                continue
            val_stats = get_list_stats(val)
            for sk, sv in val_stats.items():
                log_info['eval/' + key + '/' + sk] = sv
        if len(successes) > 0:
            log_info['eval/success'] = np.mean(successes)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        log_info['eval/eval_time'] = elapsed_time
        return log_info

    def get_train_log(self, optim_infos, traj=None):
        log_info = dict()
        vector_keys = set()
        scalar_keys = set()
        for oinf in optim_infos:
            for key in oinf.keys():
                if 'vec_' in key:
                    vector_keys.add(key)
                else:
                    scalar_keys.add(key)

        for key in scalar_keys:
            log_info[key] = np.mean([inf[key] for inf in optim_infos if key in inf])

        for key in vector_keys:
            k_stats = get_list_stats([inf[key] for inf in optim_infos if key in inf])
            for sk, sv in k_stats.items():
                log_info[f'{key}/' + sk] = sv
        if traj is not None:
            actions_stats = get_list_stats(torch_to_np(traj.actions))
            for sk, sv in actions_stats.items():
                log_info['rollout_action/' + sk] = sv
            log_info['rollout_steps_per_iter'] = traj.total_steps

            ep_returns_stats = get_list_stats(self.runner.train_ep_return)
            for sk, sv in ep_returns_stats.items():
                log_info['episode_return/' + sk] = sv
            ep_len_stats = get_list_stats(self.runner.train_ep_len)
            for sk, sv in ep_len_stats.items():
                log_info['episode_length/' + sk] = sv
            if len(self.runner.train_success) > 0:
                log_info['episode_success'] = np.mean(self.runner.train_success)

            if 'episode_return/mean' in log_info:
                self.smooth_train_return = smooth_value(log_info['episode_return/mean'],
                                                        self.smooth_train_return,
                                                        self.smooth_tau)
                log_info['smooth_return/mean'] = self.smooth_train_return

                if self.cfg.save_best_on_success and len(self.runner.train_success) > 0:
                    self._train_is_best, self._best_train_success_rate = self._check_ckpt_is_best(
                        log_info['episode_success'],
                        best_history_val=self._best_train_success_rate,
                        bigger_is_better=True
                    )
                else:
                    self._train_is_best, self._best_train_ret = self._check_ckpt_is_best(self.smooth_train_return,
                                                                                         best_history_val=self._best_train_ret,
                                                                                         bigger_is_better=True)

            ex_info = traj[0].info
            if isinstance(ex_info, Sequence):
                traj_info_keys = ex_info[0].keys()
            else:  # dict
                traj_info_keys = traj[0].info.keys()
            reward_keys = [x for x in traj_info_keys if 'reward' in x]

            if len(reward_keys) > 0:
                traj_infos = traj.infos
                for rkey in reward_keys:
                    rlist = aggregate_traj_info(traj_infos, rkey)
                    rlist_stats = get_list_stats(rlist)
                    for rk, rv in rlist_stats.items():
                        log_info[f'{rkey}/{rk}'] = rv
        log_info['rollout_steps'] = self.cur_step
        train_log_info = dict()
        for key, val in log_info.items():
            train_log_info['train/' + key] = val
        return train_log_info

    def get_dataloader(self, dataset, batch_size):
        subset = SubsetRandomSampler(range(len(dataset)))
        batch_sampler = BatchSampler(subset,
                                     batch_size=batch_size,
                                     drop_last=False,
                                     )
        inds = list(batch_sampler)
        dataloader = []
        for ind in inds:
            sub_data = dict()
            if isinstance(dataset, DictDataset):
                for keys in dataset.data.keys():
                    sub_data[keys] = dataset.data[keys][ind]
            elif isinstance(dataset, TrajDatasetSplitByDone):
                for keys in dataset.data.keys():
                    if keys == 'time_steps':
                        sub_data[keys] = torch.tensor(dataset.data[keys])[ind]
                    else:
                        sub_data[keys] = dataset.data[keys][ind]
            dataloader.append(sub_data)
        return dataloader

    def _check_ckpt_is_best(self, cur_val, best_history_val, bigger_is_better=True):
        is_best = cur_val > best_history_val if bigger_is_better else cur_val < best_history_val
        if is_best:
            best_history_val = cur_val
        return is_best, best_history_val

    def _get_batch_size(self, dataset):
        if hasattr(self.cfg.alg, 'num_batches') and self.cfg.alg.num_batches is not None:
            batch_size = max(1, int(len(dataset) / self.cfg.alg.num_batches))
        else:
            batch_size = self.cfg.alg.batch_size
        return batch_size
