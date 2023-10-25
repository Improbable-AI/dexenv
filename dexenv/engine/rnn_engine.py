import numpy as np
import time
import torch
from dataclasses import dataclass
from itertools import count
from tqdm import tqdm

from dexenv.engine.base_engine import BaseEngine
from dexenv.utils.common import get_list_stats
from dexenv.utils.common import smooth_value
from dexenv.utils.common import stack_data
from dexenv.utils.common import stat_for_traj_data
from dexenv.utils.dataset import DictDataset
from dexenv.utils.info_util import aggregate_traj_info
from dexenv.utils.info_util import get_element_from_traj_infos
from dexenv.utils.info_util import get_element_from_traj_infos_row_ids
from dexenv.utils.info_util import info_has_key
from dexenv.utils.torch_utils import swap_axes
from dexenv.utils.torch_utils import torch_to_np


def extract_traj_data_rnn(cfg, states, data, act_dim=9):
    states_ori = states
    if cfg.vision.pred_rot_dist:
        if cfg.task.env.robot == 'dclaw_4f':
            quat_diff = states[..., 41:45]
        else:
            raise NotImplementedError
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[..., 0:3], p=2, dim=-1), max=1.0))
        data['rot_dist'] = rot_dist

    if cfg.vision.act_in:
        states = states_ori
        prev_actions = states[..., -act_dim:]
        data['prev_actions'] = prev_actions


@dataclass
class RNNEngine(BaseEngine):
    act_dim: int = 12

    def train(self):

        for iter_t in count():
            if iter_t % self.cfg.logging.eval_interval == 0 and self.cfg.alg.run_eval:
                self.do_eval()
            traj, rollout_time = self.rollout_once(sample=True,
                                                   get_last_val=False,
                                                   reset_first=True,
                                                   time_steps=self.cfg.alg.train_rollout_steps)
            optim_infos = self.train_once(traj)
            if iter_t % self.cfg.logging.log_interval == 0:
                t1 = time.perf_counter()
                train_log_info = self.get_train_log(optim_infos, traj)
                train_log_info['train/optim_time'] = t1 - self.optim_stime
                train_log_info['train/rollout_time'] = rollout_time

            if self.cur_step > self.cfg.alg.max_steps:
                break
            if iter_t % self.cfg.logging.ckpt_interval == 0:
                self.agent.save_model(is_best=self._train_is_best,
                                      step=self.cur_step,
                                      wandb_run=self.wandb_runs,
                                      eval=False)

    @torch.no_grad()
    def traj_preprocess(self, traj):
        action_infos = traj.action_infos
        exp_act_loc = stack_data([ainfo['exp_act_loc'] for ainfo in action_infos])
        exp_act_scale = stack_data([ainfo['exp_act_scale'] for ainfo in action_infos])
        hidden_state = action_infos[0]['in_hidden_state']
        if hidden_state is not None:
            hidden_state = swap_axes(hidden_state, (0, 1))
        else:
            hidden_state = torch.zeros((exp_act_loc.shape[1],  # batch_size
                                        1,  # num_layers*num_directions
                                        256))  # hidden state dimension

        data = dict(
            ob=swap_axes(traj.obs, (0, 1)),
            state=swap_axes(traj.states, (0, 1)),
            exp_act_loc=swap_axes(exp_act_loc, (0, 1)),
            exp_act_scale=swap_axes(exp_act_scale, (0, 1)),
            hidden_state=hidden_state,
            done=swap_axes(traj.dones, (0, 1)),
        )

        states = swap_axes(traj.states, (0, 1))
        extract_traj_data_rnn(self.cfg, states, data, act_dim=self.act_dim)
        rollout_dataset = DictDataset(**data)
        batch_size = self._get_batch_size(rollout_dataset)
        dataloader = self.get_dataloader(rollout_dataset,
                                         batch_size)
        return dataloader

    def train_once(self, traj):
        self.optim_stime = time.perf_counter()
        self.cur_step += traj.total_steps
        rollout_dataloader = self.traj_preprocess(traj)
        optim_infos = []
        for oe in range(self.cfg.alg.opt_epochs):
            for batch_ndx, batch_data in enumerate(rollout_dataloader):
                if self.cfg.vision.optim_empty_cache:
                    torch.cuda.empty_cache()
                optim_info = self.agent.optimize(batch_data, batch_id=batch_ndx)
                optim_infos.append(optim_info)
        self.agent.clear_optim_grad()
        return optim_infos

    @torch.no_grad()
    def eval(self, render=False, eval_num=1,
             sleep_time=0, sample=True, smooth=True,
             no_tqdm=None,
             return_on_done=False):
        t0 = time.perf_counter()
        time_steps = []
        rets = []
        lst_step_infos = []
        successes = []
        abs_dists = []
        check_dist = None
        dists = []
        if no_tqdm is not None:
            disable_tqdm = bool(no_tqdm)
        else:
            disable_tqdm = not self.cfg.test

        for idx in tqdm(range(eval_num), disable=disable_tqdm):
            kwargs = dict(
                time_steps=self.cfg.alg.eval_rollout_steps,
                return_on_done=return_on_done,
                sample=self.cfg.alg.sample_action and sample,
                render=render,
                sleep_time=sleep_time,
                evaluation=True,
            )
            traj, _ = self.rollout_once(**kwargs)
            done_indices = traj.done_indices
            if hasattr(traj, 'raw_rewards'):
                rewards = traj.raw_rewards
            else:
                rewards = traj.rewards
            infos = traj.infos
            if check_dist is None:
                check_dist = self.cfg.test and info_has_key(infos, 'abs_dist')

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
            abs_dists.extend(get_element_from_traj_infos_row_ids(infos=infos, key='abs_dist',
                                                                 row_ids=done_indices))
            if check_dist:
                dist_tensor = aggregate_traj_info(infos, 'abs_dist')
                traj_dists = stat_for_traj_data(torch_to_np(dist_tensor), dones=torch_to_np(traj.dones), op='min')
                dists.append(traj_dists)

        raw_traj_info = {'return': rets,
                         'episode_length': np.array(time_steps),
                         'lst_step_info': lst_step_infos}
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
        if len(abs_dists) > 0:
            dist_stats = get_list_stats(abs_dists)
            for sk, sv in dist_stats.items():
                log_info['eval/abs_dist/' + sk] = sv
            abs_dists = np.array(abs_dists)
            tol = self.cfg.task.env.rew.successTolerance
            for i in range(5):
                tol_tmp = tol * (i + 1)
                if tol_tmp > 0.8:
                    break
                percent = np.mean(abs_dists < tol_tmp)
                log_info[f'eval/tol_succ/tol_{tol_tmp}'] = percent

        if smooth:
            self.smooth_eval_return = smooth_value(log_info['eval/return/mean'],
                                                   self.smooth_eval_return,
                                                   self.smooth_tau)
            log_info['eval/smooth_return/mean'] = self.smooth_eval_return
            if self.cfg.save_best_on_success and len(successes) > 0:
                self._eval_is_best, self._best_eval_success_rate = self._check_ckpt_is_best(log_info['eval/success'],
                                                                                            best_history_val=self._best_eval_success_rate,
                                                                                            bigger_is_better=True)
            else:
                self._eval_is_best, self._best_eval_ret = self._check_ckpt_is_best(self.smooth_eval_return,
                                                                                   best_history_val=self._best_eval_ret,
                                                                                   bigger_is_better=True)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        log_info['eval/eval_time'] = elapsed_time
        return log_info, raw_traj_info
