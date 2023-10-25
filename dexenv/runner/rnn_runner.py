import numpy as np
import time
import torch
from copy import deepcopy
from tqdm import tqdm

from dexenv.runner.base_runner import BasicRunner
from dexenv.utils.torch_utils import detach_tensors
from dexenv.utils.torch_utils import reset_hidden_state_at_done


class RNNRunner(BasicRunner):
    def __init__(self, *args, **kwargs):
        super(RNNRunner, self).__init__(*args, **kwargs)
        self.hidden_states = None
        self.hidden_state_shape = None

    @torch.no_grad()
    def __call__(self, time_steps, sample=True, evaluation=False,
                 return_on_done=False, render=False,
                 sleep_time=0, reset_first=False,
                 reset_kwargs=None, action_kwargs=None,
                 get_last_val=False):
        traj = self.create_traj(evaluation)
        if reset_kwargs is None:
            reset_kwargs = {}
        if action_kwargs is None:
            action_kwargs = {}
        if evaluation:
            env = self.eval_env
        else:
            env = self.train_env
        if self.obs is None or reset_first or evaluation:
            self.reset(**reset_kwargs)
        ob = self.obs
        hidden_state = self.hidden_states

        if return_on_done:
            all_dones = np.zeros(env.num_envs, dtype=bool)
        else:
            all_dones = None

        for t in tqdm(range(time_steps), desc='Step', disable=self.disable_tqdm):
            if render:
                env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            action, action_info, hidden_state = self.agent.get_action(ob,
                                                                      sample=sample,
                                                                      hidden_state=hidden_state,
                                                                      get_action_only=evaluation,
                                                                      **action_kwargs)
            if self.hidden_state_shape is None and not evaluation:
                self.get_hidden_state_shape(hidden_state)

            next_ob, reward, done, info = env.step(action)
            next_ob = deepcopy(next_ob)
            done = deepcopy(done)

            true_next_ob, true_done, done_idx, reward = self.handle_timeout(next_ob,
                                                                            done,
                                                                            reward,
                                                                            info,
                                                                            skip_record=evaluation)
            if done_idx.size > 0:
                if all_dones is not None:
                    all_dones[done_idx] = True

            data_to_store = dict(
                action=action,
                action_info=deepcopy(action_info),
                reward=deepcopy(reward),
                true_done=true_done,
                info=deepcopy(info),
                done=done,
            )
            if not evaluation or (evaluation and self.save_ob_in_eval):
                data_to_store['ob'] = ob
                if self.store_next_ob:
                    data_to_store['next_ob'] = true_next_ob
            traj.add(**data_to_store)
            ob = next_ob
            if return_on_done and np.all(all_dones):
                break

            if get_last_val and not evaluation and t == time_steps - 1 and hasattr(self.agent, 'get_val'):
                last_val, _ = self.agent.get_val(true_next_ob,
                                                 hidden_state=hidden_state)
                traj.add_extra('last_val', last_val.detach())
            hidden_state = reset_hidden_state_at_done(hidden_state, done=done)
        self.obs = ob if not evaluation else None
        self.hidden_states = detach_tensors(hidden_state) if not evaluation else None
        return traj

    def reset(self, env=None, *args, **kwargs):
        super().reset(env, *args, **kwargs)
        self.hidden_states = None

    def get_hidden_state_shape(self, hidden_state):
        if isinstance(hidden_state, tuple) or isinstance(hidden_state, list):
            self.hidden_state_shape = tuple([self.get_tensor_shape(x) for x in hidden_state])
        else:
            self.hidden_state_shape = self.get_tensor_shape(hidden_state)
        return self.hidden_state_shape

    def get_tensor_shape(self, tensor):
        if tensor is None:
            return None
        else:
            return tensor.shape
