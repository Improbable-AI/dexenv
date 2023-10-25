import numpy as np
import torch
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List

from dexenv.utils.torch_utils import torch_to_np


@dataclass
class StepData:
    ob: Any = None
    state: Any = None
    action: Any = None
    # store action infomation such as log probability, entropy
    action_info: Any = None
    next_ob: Any = None
    next_state: Any = None
    reward: Any = None
    done: Any = None
    info: Any = None
    true_done: Any = None

    def __post_init__(self):
        """
        If the ob is a dict containing keys: ob and state
        then store them into ob and state separately
        """
        if isinstance(self.ob, dict):
            self._split_dict(self.ob, next=False)
        if isinstance(self.next_ob, dict):
            self._split_dict(self.next_ob, next=True)

    def _split_dict(self, data, next=True):
        for key in data.keys():
            prefix = 'next_' if next else ''
            setattr(self, f'{prefix + key}', data[key])

    def get_info_data(self):
        idata = InfoData(info=self.info, action_info=self.action_info)
        return idata

    def split_first_dim(self):
        num_envs = None
        out = []
        for key, value in self.__dict__.items():
            if value is not None and len(value) > 0:
                num_envs = len(value)
                break
        out = [StepData() for i in range(num_envs)]
        for key, value in self.__dict__.items():
            if value is not None and len(value) > 0:
                for i in range(num_envs):
                    setattr(out[i], key, deepcopy(value[i]))
        return out


@dataclass
class StateActionData:
    ob: Any = None
    state: Any = None
    action: Any = None

    def __post_init__(self):
        """
        If the ob is a dict containing keys: ob and state
        then store them into ob and state separately
        """
        if isinstance(self.ob, dict):
            self._split_dict(self.ob)

    def _split_dict(self, data):
        for key in data.keys():
            setattr(self, f'{key}', data[key])

    def split_first_dim(self):
        num_envs = None
        out = []
        for key, value in self.__dict__.items():
            if value is not None and len(value) > 0:
                num_envs = len(value)
                break
        out = [StateActionData() for i in range(num_envs)]
        for key, value in self.__dict__.items():
            if value is not None and len(value) > 0:
                for i in range(num_envs):
                    setattr(out[i], key, deepcopy(value[i]))
        return out


@dataclass
class InfoData:
    info: Any = None
    action_info: Any = None


@dataclass
class TorchTrajectory:
    extra_data: Dict = field(default_factory=dict)
    data: Dict = field(default_factory=dict)
    # if capacity is given, then it will preallocate an array for items in StepData
    capacity: int = 500
    cur_id: int = 0
    list_data: List[InfoData] = field(default_factory=list)
    traj_keys_in_cpu: List = field(default_factory=list)

    def allocate_memory(self, step_data=None, **kwargs):
        sdata_dict = asdict(step_data) if step_data is not None else kwargs
        keys = sdata_dict.keys()
        filtered_keys = [
            x for x in keys if 'info' not in x and sdata_dict[x] is not None]
        for key in filtered_keys:
            val = sdata_dict[key]
            if isinstance(val, dict):
                for vk, vv in val.items():
                    self._allocate_memory_with_key(self.convert_dict_key(key, vk), vv)
            else:
                self._allocate_memory_with_key(key, val)

    def convert_dict_key(self, parent_key, child_key):
        prefix = 'next_' if 'next_' in parent_key else ''
        new_key = f'{prefix + child_key}'
        return new_key

    def _allocate_memory_with_key(self, key, val):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        val_shape = val.shape
        if self.traj_keys_in_cpu is not None and key in self.traj_keys_in_cpu:
            device = 'cpu'
        else:
            device = val.device
        # logger.info(f'Traj key:{key}   device:{device}')
        self.data[key] = torch.zeros((self.capacity,) + val_shape,
                                     device=device,
                                     dtype=val.dtype)

    def __getitem__(self, item):
        step_data = {k: self.data[k][item] for k in self.data.keys()}
        if len(self.list_data) > 0:
            info_data = self.list_data[item]
            step_data['info'] = info_data.info
            step_data['action_info'] = info_data.action_info
        return StepData(**step_data)

    def add(self, step_data=None, **kwargs):
        if len(self.data) < 1:
            self.allocate_memory(step_data, **kwargs)
        if self.cur_id == 0:
            self.list_data.clear()
        if step_data is not None:
            if not isinstance(step_data, StepData):
                raise TypeError('step_data should be an '
                                'instance of StepData!')
            sd = asdict(step_data)

            for key in self.data.keys():
                self._assign_value(key, sd[key], self.cur_id)
            self.list_data.append(step_data.get_info_data())
        else:
            for key in self.data.keys():
                if key in kwargs:
                    val = kwargs.pop(key)
                    if isinstance(val, dict):
                        for kk, vv in val.items():
                            kk = self.convert_dict_key(key, kk)
                            self._assign_value(kk, vv, self.cur_id)
                    else:
                        self._assign_value(key, val, self.cur_id)
            if len(kwargs) > 0:
                info = InfoData(**kwargs)
                self.list_data.append(info)
        self.cur_id = int((self.cur_id + 1) % self.capacity)

    def _assign_value(self, key, val, cur_id):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        self.data[key][cur_id] = val.to(self.data[key].device, non_blocking=True)

    def reset(self):
        for key in self.data.keys():
            self.data[key].zero_()
        self.cur_id = 0
        self.list_data.clear()

    def add_extra(self, key, value):
        self.extra_data[key] = value

    @property
    def obs(self):
        obs = self.data.get('ob', None)
        return obs[:self.capacity]

    @property
    def states(self):
        states = self.data.get('state', None)
        return states[:self.capacity]

    @property
    def actions(self):
        actions = self.data.get('action', None)
        return actions[:self.capacity]

    @property
    def action_infos(self):
        return [data.action_info for idx, data in enumerate(self.list_data) if idx < self.capacity]

    @property
    def next_obs(self):
        next_obs = self.data.get('next_ob', None)
        return next_obs[:self.capacity]

    @property
    def next_states(self):
        next_states = self.data.get('next_state', None)
        return next_states[:self.capacity]

    @property
    def rewards(self):
        rewards = self.data.get('reward', None)
        return rewards[:self.capacity]

    @property
    def true_dones(self):
        true_dones = self.data.get('true_done', None)
        return true_dones[:self.capacity]

    @property
    def dones(self):
        dones = self.data.get('done', None)
        return dones[:self.capacity]

    @property
    def infos(self):
        return [data.info for idx, data in enumerate(self.list_data) if idx < self.capacity]

    @property
    def total_steps(self):
        return len(self.data['action'][0]) * self.capacity

    @property
    def num_envs(self):
        return len(self.data['action'][0])

    @property
    def done_indices(self):
        dids = []
        dones = torch_to_np(self.dones)
        for i in range(dones.shape[1]):
            di = dones[:, i]
            did = []
            if not np.any(di):
                did.append(len(di) - 1)
            else:
                did.extend(np.where(dones[:, i])[0])
            dids.append(did)
        return dids

    @property
    def episode_steps(self):
        steps = []
        dones = torch_to_np(self.dones)
        for i in range(dones.shape[1]):
            di = dones[:, i]
            if not np.any(di):
                steps.append(len(di))
            else:
                did = np.argwhere(di).flatten() + 1
                did = np.insert(did, 0, 0)
                diff = np.diff(did)
                steps.extend(diff.tolist())
        return steps
