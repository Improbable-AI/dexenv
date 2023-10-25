import numpy as np
import torch
from collections.abc import Sequence
from copy import deepcopy

from dexenv.utils.common import stack_data

TIMEOUT_KEY = 'TimeLimit.truncated'


def get_element_from_traj_infos(infos, key, i, j):
    if isinstance(infos[0], Sequence):
        if key is None:
            return infos[i][j]
        else:
            return infos[i][j][key]
    elif isinstance(infos[0], dict):
        if key is None:
            keys = infos[i].keys()
            out = dict()
            for key in keys:
                try:
                    out[key] = infos[i][key][j]
                except:
                    continue
            return out
        else:
            return infos[i][key][j]
    else:
        raise NotImplementedError


def get_element_from_traj_infos_row_ids(infos, key, row_ids):
    if isinstance(infos[0], Sequence):
        out = []
        for col_id in range(len(row_ids)):
            for row_id in row_ids[col_id]:
                if key in infos[row_id][col_id]:
                    out.append(torch_to_np(infos[row_id][col_id][key]))
        return out
    elif isinstance(infos[0], dict):
        out = []
        for col_id in range(len(row_ids)):
            for row_id in row_ids[col_id]:
                if key in infos[row_id]:
                    out.append(torch_to_np(infos[row_id][key][col_id]))
        return out
    else:
        raise NotImplementedError


def torch_to_np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    else:
        return tensor.cpu().detach().numpy()


def info_has_key(infos, key, single_info=False):
    if not single_info:
        infos = infos[0]
    if isinstance(infos, Sequence):
        return key in infos[0]
    elif isinstance(infos, dict):
        return key in infos
    else:
        raise NotImplementedError


def aggregate_traj_info(infos, key, single_info=False):
    if single_info:
        infos = [infos]
    if isinstance(infos[0], Sequence):
        out = []
        for info in infos:
            time_out = []
            for env_info in info:
                time_out.append(env_info[key])
            out.append(np.stack(time_out))
        out = stack_data(out)
    elif isinstance(infos[0], dict):
        out = []
        for info in infos:
            tensor = info[key]
            out.append(tensor)
        out = stack_data(out)
    else:
        raise NotImplementedError
    if single_info:
        out = out.squeeze(0)
    return out


def add_data_to_info(info, key, data):
    if isinstance(info, Sequence):
        for ditem, inf in zip(data, info):
            inf[key] = deepcopy(ditem)
    elif isinstance(info, dict):
        info[key] = data
    else:
        raise NotImplementedError


def get_key_from_info(info, key, env_id):
    if isinstance(info, Sequence):
        if key in info[env_id]:
            return info[env_id][key]
        else:
            return None
    elif isinstance(info, dict):
        if key in info:
            return info[key][env_id]
        else:
            return None
    else:
        raise NotImplementedError
