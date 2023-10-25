import numpy as np
import operator
import time
import torch
from bisect import bisect
from isaacgym import gymapi
from numpy.random import default_rng

from dexenv.utils.constants import RandKey


def add_noise_to_var(tensor, noise_cfg, rng=None):
    if isinstance(tensor, np.ndarray):
        noise = get_rand_sample(noise_cfg, tensor.shape, rng=rng)
    else:
        noise = get_rand_sample_torch(noise_cfg, tensor.shape, device=tensor.device)
    op_type = noise_cfg["operation"]
    op = operator.add if op_type == 'additive' else operator.mul
    new_tensor = op(tensor, noise)
    return new_tensor


def add_noise_to_envs(env_ids, all_env_ptrs, ref_props,
                      handle_dict, prop_name,
                      attr, attr_rand_cfg, param_getters_map,
                      param_setters_map, param_setter_defaults_map,
                      sample=None):
    sample_id = 0
    setter = param_setters_map[prop_name]
    default_args = param_setter_defaults_map[prop_name]
    if prop_name == RandKey.dof_prop.value:
        ref_val = np.array([ref_props[env_id][prop_name][attr] for env_id in env_ids]).reshape(sample.shape)
    else:
        ref_val = np.array([x[attr] for env_id in env_ids for x in ref_props[env_id][prop_name]]).reshape(sample.shape)
    sample = process_sample_val(ref_val, sample, attr_rand_cfg)

    for env_id in env_ids:
        env_ptr = all_env_ptrs[env_id]
        handle = handle_dict[env_id]
        prop = param_getters_map[prop_name](env_ptr, handle)
        ref_prop = ref_props[env_id][prop_name]
        if isinstance(prop, list):
            for idx, (p, rp) in enumerate(zip(prop, ref_prop)):
                add_noise_to_prop(p, rp, attr, attr_rand_cfg, sample=sample[sample_id], need_operation=False)
                sample_id += 1
        else:
            add_noise_to_prop(prop, ref_prop, attr, attr_rand_cfg, sample=sample[sample_id], need_operation=False)
            sample_id += 1
        setter(env_ptr, handle, prop, *default_args)


def add_noise_to_prop(prop, ref_prop, attr, attr_rand_cfg, sample=None, need_operation=True):
    if isinstance(prop, gymapi.SimParams):
        if attr == 'gravity':
            if sample is None:
                sample = get_rand_sample(attr_rand_cfg, 3)
            if attr_rand_cfg['operation'] == 'scaling':
                prop.gravity.x = ref_prop['gravity'].x * sample[0]
                prop.gravity.y = ref_prop['gravity'].y * sample[1]
                prop.gravity.z = ref_prop['gravity'].z * sample[2]
            elif attr_rand_cfg['operation'] == 'additive':
                prop.gravity.x = ref_prop['gravity'].x + sample[0]
                prop.gravity.y = ref_prop['gravity'].y + sample[1]
                prop.gravity.z = ref_prop['gravity'].z + sample[2]
    else:
        if sample is None:
            need_operation = True
            if isinstance(prop, np.ndarray):
                sample = get_rand_sample(attr_rand_cfg, prop[attr].shape)
            else:
                sample = get_rand_sample(attr_rand_cfg, 1)
        if need_operation:
            new_prop_val = process_sample_val(ref_prop[attr], sample, attr_rand_cfg)
        else:
            new_prop_val = sample
        if isinstance(prop, np.ndarray):
            prop[attr] = new_prop_val
        else:
            setattr(prop, attr, new_prop_val)
    return sample


def process_sample_val(ref_val, sample, attr_rand_cfg):
    if attr_rand_cfg['operation'] == 'scaling':
        new_prop_val = ref_val * sample
    elif attr_rand_cfg['operation'] == 'additive':
        new_prop_val = ref_val + sample
    if 'num_buckets' in attr_rand_cfg and attr_rand_cfg['num_buckets'] > 0:
        new_prop_val = get_bucketed_val(new_prop_val, attr_rand_cfg)
    return new_prop_val


def get_rand_sample(noise_cfg, shape, rng=None, scale=1):
    dist = noise_cfg["distribution"]
    rng = default_rng() if rng is None else rng
    if dist == 'gaussian':
        mu, std = noise_cfg['range']
        if scale != 1:
            std *= scale
        sample = rng.normal(mu, std, shape)
    elif dist == 'uniform':
        lo, hi = noise_cfg['range']
        lo, hi = scale_low_high_bound(low=lo, high=hi, scale=scale)
        sample = rng.uniform(lo, hi, shape)
    elif dist == 'loguniform':
        lo, hi = noise_cfg['range']
        lo = np.log(lo)
        hi = np.log(hi)
        lo, hi = scale_low_high_bound(low=lo, high=hi, scale=scale)
        sample = np.exp(rng.uniform(lo, hi, shape))
    else:
        raise NotImplementedError
    return sample


def scale_low_high_bound(low, high, scale=1):
    if scale == 1:
        return low, high
    else:
        interval = high - low
        mid = (low + high) / 2.0
        scaled_interval = interval * scale
        half_scaled_interval = scaled_interval / 2.0
        low = mid - half_scaled_interval
        high = mid + half_scaled_interval
        return low, high


@torch.no_grad()
def get_rand_sample_torch(noise_cfg, shape, device):
    dist = noise_cfg["distribution"]
    if dist == 'gaussian':
        mu, std = noise_cfg['range']
        sample = torch.normal(mu, std, size=shape, device=device)
    elif dist == 'uniform':
        lo, hi = noise_cfg['range']
        sample = torch.rand(shape, device=device) * (hi - lo) + lo
    elif dist == 'loguniform':
        lo, hi = noise_cfg['range']
        lo = np.log(lo)
        hi = np.log(hi)
        sample = torch.exp(torch.rand(shape, device=device) * (hi - lo) + lo)
    else:
        raise NotImplementedError
    return sample


def get_bucketed_val(val, attr_randomization_params):
    if attr_randomization_params['distribution'] == 'uniform':
        lo, hi = attr_randomization_params['range'][0], attr_randomization_params['range'][1]
    else:
        lo = attr_randomization_params['range'][0] - 2 * attr_randomization_params['range'][1]
        hi = attr_randomization_params['range'][0] + 2 * attr_randomization_params['range'][1]
    num_buckets = attr_randomization_params['num_buckets']
    return bucket_val(val, lo, hi, num_buckets)


def bucket_val(val, lo, hi, num_buckets):
    lo_diff = val - lo
    interval = (hi - lo) / num_buckets
    return lo + (lo_diff // interval) * interval
