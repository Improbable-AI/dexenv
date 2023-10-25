import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.gymutil import get_bucketed_val
from loguru import logger

from dexenv.utils.common import get_all_files_with_suffix


@torch.no_grad()
def load_an_object_asset(gym, sim, asset_root, object_urdf, asset_options=None, vhacd=True):
    if asset_options is None:
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.override_inertia = True
        # asset_options.override_com = True
        if vhacd:
            asset_options.convex_decomposition_from_submeshes = True
    rela_file = object_urdf.relative_to(asset_root).as_posix()
    obj_asset = gym.load_asset(sim,
                               asset_root.as_posix(),
                               rela_file,
                               asset_options)
    return obj_asset


@torch.no_grad()
def load_a_goal_object_asset(gym, sim, asset_root, object_urdf, asset_options=None, vhacd=True):
    if asset_options is None:
        asset_options = gymapi.AssetOptions()
        if vhacd:
            asset_options.convex_decomposition_from_submeshes = True
        asset_options.thickness = 0.001
        asset_options.disable_gravity = True
        asset_options.override_inertia = True
        # asset_options.override_com = True

    rela_file = object_urdf.relative_to(asset_root).as_posix()
    obj_asset = gym.load_asset(sim,
                               asset_root.as_posix(),
                               rela_file,
                               asset_options)
    return obj_asset


def get_camera_params(width=640, height=480, hov=75, cuda=True):
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = hov
    camera_props.width = width
    camera_props.height = height
    camera_props.enable_tensors = cuda
    return camera_props


@torch.no_grad()
def load_obj_texture(gym, sim, object_urdf):
    texture_files = get_all_files_with_suffix(object_urdf.parent, 'png')
    num_textures = len(texture_files)
    if num_textures > 1:
        logger.warning(f'Multiple image files exist, will use the first image as the texture!')
    elif num_textures == 0:
        raise RuntimeError(f'No texture file is found!')
    texture_file = texture_files[0]
    texture_handle = gym.create_texture_from_file(sim,
                                                  texture_file.as_posix(),
                                                  )
    return texture_handle


def apply_prop_samples(prop, og_prop, attr, attr_randomization_params, sample):
    if isinstance(prop, gymapi.SimParams):
        if attr == 'gravity':
            if attr_randomization_params['operation'] == 'scaling':
                prop.gravity.x = og_prop['gravity'].x * sample[0]
                prop.gravity.y = og_prop['gravity'].y * sample[1]
                prop.gravity.z = og_prop['gravity'].z * sample[2]
            elif attr_randomization_params['operation'] == 'additive':
                prop.gravity.x = og_prop['gravity'].x + sample[0]
                prop.gravity.y = og_prop['gravity'].y + sample[1]
                prop.gravity.z = og_prop['gravity'].z + sample[2]
    elif isinstance(prop, np.ndarray):
        if attr_randomization_params['operation'] == 'scaling':
            new_prop_val = og_prop[attr] * sample
        elif attr_randomization_params['operation'] == 'additive':
            new_prop_val = og_prop[attr] + sample

        if 'num_buckets' in attr_randomization_params and attr_randomization_params['num_buckets'] > 0:
            new_prop_val = get_bucketed_val(new_prop_val, attr_randomization_params)
        prop[attr] = new_prop_val
    else:
        cur_attr_val = og_prop[attr]
        if attr_randomization_params['operation'] == 'scaling':
            new_prop_val = cur_attr_val * sample
        elif attr_randomization_params['operation'] == 'additive':
            new_prop_val = cur_attr_val + sample

        if 'num_buckets' in attr_randomization_params and attr_randomization_params['num_buckets'] > 0:
            new_prop_val = get_bucketed_val(new_prop_val, attr_randomization_params)
        setattr(prop, attr, new_prop_val)
