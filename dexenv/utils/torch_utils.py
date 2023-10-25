import numpy as np
import random
import re
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from collections.abc import Sequence
from loguru import logger
from pathlib import Path
from torch.distributions import Categorical
from torch.distributions import Independent
from torch.distributions import TransformedDistribution

from dexenv.utils.common import module_available
from dexenv.utils.common import pathlib_file
from dexenv.utils.wandb_utils import create_artifact_download_path
from dexenv.utils.wandb_utils import create_artifact_name
from dexenv.utils.wandb_utils import create_model_name

PYTORCH3D_AVAILABLE = module_available('pytorch3d')
if PYTORCH3D_AVAILABLE:
    import pytorch3d.transforms.rotation_conversions as py3d_rot_cvt

"""
Isaac and Scipy uses the [xyzw] convention for quaternion
Pytorch3D uses the [wxyz] convention
"""


@torch.no_grad()
def random_quaternions(num, dtype=None, device=None, order='xyzw'):
    """
    return quaternions in [w, x, y, z] or [x, y, z, w]
    """
    if PYTORCH3D_AVAILABLE:
        quats = py3d_rot_cvt.random_quaternions(num, dtype=dtype, device=device)
    else:
        """
        http://planning.cs.uiuc.edu/node198.html
        """
        ran = torch.rand(num, 3, dtype=dtype, device=device)
        r1, r2, r3 = ran[:, 0], ran[:, 1], ran[:, 2]
        pi2 = 2 * np.pi
        r1_1 = torch.sqrt(1.0 - r1)
        r1_2 = torch.sqrt(r1)
        t1 = pi2 * r2
        t2 = pi2 * r3

        quats = torch.zeros(num, 4, dtype=dtype, device=device)
        quats[:, 0] = r1_1 * (torch.sin(t1))
        quats[:, 1] = r1_1 * (torch.cos(t1))
        quats[:, 2] = r1_2 * (torch.sin(t2))
        quats[:, 3] = r1_2 * (torch.cos(t2))

    assert order in ['xyzw', 'wxyz']
    if order == 'xyzw':
        quats = quat_wxyz_to_xyzw(quats)
    return quats


@torch.no_grad()
def quat_xyzw_to_wxyz(quat_xyzw):
    quat_wxyz = torch.index_select(quat_xyzw, -1,
                                   torch.LongTensor([3, 0, 1, 2]).to(quat_xyzw.device))
    return quat_wxyz


@torch.no_grad()
def quat_wxyz_to_xyzw(quat_wxyz):
    quat_xyzw = torch.index_select(quat_wxyz, -1,
                                   torch.LongTensor([1, 2, 3, 0]).to(quat_wxyz.device))
    return quat_xyzw


def torch_to_np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    else:
        return tensor.cpu().detach().numpy()


def swap_flatten_leading_axes(array):
    """
    Swap and then flatten the array along axes 0 and 1
    """
    s = array.shape
    return swap_axes(array,
                     (0, 1)).reshape(s[0] * s[1], *s[2:])


@torch.no_grad()
def unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim, sorted=False)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


def swap_axes(data, axes):
    if isinstance(data, torch.Tensor):
        return torch.transpose(data, *axes)
    else:
        return np.swapaxes(data, *axes)


def torch_long(array, device='cpu'):
    if isinstance(array, torch.Tensor):
        return array.long().to(device)
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array).long().to(device)
    elif isinstance(array, list):
        return torch.LongTensor(array).to(device)
    elif isinstance(array, dict):
        new_dict = dict()
        for k, v in array.items():
            new_dict[k] = torch_long(v, device)
        return new_dict


def action_entropy(action_dist, log_prob=None):
    entropy = action_dist.entropy()
    return entropy


def freeze_model(model, eval=True):
    if isinstance(model, Sequence):  # a list or a tuple
        for md in model:
            freeze_model(md)
    else:
        if eval:
            model.eval()
        for param in model.parameters():
            param.requires_grad = False


def action_from_dist(action_dist, sample=True):
    if isinstance(action_dist, Categorical):
        if sample:
            return action_dist.sample()
        else:
            return action_dist.probs.argmax(dim=-1)
    elif isinstance(action_dist, Independent):
        if sample:
            return action_dist.rsample()
        else:
            return action_dist.mean
    elif isinstance(action_dist, TransformedDistribution):
        if not sample:
            if isinstance(action_dist.base_dist, Independent):
                out = action_dist.base_dist.mean
                out = action_dist.transforms[0](out)
                return out
            else:
                raise TypeError('Deterministic sampling is not '
                                'defined for transformed distribution!')
        if action_dist.has_rsample:
            return action_dist.rsample()
        else:
            return action_dist.sample()
    elif isinstance(action_dist, torch.Tensor):
        # if the input is not a distribution
        return action_dist
    else:
        raise TypeError('Getting actions for the given '
                        'distribution is not implemented!')


def action_log_prob(action, action_dist):
    try:
        log_prob = action_dist.log_prob(action)
    except NotImplementedError:
        raise NotImplementedError('Getting log_prob of actions for the '
                                  'given distribution is not implemented!')
    return log_prob


@torch.no_grad()
def torch_rand_uniform(lower, upper, shape=None, device='cpu'):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    shape = (1,) if shape is None else shape
    if isinstance(shape, int):
        shape = (shape,)
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.no_grad()
def torch_rand_choice(values, size=None, replace=False):
    if isinstance(values, int):
        values = torch.arange(values)
    if size is None:
        size = 1
    if replace:
        return values[torch.randint(len(values), (size,))]
    else:
        indices = torch.tensor(random.sample(range(len(values)), size))
        ## note this can be slow when len(values) and size are very very big
        # indices = torch.randperm(len(values))[:size]
        return values[indices]


def clip_grad(params, max_grad_norm):
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(params,
                                                   max_grad_norm)
        grad_norm = grad_norm.item()
    else:
        grad_norm = get_grad_norm(params)
    return grad_norm


def get_grad_norm(model):
    total_norm = 0
    iterator = model.parameters() if isinstance(model, nn.Module) else model
    for p in iterator:
        if p.grad is None:
            continue
        total_norm += p.grad.data.pow(2).sum().item()
    total_norm = total_norm ** 0.5
    return total_norm


def load_state_dict(model, pretrained_dict, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []
    if not pretrained_dict:
        logger.warning(f'Pretrained model is empty in func: load_state_dict')
        return
    model_dict = model.state_dict()
    p_dict = dict()
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape and k not in exclude_keys:
            p_dict[k] = v
        else:
            logger.warning(f'Param [{k}] from the trained model is not loaded!')
            if k not in model_dict:
                logger.warning(f'Param [{k}] is not in the model!')
                continue
            if v.shape != model_dict[k].shape:
                logger.warning(f'Param [{k}] shape mismatch: pretrained shape: {v.shape},  model shape:{model_dict[k].shape}')
                continue
            if k in exclude_keys:
                logger.warning(f'Param [{k}] is in the exclude_keys list!')
                continue
    model_dict_keys = model_dict.keys()
    model_dict_keys_unchanged = [x for x in model_dict_keys if x not in pretrained_dict]
    if len(model_dict_keys_unchanged) > 0:
        for ku in model_dict_keys_unchanged:
            logger.warning(f'Param [{ku}] in the current model is missing in the trained model!')
    model_dict.update(p_dict)
    model.load_state_dict(model_dict)


def load_ckpt_data(wandb_run_id, project_name=None, pretrain_model=None, eval=False):
    if pretrain_model is not None:
        # if the pretrain_model is the path of the folder
        # that contains the checkpoint files, then it will
        # load the most recent one.
        if pretrain_model.startswith('wandb:'):
            pretrain_model = parse_ckpt_path(pretrain_model[6:], project_name=project_name)
        else:
            if isinstance(pretrain_model, str):
                pretrain_model = Path(pretrain_model)
            if pretrain_model.suffix not in ['.pt', '.ckpt']:
                pretrain_model = get_latest_ckpt(pretrain_model)
        ckpt_data = load_torch_model(pretrain_model)
        return ckpt_data
    ckpt_file = parse_ckpt_path(wandb_run_id, eval=eval, project_name=project_name)
    ckpt_data = load_torch_model(ckpt_file)
    return ckpt_data


def parse_ckpt_path(wandb_run_id, project_name=None, eval=False):
    if '/' in wandb_run_id:
        wandb_run_id_split = wandb_run_id.split('/')
        project_name = wandb_run_id_split[-2]
        wandb_run_id = wandb_run_id_split[-1]
    artifact_name = create_artifact_name(wandb_run_id, eval=eval)
    if not ':' in artifact_name:
        tag = 'latest'
        artifact_name = f'{artifact_name}:{tag}'

    artifact_dir = pathlib_file(create_artifact_download_path(artifact_name))
    download_artifact = True
    logger.info(f'Artifact directory:{artifact_dir}')
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    if download_artifact:
        api = wandb.Api(overrides=dict(project=project_name))
        artifact = api.artifact(artifact_name)
        artifact_dir = pathlib_file(artifact.download(root=artifact_dir))
    model_name = create_model_name(eval=eval)
    ckpt_file = artifact_dir.joinpath(model_name)
    return ckpt_file


def reset_hidden_state_at_done(hidden_state, done=None):
    if hidden_state is None:
        return hidden_state
    if done is not None:
        # if the last step is the end of an episode,
        # then reset hidden state
        if isinstance(done, torch.Tensor):
            done_idx = done.flatten().nonzero(as_tuple=True)[0].cpu().numpy()
        else:
            done_idx = np.argwhere(done).flatten()
        if done_idx.size > 0:
            if isinstance(hidden_state, Sequence):
                hid1 = hidden_state[0]
                hid2 = hidden_state[1]
                ld, b, hz = hid1.shape
                hid1[:, done_idx] = torch.zeros(ld, done_idx.size, hz,
                                                device=hid1.device)
                if hid2 is None:  # in evaluation, we do not need to compute value
                    hidden_state = (hid1, None)
                else:
                    ld, b, hz = hid2.shape
                    hid2[:, done_idx] = torch.zeros(ld, done_idx.size, hz,
                                                    device=hid2.device)
                    hidden_state = (hid1, hid2)
            else:
                ld, b, hz = hidden_state.shape
                hidden_state[:, done_idx] = torch.zeros(ld, done_idx.size, hz,
                                                        device=hidden_state.device)
    return hidden_state


def detach_tensors(tensor):
    if isinstance(tensor, Sequence):
        out = []
        for ten in tensor:
            out.append(ten.detach())
        return out
    return tensor.detach() if tensor is not None else tensor


def load_torch_model(model_file):
    logger.info(f'Loading model from {model_file}')
    if isinstance(model_file, str):
        model_file = Path(model_file)
    if not model_file.exists():
        raise ValueError(f'Checkpoint file ({model_file}) '
                         f'does not exist!')
    ckpt_data = torch.load(model_file)
    return ckpt_data


def save_model(data, wandb_run, env=None, is_best=False, step=None, eval=False):
    run_dir = pathlib_file(wandb_run.dir)
    metadata = dict(
        step=step
    )
    model_name = create_model_name()
    ckpt_file = run_dir.joinpath(model_name)
    logger.info(f'Saving checkpoint, step: {step}.')
    torch.save(data, ckpt_file)
    artifact_name = create_artifact_name(wandb_run.id, eval=eval)
    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_file(ckpt_file)
    aliases = ["latest", "best"] if is_best else ["latest"]
    wandb_run.log_artifact(artifact, aliases=aliases)


def get_latest_ckpt(path):
    ckpt_files = [x for x in list(path.iterdir()) if x.suffix == '.pt']
    num_files = len(ckpt_files)
    if num_files < 1:
        raise ValueError('No checkpoint files found!')
    elif num_files == 1:
        return ckpt_files[0]
    else:
        filenames = [x.name for x in ckpt_files]
        latest_file = None
        latest_step = -np.inf
        for idx, fn in enumerate(filenames):
            num = re.findall(r'\d+', fn)
            if not num:
                continue
            step_num = int(num[0])
            if step_num > latest_step:
                latest_step = step_num
                latest_file = ckpt_files[idx]
        return latest_file


def torch_float(array, device='cpu'):
    if isinstance(array, torch.Tensor):
        return array.float().to(device)
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array).float().to(device)
    elif isinstance(array, list):
        return torch.FloatTensor(array).to(device)
    elif isinstance(array, dict):
        new_dict = dict()
        for k, v in array.items():
            new_dict[k] = torch_float(v, device)
        return new_dict


def move_to(models, device):
    if isinstance(models, Sequence):
        for model in models:
            move_to(model, device)
    else:
        if models is not None:
            models.to(device)


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
