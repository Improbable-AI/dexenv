import gym
import torch.nn as nn
from collections.abc import Sequence
from loguru import logger

from dexenv.models.diag_gaussian_pol.diag_gaussian_policy import \
    DiagGaussianPolicy
from dexenv.models.utils import get_activation
from dexenv.models.value_nets.value_net import ValueNet


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        act = get_activation(act)
        self.body = nn.Sequential(
            nn.Linear(in_dim, 512),
            act(),
            nn.Linear(512, 256),
            act(),
            nn.Linear(256, out_dim),
            act(),
        )

    def forward(self, x):
        if isinstance(x, dict):
            # assert len(x) == 1
            x = list(x.values())[0]
        out = self.body(x)
        return out


def get_mlp_critic(ob_size, act='gelu'):
    logger.info(f'Critic state input size:{ob_size}')
    critic_body = SimpleMLP(in_dim=ob_size, out_dim=256, act=act)
    critic = ValueNet(critic_body,
                      in_features=256)
    return critic


def get_mlp_actor(ob_size, env, act='gelu', init_log_std=-0.2):
    logger.info(f'Actor state input size:{ob_size}')
    actor_body = SimpleMLP(in_dim=ob_size, out_dim=256, act=act)

    if isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        head_func = DiagGaussianPolicy
        actor = head_func(actor_body,
                          in_features=256,
                          action_dim=act_size,
                          init_log_std=init_log_std)
    else:
        raise TypeError(f'Unknown action space type: {env.action_space}')
    return actor


def get_expert_actor(ob_size, action_dim, act='gelu', init_log_std=-0.2):
    logger.info(f'Actor state input size:{ob_size}')
    actor_body = SimpleMLP(in_dim=ob_size, out_dim=256, act=act)
    head_func = DiagGaussianPolicy
    actor = head_func(actor_body,
                      in_features=256,
                      action_dim=action_dim,
                      init_log_std=init_log_std)

    return actor


def get_mlp_models(ob_size, env, act='gelu', init_log_std=-0.2):
    logger.info(f'Actor Critic state input size:{ob_size}')
    actor = get_mlp_actor(ob_size, env, act=act,
                          init_log_std=init_log_std)
    critic = get_mlp_critic(ob_size, act=act)
    return actor, critic
