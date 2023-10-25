import hydra
import isaacgym
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributions import Independent
from torch.distributions import Normal

import dexenv
from dexenv.utils.common import set_print_formatting
from dexenv.utils.create_task_env import create_task_env


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, out_dim),
            nn.ELU(),
        )

    def forward(self, x):
        out = self.body(x)
        return out


class DiagGaussianPolicy(nn.Module):
    def __init__(self,
                 body_net,
                 action_dim,
                 init_log_std=-0.2,
                 in_features=None, ):
        super().__init__()
        self.body = body_net

        self.head_mean = nn.Linear(in_features, action_dim)
        self.head_logstd = nn.Parameter(torch.full((action_dim,),
                                                   init_log_std))

    def forward(self, x=None, **kwargs):
        body_x = self.body(x, **kwargs)
        mean = self.head_mean(body_x)
        log_std = self.head_logstd.expand_as(mean)
        std = torch.exp(log_std)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        return action_dist


def get_actor(ob_size, act_dim):
    actor_body = SimpleMLP(in_dim=ob_size, out_dim=256)
    actor = DiagGaussianPolicy(body_net=actor_body, in_features=256, action_dim=act_dim)
    return actor


def load_expert(actor, expert_path):
    ckpt_data = torch.load(expert_path)
    actor_dict = actor.state_dict()
    pretrained_dict = ckpt_data['actor_state_dict']
    actor_dict.update(pretrained_dict)
    actor.load_state_dict(actor_dict)
    return actor


@hydra.main(config_path=dexenv.PROJECT_ROOT.joinpath('conf').as_posix(), config_name="debug_dclaw")
def random(cfg: DictConfig):
    set_print_formatting()
    env = create_task_env(cfg)
    env.reset()
    while True:
        action = torch.rand((cfg.task.env.numEnvs, 12), device='cuda')
        env.step(action)


@hydra.main(config_path=dexenv.PROJECT_ROOT.joinpath('conf').as_posix(), config_name="debug_dclaw")
def main(cfg: DictConfig):
    set_print_formatting()
    env = create_task_env(cfg)
    expert_path = dexenv.LIB_PATH.joinpath('pretrained', 'artifacts', 'teacher', 'train-model.pt')
    actor = get_actor(ob_size=env.observation_space['ob'].shape[0], act_dim=env.action_space.shape[0])
    actor.cuda()
    load_expert(actor, expert_path)
    ob = env.reset()
    while True:
        action_dist = actor(ob['ob'])
        action = action_dist.rsample()
        ob = env.step(action)[0]


if __name__ == '__main__':
    main()
