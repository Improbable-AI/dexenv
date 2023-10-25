import torch
import torch.nn as nn
from collections.abc import Sequence
from torch.distributions import Independent
from torch.distributions import Normal
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import TanhTransform

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class RNNDiagGaussianPolicy(nn.Module):
    def __init__(self,
                 body_net,
                 action_dim,
                 init_log_std=-0.2,
                 std_cond_in=False,
                 tanh_on_dist=False,
                 in_features=None,
                 clamp_log_std=False):  # add tanh on the action distribution
        super().__init__()
        self.std_cond_in = std_cond_in
        self.tanh_on_dist = tanh_on_dist
        self.body = body_net
        self.clamp_log_std = clamp_log_std

        if in_features is None:
            for i in reversed(range(len(self.body.fcs))):
                layer = self.body.fcs[i]
                if hasattr(layer, 'out_features'):
                    in_features = layer.out_features
                    break

        self.head_mean = nn.Linear(in_features, action_dim)
        if self.std_cond_in:
            self.head_logstd = nn.Linear(in_features, action_dim)
        else:
            self.head_logstd = nn.Parameter(torch.full((action_dim,),
                                                       init_log_std))

    def forward(self, x=None, body_x=None, hidden_state=None, return_hidden_state=True, **kwargs):
        if x is None and body_x is None:
            raise ValueError('One of [x, body_x] should be provided!')
        if body_x is None:
            body_x, hidden_state = self.body(x=x,
                                             hidden_state=hidden_state,
                                             **kwargs)
        body_out = body_x[0] if isinstance(body_x, Sequence) else body_x
        mean = self.head_mean(body_out)
        if self.std_cond_in:
            log_std = self.head_logstd(body_out)
        else:
            log_std = self.head_logstd.expand_as(mean)
        if self.clamp_log_std:
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        if self.tanh_on_dist:
            action_dist = TransformedDistribution(action_dist,
                                                  [TanhTransform(cache_size=1)])
        if return_hidden_state:
            return action_dist, body_x, hidden_state
        else:
            return action_dist, body_x
