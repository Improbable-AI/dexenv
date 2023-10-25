import MinkowskiEngine as ME
import numpy as np
import torch
from loguru import logger
from torch import nn

from dexenv.models.diag_gaussian_pol.rnn_diag_gaussian_policy import \
    RNNDiagGaussianPolicy
from dexenv.models.sequence.rnn_base import RNNBase
from dexenv.models.utils import get_activation
from dexenv.models.voxel_model import VoxelModel
from dexenv.utils.minkowski_utils import batched_coordinates_array
from dexenv.utils.pcd_augmentation import JitterPoints
from dexenv.utils.pcd_augmentation import ProbCompose
from dexenv.utils.pcd_augmentation import RandomDropout
from dexenv.utils.torch_utils import unique
from dexenv.utils.voxel_utils import create_input_batch


class VoxelRNN(RNNBase):
    def __init__(self, cfg, act_dim=9):
        self.cfg = cfg
        body_net = VoxelModel(embed_dim=self.cfg.embed_dim,
                              out_features=self.cfg.embed_dim,
                              batch_norm=self.cfg.batch_norm,
                              act=self.cfg.mink_act,
                              channel_groups=self.cfg.channel_groups,
                              in_channels=self.cfg.color_channels,
                              layer_norm=self.cfg.layer_norm)
        act = get_activation(self.cfg.act)
        super().__init__(body_net=body_net,
                         rnn_features=cfg.rnn_features,
                         in_features=cfg.embed_dim,
                         rnn_layers=cfg.rnn_layers,
                         act=act)
        logger.warning(f'Act dim in Policy input is:{act_dim}')
        if self.cfg.act_in:
            in_c = act_dim
            self.action_embed = nn.Sequential(
                nn.Linear(in_c, 32),
                act(),
                nn.Linear(32, 32)
            )
            self.voxel_act_merger = nn.Linear(self.cfg.embed_dim + 32, self.cfg.embed_dim)

        if self.cfg.pred_rot_dist:
            self.rot_dist_fcs = nn.Sequential(
                nn.Linear(self.cfg.embed_dim, 128),
                act(),
                nn.Linear(128, 1)
            )

        self.auxiliary_training = self.cfg.pred_rot_dist

        if self.cfg.ptd_noise:
            transform_p = self.cfg.ptd_noise_prob
            transforms = [
                dict(p=transform_p, t=JitterPoints(sigma=np.ceil(0.004 / self.cfg.quantization_size),
                                                   clip=np.ceil(0.008 / self.cfg.quantization_size))),
                dict(p=transform_p, t=RandomDropout(dropout_ratio=[0., 0.2])),
            ]
            self.transforms = ProbCompose(transforms)

    def forward(self, x, hidden_state=None, done=None, *args, **kwargs):
        if self.cfg.act_in:
            prev_action = x['prev_action']
            x = x['ptd']
        if isinstance(x, dict):
            coords = x['coords']
            color = x['color']
        else:
            coords = x
            color = None
        x_sparse_tensor, b, t = self.convert_to_sparse_tensor(coords=coords, color=color)

        with torch.set_grad_enabled(True):
            voxel_features = self.body(x_sparse_tensor)
            voxel_features = self.unflat_batch_time(voxel_features, b, t)
            rnn_input = voxel_features
            if self.cfg.act_in:
                prev_action_embedding = self.action_embed(prev_action)
                voxel_features_w_act = torch.cat((voxel_features, prev_action_embedding), dim=-1)
                rnn_input = self.voxel_act_merger(voxel_features_w_act)

        rnn_out, hidden_state = self.forward_gru(rnn_input, done, hidden_state)
        out = rnn_out

        if self.auxiliary_training:
            out = [out]
            if self.cfg.pred_rot_dist:
                rot_dist = self.rot_dist_fcs(voxel_features)
                out.append(rot_dist)
            else:
                out.append(None)
        return out, hidden_state

    @torch.no_grad()
    def convert_to_sparse_tensor(self, coords, color=None):
        b = coords.shape[0]
        t = coords.shape[1]
        flat_coords = self.flat_batch_time(coords)

        if self.cfg.ptd_noise:
            flat_coords = self.transforms(flat_coords)[0]

        coordinates_batch = batched_coordinates_array(flat_coords, device=coords.device)
        coordinates_batch, uindices = unique(coordinates_batch, dim=0)
        features_batch = torch.full((coordinates_batch.shape[0], self.cfg.color_channels),
                                    0.5, device=coordinates_batch.device)
        batch = {
            "coordinates": coordinates_batch,
            "features": features_batch,
        }
        input_batch_sparse = create_input_batch(batch, device=coords.device,
                                                quantization_size=None,
                                                speed_optimized=self.cfg.speed_optimized,
                                                quantization_mode=self.cfg.quantization_mode)
        return input_batch_sparse, b, t

    def flat_batch_time(self, x):
        return x.view(x.shape[0] * x.shape[1], *x.shape[2:])

    def unflat_batch_time(self, x, b, t):
        return x.view(b, t, *x.shape[1:])


def get_voxel_rnn_model(cfg, env=None, act_dim=12):
    rnn_body = VoxelRNN(cfg=cfg.vision, act_dim=act_dim)
    act_size = act_dim
    actor = RNNDiagGaussianPolicy(rnn_body,
                                  in_features=cfg.vision.rnn_features,
                                  action_dim=act_size,
                                  std_cond_in=cfg.alg.std_cond_in)
    return actor
