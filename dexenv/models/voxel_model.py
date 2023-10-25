import torch
from torch import nn

from dexenv.models.utils import get_activation
from dexenv.models.voxel_impala import ImpalaVoxelCNN


class VoxelModel(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 out_features=256,
                 batch_norm=False,
                 act='gelu',
                 channel_groups=[16, 32, 32],
                 in_channels=3,
                 layer_norm=False,
                 no_pool=False,
                 ):
        super().__init__()
        voxel_kwargs = dict(
            in_channels=in_channels,
            out_channels=embed_dim,
            batch_norm=batch_norm,
            act=act,
            channel_groups=channel_groups,
            no_pool=no_pool,
        )
        backbone = ImpalaVoxelCNN

        self.voxel_backbone = backbone(**voxel_kwargs)

        act = get_activation(act)
        in_channels = embed_dim
        if layer_norm:
            self.projector = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, out_features),
                act()
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(in_channels, out_features),
                act()
            )

    def forward(self, scene_voxels):
        voxel_embed = self.voxel_backbone(scene_voxels)
        voxel_embed = voxel_embed.features
        out = self.projector(voxel_embed)
        return out
