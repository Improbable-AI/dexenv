import torch
from collections.abc import Sequence

from dexenv.utils.torch_utils import torch_rand_choice
from dexenv.utils.torch_utils import torch_rand_uniform


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None):
        """
        :param sigma:
        :param clip:
        """
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip

    def __call__(self, coords, color=None):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        jitter = self.sigma * torch.randn(coords.shape, device=coords.device)

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)
        jitter = jitter.to(coords.dtype)

        coords = coords + jitter
        return coords, color


class RandomDropout:
    def __init__(self, dropout_ratio):
        if isinstance(dropout_ratio, Sequence):
            assert len(dropout_ratio) == 2
            assert 0 <= dropout_ratio[0] <= 1
            assert 0 <= dropout_ratio[1] <= 1
            self.dropout_ratio_min = float(dropout_ratio[0])
            self.dropout_ratio_max = float(dropout_ratio[1])
        else:
            assert 0 <= dropout_ratio <= 1
            self.dropout_ratio_min = None
            self.dropout_ratio_max = float(dropout_ratio)

    def __call__(self, coords, color=None):
        n = coords.shape[-2]
        if self.dropout_ratio_min is None:
            r = self.dropout_ratio_max
        else:
            # Randomly select removal ratio
            r = torch_rand_uniform(self.dropout_ratio_min, self.dropout_ratio_max)

        mask = torch_rand_choice(n, size=int(n * (1 - r)), replace=False).to(coords.device)
        out_coords = torch.index_select(coords, dim=-2, index=mask)
        out_color = torch.index_select(color, dim=-2, index=mask) if color is not None else color
        return out_coords, out_color


class ProbCompose:
    def __init__(self, transforms):
        """
        a list of dict(p=probability, t=transform)
        :param transforms:
        """
        self.transforms = transforms

    def __call__(self, coords, color=None):
        for ptrans in self.transforms:
            prob = ptrans['p']
            transform = ptrans['t']
            if torch.rand(1) < prob:
                coords, color = transform(coords=coords, color=color)
        return coords, color
