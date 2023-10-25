import numpy as np
import torch


def batched_coordinates_array(coords, device=None):
    if device is None:
        if isinstance(coords, torch.Tensor):
            device = coords[0].device
        else:
            device = "cpu"

    N = coords.shape[0] * coords.shape[1]
    flat_coords = coords.reshape(N, 3)
    batch_indices = torch.arange(coords.shape[0], device=device).repeat_interleave(coords.shape[1]).view(-1, 1)
    bcoords = torch.cat((batch_indices, flat_coords), dim=-1)
    return bcoords
