import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import Dict
from typing import List


class DictDataset(Dataset):
    def __init__(self, **kwargs):
        self.data = kwargs
        self.length = next(iter(self.data.values())).shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = dict()
        for key, val in self.data.items():
            sample[key] = val[idx]
        return sample


def process_data(data: Dict[str, torch.Tensor]):
    keys = list(data.keys())
    dkey = 'done'
    assert dkey in keys
    keys.remove(dkey)
    xk = keys[0]
    data_pool = {k: [] for k in keys}
    tskey = 'time_steps'
    tskey_pool: List[int] = []
    for i in range(data[xk].shape[0]):
        ds = data[dkey][i]
        d_ids = torch.nonzero(ds)[:, 0] + 1
        d_ids = torch.cat((torch.zeros(1, dtype=torch.long, device=d_ids.device), d_ids))
        seg_len = d_ids[1:] - d_ids[:-1]
        res: int = int(ds.shape[0] - seg_len.sum())
        if res > 0:
            res_tensor = torch.tensor([res], dtype=seg_len.dtype, device=d_ids.device)
            seg_len = torch.cat((seg_len, res_tensor))
        seg_len: List[int] = seg_len.tolist()
        tskey_pool.extend(seg_len)

        for ke in keys:
            kd = data[ke][i]
            traj_seg = torch.split(kd, seg_len)
            data_pool[ke].extend(traj_seg)
    return data_pool, tskey_pool


class TrajDatasetSplitByDone(Dataset):
    def __init__(self, max_steps=None, pad=True, **kwargs):
        data = dict()
        # input: [N, T]
        for key, val in kwargs.items():
            if isinstance(val, np.ndarray):
                data[key] = torch.from_numpy(val)
            else:
                data[key] = val
        keys = list(data.keys())
        dkey = 'done'
        assert dkey in keys
        keys.remove(dkey)
        xk = keys[0]
        tskey = 'time_steps'
        self.data_pool, tskey_pool = process_data(data)
        if pad:
            for ke in keys:
                if max_steps is not None:
                    key_data = self.data_pool[ke]
                    key_data.append(torch.zeros(max_steps, *key_data[0].shape[1:], dtype=key_data[0].dtype))
                    self.data_pool[ke] = pad_sequence(key_data, batch_first=True)[:-1]
                else:
                    self.data_pool[ke] = pad_sequence(self.data_pool[ke], batch_first=True)
        self.data_pool[tskey] = torch.tensor(tskey_pool, dtype=torch.int32)
        self.length = len(self.data_pool[xk])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = dict()
        for key, val in self.data_pool.items():
            sample[key] = val[idx]
        return sample

    @property
    def data(self):
        return self.data_pool
