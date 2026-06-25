# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import numpy as np
import gc


class SingleDataSet(Dataset):
    def __init__(self, data, label, size, step, delete_nan=False):
        super(SingleDataSet, self).__init__()
        assert len(data.shape) == 3
        data = data.unfold(-1, size=size, step=step).transpose(1, 2)  
        self.num_window = data.shape[1]
        if len(label.shape) > 1:
            label = label.squeeze(-1)
          
        if delete_nan:
            have_nan = torch.isnan(data).reshape(data.size(0), -1)
            have_nan = torch.any(have_nan, dim=1)
            data = data[~have_nan]
        else:
            data = torch.nan_to_num(data, nan=0.0)
        self.data = data.numpy().astype(np.float32)
        self.label = label
        self.cache = defaultdict(tuple)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if item in self.cache:
            return self.cache[item]
        raw_data = self.data[item]

        try:
            self.cache[item] = (raw_data, self.label[item])
        except MemoryError:
            self._evict_cache()
            self.cache[item] = (raw_data, self.label[item])

        return self.cache[item]

    def _evict_cache(self):
        if self.cache:
            evict_key = next(iter(self.cache.keys()))
            print(f"Evicting cached item with index: {evict_key}")
            del self.cache[evict_key]
            gc.collect()


def get_data_loader(x: torch.Tensor, y: torch.Tensor, delete_nan, window_size: int, window_step: int
                    , batch_size: int = 16, shuffle=True, num_worker: int = 3, seed=50):
    persistent = True if num_worker >= 1 else False
    m_set = SingleDataSet(x, y, window_size, window_step, delete_nan)
    generator = torch.manual_seed(seed)
    loader = DataLoader(dataset=m_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
                        persistent_workers=persistent, generator=generator)

    return loader

