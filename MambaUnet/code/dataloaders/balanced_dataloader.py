from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import random

from torch.utils.data import Sampler
import random

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.slices_mvo_len = len(dataset.slices_mvo)
        self.slices_len = len(dataset.slices)

    def __iter__(self):
        indices_slices = list(range(self.slices_len + self.slices_mvo_len))
        fixed = indices_slices[:self.slices_mvo_len]
        shuffled = indices_slices[self.slices_mvo_len:]
        random.shuffle(shuffled)
        indices_slices = fixed + shuffled
        indices_slices_mvo = list(range(self.slices_mvo_len))
        start_point = self.slices_mvo_len

        
        balanced_indices = []
        
        min_len = min(len(indices_slices)-self.slices_mvo_len, len(indices_slices_mvo))
        num_batches = min_len * 2 // self.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.batch_size // 2
            batch_indices = indices_slices[(start_idx+start_point):(start_idx+start_point) + self.batch_size // 2] + \
                            indices_slices_mvo[start_idx:start_idx + self.batch_size // 2]
            balanced_indices.extend(batch_indices)
        
        remaining_slices = indices_slices[len(balanced_indices) // 2 + 454:]
        
        for idx in range(0, len(remaining_slices)-4, self.batch_size // 2):
            batch_indices = remaining_slices[idx:idx + self.batch_size // 2] + \
                            indices_slices_mvo[idx % len(indices_slices_mvo):(idx + self.batch_size // 2) % len(indices_slices_mvo)]
            balanced_indices.extend(batch_indices)
        
        return iter(balanced_indices)
    
    def __len__(self):
        return self.slices_mvo_len + self.slices_len -10