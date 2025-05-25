from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from augment_functions import augment_sample, augment_sample_random_mask, resize_to_orig, resize_encoder, random_mask

class WeatherBenchDataset(Dataset):
    def __init__(self, data, max_delta_t=5, decay=0.1, window=30, gap=1000, augment_sample_random_mask=0.7):
        self.data = data
        self.augment_sample_random_mask = augment_sample_random_mask
        self.window = window
        self.gap = gap
        self.max_delta_t = max_delta_t
        self.delta_ts = np.arange(1, max_delta_t + 1)
        self.delta_weights = np.exp(-decay * self.delta_ts)
        self.delta_weights /= self.delta_weights.sum()

    def __len__(self):
        return (self.data.shape[0] - (self.max_delta_t))
    
    def _hard_neg_idx(self, t):
        exclude_range = set(range(t-self.max_delta_t, t + self.max_delta_t + 1))
        candidates = [i for i in range(t-(self.window+self.max_delta_t), t+self.window+self.max_delta_t+1) if i not in exclude_range and 0 <= i < len(self.data)- self.max_delta_t]
        return np.random.choice(candidates)

    def _soft_neg_idx(self, t):
        candidates = list(range(0, t - self.gap)) + list(range(t + self.gap, (len(self.data)-self.max_delta_t)))
        return np.random.choice(candidates)

    def _create_sample(self, idx):        
        X = self.data[idx]
        augment_idx = np.random.choice(np.arange(1, self.max_delta_t + 1), p=self.delta_weights)
        X_prime = self.data[idx+augment_idx]
        X_enc = resize_encoder(X)
        X_masked = random_mask(X_enc, mask_prob=self.augment_sample_random_mask)
        x = augment_sample(X)
        x_prime = augment_sample_random_mask(X_prime, random_mask_prob=self.augment_sample_random_mask)
        x_prime_idx = augment_sample_random_mask(X, random_mask_prob=self.augment_sample_random_mask)
        return x, x_prime, x_prime_idx, X, X_masked


    def __getitem__(self, idx):
        hard_idx = self._hard_neg_idx(idx)
        soft_idx = self._soft_neg_idx(idx)
        x, x_prime, x_prime_idx, X, X_masked = self._create_sample(idx)
        x_soft, x_prime_soft, x_prime_soft_idx, X_soft, X_masked_soft = self._create_sample(soft_idx)
        x_hard, x_prime_hard, x_prime_hard_idx, X_hard, X_masked_hard = self._create_sample(hard_idx)
        return torch.stack([x, x_soft, x_hard]), torch.stack([x_prime, x_prime_soft, x_prime_hard]), torch.stack([x_prime_idx, x_prime_soft_idx, x_prime_hard_idx]), torch.stack([X, X_soft, X_hard]), torch.stack([X_masked, X_masked_soft, X_masked_hard])
        