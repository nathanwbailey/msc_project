from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from augment_functions import augment_sample, augment_sample_random_mask, resize_to_orig, resize_encoder, random_mask

class WeatherBenchDataset(Dataset):
    def __init__(self, data, window=30, gap=1000, augment_sample_random_mask=0.7):
        self.data = data
        self.augment_sample_random_mask = augment_sample_random_mask
        self.window = window
        self.gap = gap

    def __len__(self):
        return self.data.shape[0]
    
    def _hard_neg_idx(self, t):
        candidates = [i for i in range(t-self.window, t+self.window+1) if i!=t and 0 <= i < len(self.data)]
        return np.random.choice(candidates)

    def _soft_neg_idx(self, t):
        candidates = list(range(0, t - self.gap)) + list(range(t + self.gap, len(self.data)))
        return np.random.choice(candidates)

    def _create_sample(self, idx):
        X = self.data[idx]
        X_enc = resize_encoder(X)
        X_masked = random_mask(X_enc, mask_prob=self.augment_sample_random_mask)
        x = augment_sample(X)
        x_prime = augment_sample_random_mask(X, random_mask_prob=self.augment_sample_random_mask)
        return x, x_prime, X, X_masked


    def __getitem__(self, idx):
        hard_idx = self._hard_neg_idx(idx)
        soft_idx = self._soft_neg_idx(idx)
        x, x_prime, X, X_masked = self._create_sample(idx)
        x_soft, x_prime_soft, X_soft, X_masked_soft = self._create_sample(soft_idx)
        x_hard, x_prime_hard, X_hard, X_masked_hard = self._create_sample(hard_idx)
        return torch.stack([x, x_soft, x_hard]), torch.stack([x_prime, x_prime_soft, x_prime_hard]), torch.stack([X, X_soft, X_hard]), torch.stack([X_masked, X_masked_soft, X_masked_hard])
        