from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from augment_functions import augment_sample, augment_sample_random_mask, resize_to_orig, resize_encoder, random_mask

class WeatherBenchDataset(Dataset):
    def __init__(self, data, max_delta_t=5, decay=0.1, augment_sample_random_mask=0.7):
        self.data = data
        self.augment_sample_random_mask = augment_sample_random_mask
        self.max_delta_t = max_delta_t
        self.delta_ts = np.arange(1, max_delta_t + 1)
        self.delta_weights = np.exp(-decay * self.delta_ts)
        self.delta_weights /= self.delta_weights.sum()

    def __len__(self):
        return (self.data.shape[0] - self.max_delta_t)

    def __getitem__(self, idx):
        X = self.data[idx]
        augment_idx = np.random.choice(np.arange(1, self.max_delta_t + 1), p=self.delta_weights)
        X_prime = self.data[idx+augment_idx]
        X_enc = resize_encoder(X)
        X_masked = random_mask(X_enc, mask_prob=self.augment_sample_random_mask)
        x = augment_sample(X)
        x_prime = augment_sample_random_mask(X_prime, random_mask_prob=self.augment_sample_random_mask)
        x_orig = resize_to_orig(x)
        return x, x_prime, x_orig, X_enc, X, X_masked