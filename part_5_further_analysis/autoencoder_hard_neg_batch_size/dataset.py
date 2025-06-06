import numpy as np
import pandas as pd
import torch
from augment_functions import (augment_sample, augment_sample_random_mask,
                               random_mask, resize_encoder, resize_to_orig)
from torch.utils.data import Dataset


class WeatherBenchDataset(Dataset):
    def __init__(
        self,
        data,
        max_delta_t=5,
        decay=0.1,
        window=30,
        gap=1000,
        mask_prob_low=0.5,
        mask_prob_high=0.9,
    ):
        self.data = data
        self.mask_prob_low = mask_prob_low
        self.mask_prob_high = mask_prob_high
        self.window = window
        self.gap = gap
        self.max_delta_t = max_delta_t
        self.delta_ts = np.arange(1, max_delta_t + 1)
        self.delta_weights = np.exp(-decay * self.delta_ts)
        self.delta_weights /= self.delta_weights.sum()

    def __len__(self):
        return self.data.shape[0] - (2 * self.max_delta_t)

    def _hard_neg_idx(self, t):
        exclude_range = set(
            range(t - 2 * self.max_delta_t, t + 2 * self.max_delta_t + 1)
        )
        candidates = [
            i
            for i in range(
                t - (self.window + 2 * self.max_delta_t),
                t + self.window + 2 * self.max_delta_t + 1,
            )
            if i not in exclude_range
            and 0 <= i < len(self.data) - 2 * self.max_delta_t
        ]
        return np.random.choice(candidates)

    def _soft_neg_idx(self, t):
        candidates = list(range(0, t - self.gap)) + list(
            range(t + self.gap, (len(self.data) - 2 * self.max_delta_t))
        )
        return np.random.choice(candidates)

    def _create_sample(self, idx):
        idx = idx + self.max_delta_t
        X = self.data[idx]
        augment_idx = np.random.choice(
            np.arange(1, self.max_delta_t + 1), p=self.delta_weights
        )
        X_enc = resize_encoder(X)
        X_masked = random_mask(
            X_enc,
            mask_prob_low=self.mask_prob_low,
            mask_prob_high=self.mask_prob_high,
        )
        return X, X_masked

    def __getitem__(self, idx):
        X, X_masked = self._create_sample(idx)
        hard_idx = self._hard_neg_idx(idx)
        soft_idx = self._soft_neg_idx(idx)
        X_soft, X_masked_soft = self._create_sample(soft_idx)
        X_hard, X_masked_hard = self._create_sample(hard_idx)
        return (
            torch.stack([X, X_soft, X_hard]),
            torch.stack([X_masked, X_masked_soft, X_masked_hard]),
        )
