from torch.utils.data import Dataset
import torch
import numpy as np
from augment_functions import (
    augment_sample,
    augment_sample_random_mask,
    resize_encoder,
    random_mask,
)

class WeatherBenchDataset(Dataset):
    """
    PyTorch Dataset for WeatherBench data with temporal augmentation and masking.
    Provides positive and negative samples for contrastive learning.
    """
    def __init__(
        self,
        data,
        delta_t=5,
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
        self.delta_t = delta_t


    def __len__(self):
        # Ensures indices are valid for augmentation
        return self.data.shape[0] - 2 * self.delta_t

    def _hard_neg_idx(self, t):
        """
        Returns a hard negative index, avoiding a window around t.
        """
        exclude = set(range(t - 2 * self.delta_t, t + 2 * self.delta_t + 1))
        candidates = [
            i
            for i in range(
                t - (self.window + 2 * self.delta_t),
                t + self.window + 2 * self.delta_t + 1,
            )
            if i not in exclude and 0 <= i < len(self.data) - 2 * self.delta_t
        ]
        return np.random.choice(candidates)

    def _soft_neg_idx(self, t):
        """
        Returns a soft negative index, far from t by at least 'gap'.
        """
        candidates = (
            list(range(0, t - self.gap))
            + list(range(t + self.gap, len(self.data) - 2 * self.delta_t))
        )
        return np.random.choice(candidates)

    def _create_sample(self, idx):
        """
        Creates augmented and masked samples for a given index.
        """
        idx = idx + self.delta_t
        X = self.data[idx]
        X_prime_seq = self.data[idx+1:idx+self.delta_t+1]
        X_prime_2_seq = self.data[idx-self.delta_t:idx]

        X_enc = resize_encoder(X)
        X_masked = random_mask(
            X_enc, mask_prob_low=self.mask_prob_low, mask_prob_high=self.mask_prob_high
        )
        x = augment_sample(X)
        
        x_prime = torch.stack([augment_sample_random_mask(
            X_prime, mask_prob_low=self.mask_prob_low, mask_prob_high=self.mask_prob_high
        ) for X_prime in X_prime_seq])

        x_prime_2 = torch.stack([augment_sample_random_mask(
            X_prime_2, mask_prob_low=self.mask_prob_low, mask_prob_high=self.mask_prob_high
        ) for X_prime_2 in X_prime_2_seq])

        return (
            x,
            x_prime,
            x_prime_2,
            X,
            X_masked,
            X_prime_seq,
            X_prime_2_seq,
        )

    def __getitem__(self, idx):
        """
        Returns a tuple of stacked tensors for anchor, soft negative, and hard negative samples.
        """
        x, x_prime, x_prime_2, X, X_masked, X_prime, X_prime_2 = self._create_sample(idx)

        hard_idx = self._hard_neg_idx(idx)
        soft_idx = self._soft_neg_idx(idx)

        x_soft, x_prime_soft, x_prime_2_soft, X_soft, X_masked_soft, X_prime_soft, X_prime_2_soft = self._create_sample(soft_idx)
        x_hard, x_prime_hard, x_prime_2_hard, X_hard, X_masked_hard, X_prime_hard, X_prime_2_hard = self._create_sample(hard_idx)

        return (
            torch.stack([x, x_soft, x_hard]),
            torch.stack([x_prime, x_prime_soft, x_prime_hard]),
            torch.stack([x_prime_2, x_prime_2_soft, x_prime_2_hard]),
            torch.stack([X, X_soft, X_hard]),
            torch.stack([X_masked, X_masked_soft, X_masked_hard]),
            torch.stack([X_prime, X_prime_soft, X_prime_hard]),
            torch.stack([X_prime_2, X_prime_2_soft, X_prime_2_hard]),
            np.stack([idx, soft_idx, hard_idx]),
        )