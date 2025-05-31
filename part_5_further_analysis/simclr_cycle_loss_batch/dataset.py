import numpy as np
import torch
from torch.utils.data import Dataset
from augment_functions import (
    augment_sample,
    augment_sample_random_mask,
    random_mask,
    resize_encoder,
)


class WeatherBenchDataset(Dataset):
    """
    PyTorch Dataset for WeatherBench data with temporal augmentation and masking.
    """

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
        """
        Args:
            data (np.ndarray): Input data array.
            max_delta_t (int): Maximum time delta for augmentation.
            decay (float): Decay rate for delta weighting.
            window (int): Window size for hard negative sampling.
            gap (int): Gap for soft negative sampling.
            mask_prob_low (float): Lower bound for masking probability.
            mask_prob_high (float): Upper bound for masking probability.
        """
        self.data = data
        self.max_delta_t = max_delta_t
        self.window = window
        self.gap = gap
        self.mask_prob_low = mask_prob_low
        self.mask_prob_high = mask_prob_high

        self.delta_ts = np.arange(1, max_delta_t + 1)
        self.delta_weights = np.exp(-decay * self.delta_ts)
        self.delta_weights /= self.delta_weights.sum()

    def __len__(self):
        return self.data.shape[0] - (4 * self.max_delta_t)

    def _hard_neg_idx(self, t):
        """
        Sample a hard negative index, avoiding a range around t.
        """
        exclude_range = set(
            range(t - 4 * self.max_delta_t, t + 4 * self.max_delta_t + 1)
        )
        candidates = [
            i
            for i in range(
                t - (self.window + 4 * self.max_delta_t),
                t + self.window + 4 * self.max_delta_t + 1,
            )
            if i not in exclude_range and 0 <= i < len(self.data) - 4 * self.max_delta_t
        ]
        return np.random.choice(candidates)

    def _soft_neg_idx(self, t):
        """
        Sample a soft negative index, far from t by at least 'gap'.
        """
        candidates = list(range(0, t - self.gap)) + list(
            range(t + self.gap, len(self.data) - 4 * self.max_delta_t)
        )
        return np.random.choice(candidates)

    def _create_sample(self, idx):
        """
        Create augmented and masked samples for a given index.
        """
        idx = idx + 2*self.max_delta_t
        X = self.data[idx]

        augment_idx = np.random.choice(self.delta_ts, p=self.delta_weights)
        X_prime = self.data[idx + augment_idx]
        X_prime_2 = self.data[idx - augment_idx]


        X_delta = resize_encoder(self.data[idx + augment_idx])
        X_minus_delta = resize_encoder(self.data[idx - augment_idx])

        X_enc = resize_encoder(X)
        
        X_masked = random_mask(
            X_enc,
            mask_prob_low=self.mask_prob_low,
            mask_prob_high=self.mask_prob_high,
        )

        x = augment_sample(X)
        x_prime = augment_sample_random_mask(
            X_prime,
            mask_prob_low=self.mask_prob_low,
            mask_prob_high=self.mask_prob_high,
        )
        x_prime_2 = augment_sample_random_mask(
            X_prime_2,
            mask_prob_low=self.mask_prob_low,
            mask_prob_high=self.mask_prob_high,
        )

        return x, x_prime, x_prime_2, X, X_masked, X_prime, X_prime_2, X_enc, X_delta, X_minus_delta

    def __getitem__(self, idx):
        """
        Returns:
            dict: Dictionary of torch.Tensors for augmented and masked samples,
                  with keys labeling anchor, soft negative, and hard negative.
        """
        # Anchor sample
        x, x_prime, x_prime_2, X, X_masked, X_prime, X_prime_2, X_enc, X_delta, X_minus_delta = self._create_sample(idx)
        # Negative indices
        hard_idx = self._hard_neg_idx(idx)
        soft_idx = self._soft_neg_idx(idx)
        # Soft negative sample
        x_soft, x_prime_soft, x_prime_2_soft, X_soft, X_masked_soft, X_prime_soft, X_prime_2_soft, X_enc_soft, X_delta_soft, X_minus_delta_soft = self._create_sample(soft_idx)
        # Hard negative sample
        x_hard, x_prime_hard, x_prime_2_hard, X_hard, X_masked_hard, X_prime_hard, X_prime_2_hard, X_enc_hard, X_delta_hard, X_minus_delta_hard = self._create_sample(hard_idx)

        return {
            "x_pos_1": torch.stack([x, x_soft, x_hard]),
            "x_pos_2": torch.stack([x_prime, x_prime_soft, x_prime_hard]),
            "x_pos_3": torch.stack([x_prime_2, x_prime_2_soft, x_prime_2_hard]),
            "X_orig": torch.stack([X, X_soft, X_hard]),
            "X_masked": torch.stack([X_masked, X_masked_soft, X_masked_hard]),
            "X_masked_delta": torch.stack([X_prime, X_prime_soft, X_prime_hard]),
            "X_masked_delta_2": torch.stack([X_prime_2, X_prime_2_soft, X_prime_2_hard]),
            "X_enc": torch.stack([X_enc, X_enc_soft, X_enc_hard]),
            "X_delta": torch.stack([X_delta, X_delta_soft, X_delta_hard]),
            "X_minus_delta": torch.stack([X_minus_delta, X_minus_delta_soft, X_minus_delta_hard]),
        }
