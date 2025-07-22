import numpy as np
import pandas as pd
import torch
from augment_functions import (augment_sample, augment_sample_random_mask,
                               random_mask, resize_encoder, resize_to_orig)
from torch.utils.data import Dataset


class WeatherBenchDataset(Dataset):
    def __init__(self, data, augment_sample_random_mask=0.7):
        self.data = data
        self.augment_sample_random_mask = augment_sample_random_mask

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = self.data[idx]
        X_enc = resize_encoder(X)
        X_masked = random_mask(
            X_enc, mask_prob=self.augment_sample_random_mask
        )
        x = augment_sample(X)
        x_prime = augment_sample_random_mask(
            X, random_mask_prob=self.augment_sample_random_mask
        )
        x_orig = resize_to_orig(x)
        return x, x_prime, x_orig, X_enc, X, X_masked
