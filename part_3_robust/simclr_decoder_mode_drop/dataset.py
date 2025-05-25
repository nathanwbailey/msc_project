from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from augment_functions import augment_sample, augment_sample_random_mask, resize_to_orig, resize_encoder, random_mask

class WeatherBenchDataset(Dataset):
    def __init__(self, data, augment_sample_random_mask=0.7):
        self.data = data
        self.augment_sample_random_mask = augment_sample_random_mask

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = self.data[idx]
        X_enc = resize_encoder(X)
        X_masked = random_mask(X_enc, mask_prob=self.augment_sample_random_mask)
        x, channel = augment_sample(X)
        x_prime = augment_sample_random_mask(X, channel, random_mask_prob=self.augment_sample_random_mask)
        x_orig = resize_to_orig(x)
        return x, x_prime, x_orig, X_enc, X, X_masked