import numpy as np
import torch
from augment_functions import random_mask, resize_encoder
from torch.utils.data import Dataset


class WeatherBenchDataset(Dataset):
    def __init__(self, data, mask_prob):
        self.data = data
        self.mask_prob = mask_prob

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = self.data[idx]
        X_enc = resize_encoder(X)
        X_masked = random_mask(X_enc, mask_prob=self.mask_prob)
        return X, X_masked
