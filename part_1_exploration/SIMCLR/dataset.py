import numpy as np
import pandas as pd
import torch
from augment_functions import augment_sample
from torch.utils.data import Dataset


class WeatherBenchDataset(Dataset):
    def __init__(self, data, max_delta_t=30, decay=0.9):
        self.data = data
        self.max_delta_t = max_delta_t
        self.delta_ts = np.arange(0, max_delta_t)
        self.delta_weights = np.exp(-decay * self.delta_ts)
        self.delta_weights /= self.delta_weights.sum()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x_orig = self.data[idx]
        x = augment_sample(x_orig)
        x_prime = augment_sample(x_orig)
        return x, x_prime
