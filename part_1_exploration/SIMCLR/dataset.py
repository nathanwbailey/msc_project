import numpy as np
from augment_functions import augment_sample
from torch.utils.data import Dataset


class WeatherBenchDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x_orig = self.data[idx]
        x = augment_sample(x_orig)
        x_prime = augment_sample(x_orig)
        return x, x_prime
