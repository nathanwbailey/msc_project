import numpy as np
import torch
from torch.utils.data import Dataset


def random_mask(sample, mask_prob=0.7):
    random_tensor = torch.rand(sample.shape, device=sample.device)
    mask = (random_tensor > mask_prob).float()
    masked_image = sample * mask
    return masked_image


class WeatherBenchDatasetWindow(Dataset):
    def __init__(self, data, context_length, target_length, stride=1):
        self.data = data
        self.context_length = context_length
        self.target_length = target_length
        self.stride = stride

    def __len__(self):
        return (
            self.data.shape[0] - (self.context_length + self.target_length)
        ) // self.stride + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_length]
        y = self.data[
            idx
            + self.context_length : idx
            + self.context_length
            + self.target_length
        ]
        return x, y
