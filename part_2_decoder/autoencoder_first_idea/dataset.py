import numpy as np
import torch
from augment_functions import (augment_sample, augment_sample_random_mask,
                               resize_to_orig)
from torch.utils.data import Dataset


def random_mask(sample, mask_prob=0.7):
    random_tensor = torch.rand(sample.shape, device=sample.device)
    mask = (random_tensor > mask_prob).float()
    masked_image = sample * mask
    return masked_image


class WeatherBenchDataset(Dataset):
    def __init__(self, data, mask_prob):
        self.data = data
        self.mask_prob = mask_prob

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x_orig = self.data[idx]
        x = augment_sample(x_orig)
        x_prime = augment_sample_random_mask(x_orig, self.mask_prob)
        x_orig_aug = resize_to_orig(x)
        return x, x_prime, x_orig, x_orig_aug
