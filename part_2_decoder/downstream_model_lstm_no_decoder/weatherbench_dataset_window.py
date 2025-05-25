import torch
import torch.nn.functional as F
from augment_functions import random_mask
from torch.utils.data import Dataset


def resize_encoder(sample):
    sample = F.interpolate(
        sample, size=(144, 72), mode="bicubic", align_corners=False
    )
    return sample


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
        x = random_mask(
            resize_encoder(self.data[idx : idx + self.context_length])
        )
        y = random_mask(
            resize_encoder(
                self.data[
                    idx
                    + self.context_length : idx
                    + self.context_length
                    + self.target_length
                ]
            )
        )
        return x, y
