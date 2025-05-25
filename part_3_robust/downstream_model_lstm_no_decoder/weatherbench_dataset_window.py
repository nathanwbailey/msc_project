from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random

def resize_encoder(sample):
    sample = F.interpolate(sample, size=(144, 72), mode='bicubic', align_corners=False)
    return sample

def random_mask(sample, mask_prob_low=0.7, mask_prob_high=0.7, mode_drop=None):
    if mask_prob_low == mask_prob_high:
        mask_prob = mask_prob_low
    else:
        mask_prob = random.uniform(mask_prob_low, mask_prob_high)
    random_tensor = torch.rand(sample.shape, device=sample.device)
    mask = (random_tensor > mask_prob).float()
    masked_image = sample * mask
    return masked_image

class WeatherBenchDatasetWindow(Dataset):
    def __init__(self, data, context_length, target_length, stride=1, mask_prob_low=0.7, mask_prob_high=0.7):
        self.data = data
        self.context_length = context_length
        self.target_length = target_length
        self.stride = stride
        self.mask_prob_low=mask_prob_low
        self.mask_prob_high=mask_prob_high
        
    def __len__(self):
        return (self.data.shape[0] - (self.context_length + self.target_length)) // self.stride + 1

    def __getitem__(self, idx):
        x = random_mask(resize_encoder(self.data[idx:idx + self.context_length]), mask_prob_low=self.mask_prob_low, mask_prob_high=self.mask_prob_high)
        y = random_mask(resize_encoder(self.data[idx + self.context_length:idx + self.context_length + self.target_length]), mask_prob_low=self.mask_prob_low, mask_prob_high=self.mask_prob_high)
        return x, y