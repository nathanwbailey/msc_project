from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F
import random

def resize_encoder(sample):
    sample = sample.unsqueeze(0)
    sample = F.interpolate(sample, size=(144, 72), mode='bicubic', align_corners=False)
    return sample.squeeze(0)

def random_mask(sample, mask_prob_low=0.5, mask_prob_high=0.9):
    mask_prob = random.uniform(mask_prob_low, mask_prob_high)
    random_tensor = torch.rand(sample.shape, device=sample.device)
    mask = (random_tensor > mask_prob).float()
    masked_image = sample * mask
    return masked_image

class WeatherBenchDataset(Dataset):
    def __init__(self, data, mask_prob_low=0.5, mask_prob_high=0.9):
        self.data = data
        self.mask_prob_low = mask_prob_low
        self.mask_prob_high = mask_prob_high
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = self.data[idx]
        X_enc = resize_encoder(X)
        X_masked = random_mask(X_enc, mask_prob_low=self.mask_prob_low, mask_prob_high=self.mask_prob_high)
        return X, X_enc, X_masked
    
