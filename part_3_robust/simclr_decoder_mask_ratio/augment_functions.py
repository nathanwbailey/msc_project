import torch
import torch.nn.functional as F
import torchvision.transforms as T
import random

def random_crop(sample):
    sample = F.interpolate(sample, size=(160, 80), mode='bicubic', align_corners=False)
    crop = T.RandomCrop((144, 72))
    sample = crop(sample)
    return sample

def resize_encoder(sample):
    sample = sample.unsqueeze(0)
    sample = F.interpolate(sample, size=(144, 72), mode='bicubic', align_corners=False)
    return sample.squeeze(0)

def smooth(sample):
    K = 5
    padding = K // 2
    C = sample.shape[1]
    mean_kernel = torch.ones((C, 1, K, K), dtype=torch.float32) / (K * K)
    sample = F.conv2d(sample, weight=mean_kernel, padding=padding, groups=C)
    return sample

def random_mask(sample, mask_prob_low=0.5, mask_prob_high=0.9):
    mask_prob = random.uniform(mask_prob_low, mask_prob_high)
    random_tensor = torch.rand(sample.shape, device=sample.device)
    mask = (random_tensor > mask_prob).float()
    masked_image = sample * mask
    return masked_image

def augment_sample_random_mask(sample, mask_prob_low=0.5, mask_prob_high=0.9):
    sample = random_crop(sample.unsqueeze(0)).squeeze(0)
    sample = smooth(sample.unsqueeze(0)).squeeze(0)
    sample = random_mask(sample, mask_prob_low=mask_prob_low, mask_prob_high=mask_prob_high)
    return sample

def augment_sample(sample):
    sample = random_crop(sample.unsqueeze(0)).squeeze(0)
    sample = smooth(sample.unsqueeze(0)).squeeze(0)
    return sample
