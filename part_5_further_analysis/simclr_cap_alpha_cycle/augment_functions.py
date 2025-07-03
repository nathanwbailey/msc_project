import random

import torch
import torch.nn.functional as F
import torchvision.transforms as T


def random_crop(sample):
    sample = F.interpolate(
        sample, size=(160, 80), mode="bicubic", align_corners=False
    )
    crop = T.RandomCrop((144, 72))
    sample = crop(sample)
    return sample


def resize_to_orig(sample):
    sample = sample.unsqueeze(0)
    sample = F.interpolate(
        sample, size=(64, 32), mode="bicubic", align_corners=False
    )
    return sample.squeeze(0)


def resize_encoder(sample):
    sample = sample.unsqueeze(0)
    sample = F.interpolate(
        sample, size=(144, 72), mode="bicubic", align_corners=False
    )
    return sample.squeeze(0)


def smooth(sample):
    K = 5
    padding = K // 2
    C = sample.shape[1]
    mean_kernel = torch.ones((C, 1, K, K), dtype=torch.float32) / (K * K)
    sample = F.conv2d(sample, weight=mean_kernel, padding=padding, groups=C)
    return sample


def add_noise(sample):
    noise = torch.randn(size=sample.shape)
    return sample + noise


def flip(sample, p=0.5):
    p_flip = random.uniform(0, 1)
    if p_flip > p:
        sample = sample.flip(dims=[-1])
    return sample


def shuffle_channels(sample):
    channel_dim_order = torch.randperm(sample.shape[0])
    return sample[channel_dim_order, :, :]


def cutout(sample):
    erase = T.RandomErasing(
        p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0
    )
    sample = erase(sample)
    return sample


def gaussian_blur(sample):
    blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    sample = blur(sample)
    return sample


def random_mask(sample, mask_prob_low=0.5, mask_prob_high=0.9):
    mask_prob = random.uniform(mask_prob_low, mask_prob_high)
    random_tensor = torch.rand(sample.shape, device=sample.device)
    mask = (random_tensor > mask_prob).float()
    masked_image = sample * mask
    return masked_image


def augment_sample_random_mask(sample, mask_prob_low=0.5, mask_prob_high=0.9):
    sample = random_crop(sample.unsqueeze(0)).squeeze(0)
    sample_to_mask = smooth(sample.unsqueeze(0)).squeeze(0)
    sample = random_mask(
        sample_to_mask,
        mask_prob_low=mask_prob_low,
        mask_prob_high=mask_prob_high,
    )
    return sample


def augment_sample(sample):
    sample = random_crop(sample.unsqueeze(0)).squeeze(0)
    sample = smooth(sample.unsqueeze(0)).squeeze(0)
    return sample
