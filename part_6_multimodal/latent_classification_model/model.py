import torch
from torch import nn


class LatentClassificationModel(nn.Module):
    def __init__(self, latent_dim, num_labels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.BatchNorm1d(latent_dim // 4),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(latent_dim // 4, latent_dim // 16),
            nn.BatchNorm1d(latent_dim // 16),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(latent_dim // 16, num_labels),
        )

    def forward(self, x):
        x = self.model(x)
        return x
