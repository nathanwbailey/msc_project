import torch
import torch.nn.functional as F
import torchvision
from torch import nn

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
#             nn.LeakyReLU(),
#         )
#     def forward(self, x):
#         return self.block(x)

# class SupConModel(nn.Module):
#     def __init__(self, in_channels, latent_dim, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.encoder = nn.Sequential(
#             ConvBlock(in_channels=in_channels, out_channels=32),
#             ConvBlock(in_channels=32, out_channels=64),
#             ConvBlock(in_channels=64, out_channels=128),
#             ConvBlock(in_channels=128, out_channels=128),
#             ConvBlock(in_channels=128, out_channels=256),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#         )
#         self.fc1 = nn.Linear(256, 1024)
#         self.fc2 = nn.Linear(1024, latent_dim)
#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return F.normalize(x, dim=1)


class SupConModel(nn.Module):
    def __init__(self, in_channels, latent_dim, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = torchvision.models.resnet18()
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.projector_network = nn.Sequential(
            nn.Linear(1000, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.reshape(x.size(0), -1)
        z = self.projector_network(z)
        return F.normalize(z, dim=1)
