import torch
from torch import nn
import torch.nn.functional as F
import torchvision

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
#             nn.LeakyReLU(),
#         )
#     def forward(self, x):
#         return self.block(x)


# class BarlowTwins(nn.Module):
#     def __init__(self, in_channels, latent_dim, embedding_dim, dropout, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.encoder = nn.Sequential(
#             ConvBlock(in_channels=in_channels, out_channels=32, dropout=dropout),
#             ConvBlock(in_channels=32, out_channels=64, dropout=dropout),
#             ConvBlock(in_channels=64, out_channels=128, dropout=dropout),
#             ConvBlock(in_channels=128, out_channels=128, dropout=dropout),
#             ConvBlock(in_channels=128, out_channels=256, dropout=dropout),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#         )
#         self.encoder_repr = nn.Linear(256, latent_dim)
#         self.projector_network = nn.Sequential(
#             nn.BatchNorm1d(latent_dim),
#             nn.ReLU(),
#             nn.Linear(latent_dim, embedding_dim, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm1d(embedding_dim),
#             nn.Linear(embedding_dim, embedding_dim, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm1d(embedding_dim),
#             nn.Linear(embedding_dim, embedding_dim, bias=False),
#         )

#         # self.fc1 = nn.Linear(256, latent_dim)
#         # self.fc1 = nn.Linear(latent_dim, latent_dim)
#         # self.fc2 = nn.Linear(latent_dim, embedding_dim)
#         self.bn = nn.BatchNorm1d(embedding_dim, affine=False)
    
#     def forward(self, x1, x2):
#         z1 = self.encoder(x1)
#         z2 = self.encoder(x2)
#         z1 = z1.reshape(x1.size(0), -1)
#         z2 = z2.reshape(x2.size(0), -1)
#         z1 = self.encoder_repr(z1)
#         z1 = self.projector_network(z1)
#         z2 = self.encoder_repr(z2)
#         z2 = self.projector_network(z2)
#         return self.bn(z1), self.bn(z2)


class BarlowTwins(nn.Module):
    def __init__(self, in_channels, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = torchvision.models.resnet18()
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.projector = nn.Sequential(
            nn.Linear(1000, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.bn = nn.BatchNorm1d(latent_dim, affine=False)
    
    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        z1 = z1.reshape(x1.size(0), -1)
        z2 = z2.reshape(x2.size(0), -1)

        z1 = self.projector(z1)
        z2 = self.projector(z2)
        return self.bn(z1), self.bn(z2)
