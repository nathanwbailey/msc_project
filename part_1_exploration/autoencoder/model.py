import torch
from torch import nn
import torch.nn.functional as F
import torchvision

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.block(x)
    
class SupConModel(nn.Module):
    def __init__(self, in_channels, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=32),
            ConvBlock(in_channels=32, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, latent_dim)
    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return F.normalize(x, dim=1)


# class SupConModel(nn.Module):
#     def __init__(self, in_channels, projection_dim, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         encoder = torchvision.models.resnet18()
#         encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         layers = torch.nn.Sequential(*list(encoder.children()))
#         last_layer = layers[-1]
#         self.features_dim = last_layer.in_features
#         self.encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
#         self.fc1 = nn.Linear(self.features_dim, 2048)
#         self.fc2 = nn.Linear(2048, projection_dim)
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return F.normalize(x, dim=1)
