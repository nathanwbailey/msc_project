import torch
import torchvision
from torch import nn


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self, in_channels, initial_kernel, latent_dim, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.project_dim = 128
        self.project_layer = nn.Linear(latent_dim, 1024)
        self.decoder = nn.Sequential(
            DecoderBlock(
                in_channels=self.project_dim,
                out_channels=self.project_dim // 2,
            ),  # 4x2 -> 8x4
            DecoderBlock(
                in_channels=self.project_dim // 2,
                out_channels=self.project_dim // 4,
            ),  # 8x4 -> 16x8
            DecoderBlock(
                in_channels=self.project_dim // 4,
                out_channels=self.project_dim // 8,
            ),  # 16x8 -> 32x16
            DecoderBlock(
                in_channels=self.project_dim // 8,
                out_channels=self.project_dim // 16,
            ),  # 32x16 -> 64x32
        )
        self.channel_layer = nn.Conv2d(
            in_channels=self.project_dim // 16,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        x = self.project_layer(x)
        x = torch.reshape(x, (x.shape[0], self.project_dim, 4, 2))
        x = self.decoder(x)
        x = self.channel_layer(x)
        return x


class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.encoder = torchvision.models.resnet18()
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.decoder = self.decoder = Decoder(
            in_channels=in_channels, initial_kernel=(8, 4), latent_dim=1000
        )

    def encode(self, x):
        encoded_data = self.encoder(x)
        return encoded_data

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out