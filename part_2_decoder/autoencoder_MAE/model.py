import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        resnet = torchvision.models.resnet18()
        resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x):
        x0 = self.stem(x)
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, [x3, x2, x1, x0]


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        padding=0,
        output_padding=0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
        )
        self.res_block = ResidualBlock(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.block(x)
        x = self.res_block(x)
        return x


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


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels=in_channels)
        self.fc1 = nn.Linear(1000, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 1000)
        self.decoder = Decoder(
            in_channels=in_channels, initial_kernel=(8, 4), latent_dim=1000
        )

    def encode(self, x):
        encoded_data, _ = self.encoder(x)
        return encoded_data

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out
