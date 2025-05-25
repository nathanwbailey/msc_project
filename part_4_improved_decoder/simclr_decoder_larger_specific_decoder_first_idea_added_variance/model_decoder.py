import torch
import torch.nn.functional as F
import torchvision
from torch import nn


def replace_bn_with_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            gn = nn.GroupNorm(
                num_groups=num_channels, num_channels=num_channels
            )
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child)


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
        replace_bn_with_gn(resnet)

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
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        return self.dropout(self.relu(x + self.block(x)))


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
        )
        self.res_block = ResidualBlock(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.block(x)
        x = self.res_block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_dim = latent_dim // 2
        self.decoder = nn.Sequential(
            DecoderBlock(
                in_channels=self.project_dim,
                out_channels=self.project_dim // 2,
            ),  # 2x1 -> 4x2
            DecoderBlock(
                in_channels=self.project_dim // 2,
                out_channels=self.project_dim // 4,
            ),  # 4x2 -> 8x4
            DecoderBlock(
                in_channels=self.project_dim // 4,
                out_channels=self.project_dim // 8,
            ),  # 8x4 -> 16x8
            DecoderBlock(
                in_channels=self.project_dim // 8,
                out_channels=self.project_dim // 16,
            ),  # 16x8 -> 32x16
            DecoderBlock(
                in_channels=self.project_dim // 16,
                out_channels=self.project_dim // 32,
            ),  # 32x16 -> 64x32
        )
        self.channel_layer = nn.Conv2d(
            in_channels=self.project_dim // 32,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], self.project_dim, 2, 1))
        x = self.decoder(x)
        x = self.channel_layer(x)
        return x


class SIMCLR(nn.Module):
    def __init__(self, in_channels, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = ResNet18Encoder(in_channels)

        self.projector = nn.Sequential(
            nn.Linear(1000, latent_dim),
            nn.BatchNorm1d(latent_dim, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def encode(self, x):
        z, skip = self.encoder(x)
        return z, skip

    def forward(self, x):
        z_x, skip = self.encode(x)
        z = self.projector(z_x)
        return z, z_x


class SIMCLRDecoder(nn.Module):
    def __init__(self, in_channels, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.in_channels = in_channels
        self.decoder = Decoder(in_channels=in_channels, latent_dim=1000)

    def forward(self, x):
        z_x, skip = self.model.encode(x)
        decoded_z = self.decoder(z_x)
        z = self.model.projector(z_x)
        return z, decoded_z, z_x
