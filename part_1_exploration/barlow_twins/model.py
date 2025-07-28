import torchvision
from torch import nn


class BarlowTwins(nn.Module):
    def __init__(self, in_channels, latent_dim, *args, **kwargs):
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
