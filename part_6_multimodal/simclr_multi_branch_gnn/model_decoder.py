import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse


def replace_bn_with_gn(module):
    """Recursively replace all BatchNorm2d layers with GroupNorm."""
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
    """ResNet18 encoder with configurable input channels and GroupNorm."""

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
        return x


class ResidualBlock(nn.Module):
    """A simple residual block with dropout."""

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
    """A decoder block with transposed convolution and a residual block."""

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
        self.block = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.res_block = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.block(x)
        x = self.res_block(x)
        return x


class Decoder(nn.Module):
    """Decoder network for reconstructing from latent space."""

    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.project_dim = latent_dim // 2
        self.decoder = nn.Sequential(
            DecoderBlock(self.project_dim, self.project_dim // 2),
            DecoderBlock(self.project_dim // 2, self.project_dim // 4),
            DecoderBlock(self.project_dim // 4, self.project_dim // 8),
            DecoderBlock(self.project_dim // 8, self.project_dim // 16),
            DecoderBlock(self.project_dim // 16, self.project_dim // 32),
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


class GCN(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
        self.conv1 = GCNConv(in_channels, out_channels, add_self_loops=False)

    def forward(self, X, A):
        x = self.conv1(X, A)
        return x


class SIMCLR(nn.Module):
    """SIMCLR model with modal specific encoders and shared projection head."""

    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.in_channels = in_channels
        # Modal Shared Encoder
        encoders = []
        for _ in range(in_channels):
            encoder = ResNet18Encoder(1)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Shared Projection Head
        self.projector = nn.Sequential(
            nn.Linear(1000, latent_dim),
            nn.BatchNorm1d(latent_dim, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Fuse with a GNN
        self.gcn = GCN(
            in_channels=1000, hidden_channels=512, out_channels=1000
        )

    def encode(self, x):
        B, C, H, W = x.shape
        # Construct the adjacency Matrix for GNN
        A = torch.eye(C, device=x.device)
        # Modes are [temp, u_wind, v_wind, geo, humidity]
        # Temp and geo modes are related
        A[0, 3] = A[3, 0] = 1
        # Wind modes are related
        A[1, 2] = A[2, 1] = 1
        A, _ = dense_to_sparse(A)
        # Pass modes through encoders
        z_values = []
        for i in range(C):
            z = self.encoders[i](x[:, i, :, :].unsqueeze(1))
            z_values.append(z)

        # Batch of graphs of nodes of size 1000
        z_values = torch.stack(z_values, dim=1)
        _, _, L = z_values.shape
        # Package them up for torch geometric
        data_list = []
        for i in range(B):
            data = Data(x=z_values[i], edge_index=A)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        # Pass through GNN to fuse
        z = self.gcn(batch.x, batch.edge_index)
        z = global_mean_pool(z, batch.batch)
        return z

    def forward(self, x):
        z_x = self.encode(x)
        z = self.projector(z_x)
        return z, z_x


class SIMCLRDecoder(nn.Module):
    """SIMCLR model with an attached decoder for reconstruction."""

    def __init__(self, in_channels, model):
        super().__init__()
        self.model = model
        self.in_channels = in_channels
        self.decoder = Decoder(in_channels=in_channels, latent_dim=1000)

    def forward(self, x):
        z_x = self.model.encode(x)
        decoded_z = self.decoder(z_x)
        z = self.model.projector(z_x)
        return z, decoded_z, z_x
