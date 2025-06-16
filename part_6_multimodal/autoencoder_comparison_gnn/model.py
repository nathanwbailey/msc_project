import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import global_mean_pool


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
        return x


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

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
        self.conv1 = GCNConv(in_channels, out_channels, add_self_loops=False)

    def forward(self, X, A):
        x = self.conv1(X, A)
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


class AutoEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        super().__init__()
        self.in_channels = in_channels
        # Modal Shared Encoders
        encoders = []
        for _ in range(in_channels):
            encoder = ResNet18Encoder(1)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        self.decoder = Decoder(in_channels=in_channels, latent_dim=1000)

        # Fuse with a GNN
        self.gcn = GCN(in_channels=1000, hidden_channels=512, out_channels=1000)


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

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out
