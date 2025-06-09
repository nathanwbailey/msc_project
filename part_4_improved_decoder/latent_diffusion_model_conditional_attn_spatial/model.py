import math

import torch
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(
                torch.math.log(1.0), torch.math.log(1000), dim // 2
            )
        )
        self.register_buffer("ang_speeds", 2.0 * math.pi * frequencies)

    def forward(self, time):
        if isinstance(time, float):
            time = torch.tensor([time])
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        time = time.to(self.ang_speeds.device)
        embeddings = torch.cat(
            [
                torch.sin(time * self.ang_speeds),
                torch.cos(time * self.ang_speeds),
            ],
            dim=-1,
        )
        return embeddings


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(groups, in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=3, padding=1),
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.ReLU()

        self.bottleneck = None
        if in_channels != out_channels:
            self.bottleneck = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_add = x
        if self.bottleneck is not None:
            x_add = self.bottleneck(x)
        x = x_add + self.block(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class CrossAttention2D(nn.Module):
    def __init__(self, channels, n_heads=4, groups=32):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, n_heads)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels),
        )
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x, cond_tokens):
        # Shape data into tokens
        x = self.norm1(x)
        B, C, H, W = x.shape
        q = x.reshape(B, C, H*W).permute(2, 0, 1) # H*W, B, C
        k_v = cond_tokens.reshape(B, C, H*W).permute(2, 0, 1) # H*W, B, C
        attn_out, _ = self.attn(q, k_v, k_v) # H*W, B, C
        # Reshape back to data
        # B, C, H*W
        h = attn_out.permute(1, 2, 0)
        h = h.reshape(B, C, H, W)
        h = h.permute(0, 2, 3, 1) # (B, H, W, C)
        # Per pixel MLP: C -> 4C -> C
        h = self.mlp(h)
        h = h.permute(0, 3, 1, 2) # (B, C, H, W)
        out = self.norm2(x + h)
        return out

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=32, up=None):
        super().__init__()
        self.time_layer = nn.Linear(time_emb_dim, in_channels)
        if up is not None:
            if not up:
                self.block_in = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2),
                    nn.ReLU(),
                )
            else:
                
                self.block_in = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2),
                    nn.ReLU(),
                )
                self.time_layer = nn.Linear(time_emb_dim, in_channels // 2)
                in_channels = in_channels // 2
        else:
            self.block_in = nn.Identity()
        self.block_middle = ResidualBlock2D(in_channels, in_channels, groups=groups)
        self.cross_attn = CrossAttention2D(in_channels, groups=groups)
        self.block_out = ResidualBlock2D(in_channels, out_channels, groups=groups)

    def forward(self, x, cond_tokens, time):
        x = self.block_in(x)
        time = self.time_layer(time)
        time = time[..., None, None]
        x = x + time
        x = self.block_middle(x)
        x = self.cross_attn(x, cond_tokens)
        x = self.block_out(x)
        return x


class LatentNetwork(nn.Module):
    def __init__(self, latent_dim, time_emb_dim, emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_model = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )


        self.cond_token_proj = nn.Linear(latent_dim, emb_dim)
        self.cond_projs_enc = nn.ModuleList(
            [
                nn.Conv2d(in_channels=16, out_channels=32, stride=2, kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=64, stride=4, kernel_size=4),
                nn.Conv2d(in_channels=16, out_channels=128, stride=4, kernel_size=4),
            ]
        )

        self.cond_projs_dec = nn.ModuleList(
            [
                nn.Conv2d(in_channels=16, out_channels=128, stride=2, kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=64, stride=1, kernel_size=1),
            ]
        )

        self.reshape_layer_in = nn.Linear(latent_dim, emb_dim)
        self.initial_layer = ResidualBlock2D(in_channels=16, out_channels=32, groups=16)
        self.initial_time_layer = nn.Linear(time_emb_dim, 16)

        self.enc = nn.ModuleList(
            [
                Block(32, 64, time_emb_dim, up=False),
                Block(64, 128, time_emb_dim, up=False),
            ]
        )
        self.bottleneck = Block(128, 128, time_emb_dim, up=None)
        self.dec = nn.ModuleList(
            [
                Block(256, 64, time_emb_dim, up=True),
                Block(128, 32, time_emb_dim, up=True),
            ]
        )
        self.out_layer = ResidualBlock2D(32, 16, groups=16)
        self.out_time_layer = nn.Linear(time_emb_dim, 32)
        self.reshape_layer_out = nn.Linear(emb_dim, latent_dim)

    def forward(self, x, time, condition):
        
        # Reshape Latent Vector to be Spatial
        x = self.reshape_layer_in(x)
        B, _ = x.shape
        x = x.reshape(B, 16, 8, 8)

        
        
        t = self.time_model(time)
        # First Layer
        t_1 = self.initial_time_layer(t)
        t_1 = t_1[..., None, None]
        x = x + t_1
        h = self.initial_layer(x)

        cond_tokens = self.cond_token_proj(condition)
        cond_tokens = cond_tokens.reshape(B, 16, 8, 8)

        cond_tokens_list_enc = []
        for proj in self.cond_projs_enc:
            feats = proj(cond_tokens)
            cond_tokens_list_enc.append(feats)

        cond_tokens_list_dec = []
        for proj in self.cond_projs_dec:
            feats = proj(cond_tokens)
            cond_tokens_list_dec.append(feats)


        skips = []
        for layer, cond_tokens in zip(self.enc, cond_tokens_list_enc):
            h = layer(h, cond_tokens, t)
            skips.append(h)
        
        h = self.bottleneck(h, cond_tokens_list_enc[-1], t)

        for layer, cond_tokens in zip(self.dec, cond_tokens_list_dec):
            skip = skips.pop()
            h = torch.cat((h, skip), dim=1)
            h = layer(h, cond_tokens, t)

        t_out = self.out_time_layer(t)
        t_out = t_out[..., None, None]
        h = h + t_out
        h = self.out_layer(h)
        h = h.reshape(B, -1)
        z_out = self.reshape_layer_out(h)
        return z_out
