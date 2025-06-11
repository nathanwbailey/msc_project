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


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.BatchNorm1d(in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, out_dim),
        )
        self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()

        self.bottleneck = None
        if in_dim != out_dim:
            self.bottleneck = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x_add = x
        if self.bottleneck is not None:
            x_add = self.bottleneck(x)
        x = x_add + self.block(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class CrossAttention1D(nn.Module):
    def __init__(self, d_model, n_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, cond_tokens):
        q = x.unsqueeze(0)
        cond_tokens = cond_tokens.unsqueeze(0)
        attn_out, _ = self.attn(q, cond_tokens, cond_tokens)
        y = attn_out.squeeze(0)
        x = self.norm1(x + y)
        x = self.norm2(x + self.mlp(x))
        return x


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, time_emb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = ResidualBlock(in_dim, out_dim)
        self.cross_attn = CrossAttention1D(out_dim)
        self.layer_out = ResidualBlock(out_dim, out_dim)
        self.time_layer = nn.Linear(time_emb, in_dim)

    def forward(self, x, cond_tokens, time):
        time = self.time_layer(time)
        x = x + time
        x = self.layer(x)
        x = self.cross_attn(x, cond_tokens)
        x = self.layer_out(x)
        return x


class LatentNetwork(nn.Module):
    def __init__(self, latent_dim, time_emb_dim, emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_model = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.cond_projs_enc = nn.ModuleList(
            [
                nn.Linear(latent_dim, emb_dim // 2),
                nn.Linear(latent_dim, emb_dim // 4),
                nn.Linear(latent_dim, emb_dim // 8),
                nn.Linear(latent_dim, emb_dim // 16),
                nn.Linear(latent_dim, emb_dim // 32),
                nn.Linear(latent_dim, emb_dim // 32),
            ]
        )

        self.cond_projs_dec = nn.ModuleList(
            [
                nn.Linear(latent_dim, emb_dim // 16),
                nn.Linear(latent_dim, emb_dim // 8),
                nn.Linear(latent_dim, emb_dim // 4),
                nn.Linear(latent_dim, emb_dim // 2),
                nn.Linear(latent_dim, emb_dim),
            ]
        )
        initial_layer = nn.Sequential(
            nn.Linear(latent_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

        self.initial_layer = initial_layer
        self.initial_time_layer = nn.Linear(time_emb_dim, latent_dim)
        self.enc = nn.ModuleList(
            [
                Block(emb_dim, emb_dim // 2, time_emb_dim),
                Block(emb_dim // 2, emb_dim // 4, time_emb_dim),
                Block(emb_dim // 4, emb_dim // 8, time_emb_dim),
                Block(emb_dim // 8, emb_dim // 16, time_emb_dim),
                Block(emb_dim // 16, emb_dim // 32, time_emb_dim),
            ]
        )
        self.bottleneck = Block(emb_dim // 32, emb_dim // 32, time_emb_dim)
        self.dec = nn.ModuleList(
            [
                Block(emb_dim // 16, emb_dim // 16, time_emb_dim),
                Block(emb_dim // 8, emb_dim // 8, time_emb_dim),
                Block(emb_dim // 4, emb_dim // 4, time_emb_dim),
                Block(emb_dim // 2, emb_dim // 2, time_emb_dim),
                Block(emb_dim, emb_dim, time_emb_dim),
            ]
        )
        self.out_time_layer = nn.Linear(time_emb_dim, emb_dim * 2)
        self.out_layer = nn.Linear(emb_dim * 2, latent_dim)

    def forward(self, x, time, condition):
        time_emb = self.time_model(time)
        time_emb_init = self.initial_time_layer(time_emb)
        x = x + time_emb_init

        cond_tokens_list_enc = []
        for proj in self.cond_projs_enc:
            feats = proj(condition)
            cond_tokens_list_enc.append(feats)

        cond_tokens_list_dec = []
        for proj in self.cond_projs_dec:
            feats = proj(condition)
            cond_tokens_list_dec.append(feats)

        skips = []
        x = self.initial_layer(x)
        skips.append(x)
        for layer, cond_tokens in zip(self.enc, cond_tokens_list_enc[:-1]):
            x = layer(x, cond_tokens, time_emb)
            skips.append(x)

        x = self.bottleneck(x, cond_tokens_list_enc[-1], time_emb)

        for layer, cond_tokens in zip(self.dec, cond_tokens_list_dec):
            # x = x + skips.pop()
            x = torch.cat((x, skips.pop()), dim=1)
            x = layer(x, cond_tokens, time_emb)
        x = torch.cat((x, skips.pop()), dim=1)
        time_emb_out = self.out_time_layer(time_emb)
        x = x + time_emb_out
        x = self.out_layer(x)
        return x
