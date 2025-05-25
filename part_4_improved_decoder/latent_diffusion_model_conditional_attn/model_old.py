import torch
from torch import nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(
                torch.math.log(1.0),
                torch.math.log(1000),
                dim//2
            )
        )
        self.register_buffer("ang_speeds", 2.0 * math.pi * frequencies)
    def forward(self, time):
        if isinstance(time, float):
            time = torch.tensor([time])
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        time = time.to(self.ang_speeds.device)
        embeddings = torch.cat([torch.sin(time*self.ang_speeds), torch.cos(time*self.ang_speeds)], dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, in_dim*2),
            nn.BatchNorm1d(in_dim*2),
            nn.ReLU(),
            nn.Linear(in_dim*2, out_dim),
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
    
# class CrossAttention1D_v2(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.attn  = nn.MultiheadAttention(embed_dim=1, num_heads=1)
#         self.norm1 = nn.LayerNorm(1)
#         self.mlp   = nn.Sequential(
#             nn.Linear(1, 4),
#             nn.ReLU(),
#             nn.Linear(4, 1),
#         )
#         self.norm2 = nn.LayerNorm(1)

#     def forward(self, x, cond_tokens):
#         B, D = x.shape
#         q = x.transpose(0, 1).unsqueeze(-1) # (D, B, 1)
#         k = cond_tokens.transpose(0, 1).unsqueeze(-1)
#         v = k
#         attn_out, _ = self.attn(q, k, v) # (D, B, 1)
#         y = attn_out.squeeze(-1).transpose(0, 1) # (B, D)
#         y = y.view(-1, 1)
#         out = self.norm1(y + x.view(-1, 1))
#         out = self.norm2(out + self.mlp(out))
#         out = out.view(B, D)
#         return x + out



class LatentNetwork(nn.Module):
    def __init__(self, latent_dim, time_emb_dim, emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_model = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, latent_dim)
        )

        self.cond_projs_enc = nn.ModuleList(
            [
                nn.Linear(latent_dim, emb_dim),
                nn.Linear(latent_dim, emb_dim//2),
                nn.Linear(latent_dim, emb_dim//4)
            ]
        )

        self.cond_projs_dec = nn.ModuleList(
            [
                nn.Linear(latent_dim, emb_dim//2),
                nn.Linear(latent_dim, emb_dim),
                nn.Linear(latent_dim, latent_dim),
            ]
        )

        self.cross_atts_enc = nn.ModuleList(
            [
                CrossAttention1D(emb_dim),
                CrossAttention1D(emb_dim//2),
                CrossAttention1D(emb_dim//4)
            ]
        )

        self.cross_atts_dec = nn.ModuleList(
            [
                CrossAttention1D(emb_dim//2),
                CrossAttention1D(emb_dim),
                CrossAttention1D(latent_dim)
            ]
        )

        initial_layer = nn.Sequential(
            nn.Linear(latent_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )
        self.enc = nn.ModuleList(
            [
                initial_layer,
                ResidualBlock(emb_dim, emb_dim//2),
                ResidualBlock(emb_dim//2, emb_dim//4),
            ]
        )
        self.bottleneck = ResidualBlock(emb_dim//4, emb_dim//4)
        self.dec = nn.ModuleList(
            [
                ResidualBlock(emb_dim//4, emb_dim//2),
                ResidualBlock(emb_dim//2, emb_dim),
                nn.Linear(emb_dim, latent_dim),
            ]
        )

    def forward(self, x, time, condition):
        time_emb = self.time_model(time)
        x = x + time_emb

        cond_tokens_list_enc = []
        for proj in self.cond_projs_enc:
            feats = proj(condition)
            cond_tokens_list_enc.append(feats)

        cond_tokens_list_dec = []
        for proj in self.cond_projs_dec:
            feats = proj(condition)
            cond_tokens_list_dec.append(feats)

        skips = []
        for layer, cond_tokens, cross in zip(self.enc, cond_tokens_list_enc, self.cross_atts_enc):
            x_attn = layer(x)
            x = cross(x_attn, cond_tokens)
            skips.append(x)

        x = self.bottleneck(x)

        for layer, cond_tokens, cross in zip(self.dec, cond_tokens_list_dec, self.cross_atts_dec):
            x = x + skips.pop()
            x_attn = layer(x)
            x = cross(x_attn, cond_tokens)
        return x
    
