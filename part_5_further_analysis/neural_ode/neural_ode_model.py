import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchdiffeq import odeint

"""
see: https://github.com/rtqichen/torchdiffeq
"""


class ODEF(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, dim),
        )

    # def forward(self, t, z):
    #     t_vec = t * torch.ones(z.shape[0], 1, device=z.device)
    #     z_inp = torch.cat((z, t_vec), dim=1)
    #     dz = self.net(z_inp)
    #     return dz
    
    def forward(self, t, z):
        if z.dim() == 2: # [B, D]
            B, D = z.shape
            t_vec = t.unsqueeze(0).expand(B).unsqueeze(1) # [B, 1]
            z_inp = torch.cat((z, t_vec), dim=-1) # [B, D+1]
            dz = self.net(z_inp)
            return dz

        elif z.dim() == 3: # [B, T, D]
            B, T, D = z.shape
            # t: [T] -> t_vec: [B, T, 1]
            t_vec = t.unsqueeze(0).expand(B, T).unsqueeze(2)
            z_inp = torch.cat((z, t_vec), dim=-1) # [B, T, D+1]
            # flatten, push through MLP
            flat  = z_inp.reshape(B * T, D + 1)
            out = self.net(flat)
            dz = out.reshape(B, T, D)
            return dz
        else:
            raise ValueError(f"Unsupported z.dim()=={z.dim()}, expected 2 or 3.")

class NeuralODE(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.odefunc = ODEF(dim)

    def forward(self, z0, t_seq):
        return odeint(self.odefunc, z0, t_seq)


class LatentTestDataset(Dataset):
    def __init__(self, data, context_length, stride):
        super().__init__()
        self.data = data
        self.L = context_length
        self.stride = stride
        self.times = torch.linspace(0, 1, steps=context_length).to(
            data.device
        )

    def __len__(self):
        return (len(self.data) - self.L) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        end_context = start + self.L
        z_window = self.data[start:end_context]
        z0 = z_window[0]
        return z0, self.times, z_window


def train_neural_ode(
    num_epochs, model, trainloader, optimizer, loss_fn, device
):
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = []
        for batch in trainloader:
            optimizer.zero_grad()
            z0 = batch[0].to(device)
            t_seq = batch[1][0].to(device)
            z_true = batch[2].to(device)
            z_pred = model(z0, t_seq).permute(1, 0, 2)
            loss = loss_fn(z_pred, z_true)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        print(f"Epoch: {epoch}, Train Loss: {np.mean(train_loss):.4f}")
    return model
