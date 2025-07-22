import torch


class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, scale_factor=1, lambd=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_factor = scale_factor
        self.lambd = lambd

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        c = z1.T @ z2
        c.div_(z1.shape[0])
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)
        return loss
