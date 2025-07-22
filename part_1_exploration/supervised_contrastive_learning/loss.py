import torch


class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        torch.eq(labels, labels.T).float().to(device)
