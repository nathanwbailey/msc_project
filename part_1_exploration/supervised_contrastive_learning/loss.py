import torch
import torch.nn.functional as F

class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
    
    def forward(self, features, labels):
        device = features.device

        mask = torch.eq(labels, labels.T).float().to(device)