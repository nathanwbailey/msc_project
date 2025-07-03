import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.utils import common_functions as c_f


class WeightedLossWrapper(torch.nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = None

    def set_weights(self, weights):
        self.weights = weights

    def forward(self, embeddings, labels):
        self.loss_fn.weights = self.weights
        return self.loss_fn(embeddings, labels)


class WeightedNTXentLoss(NTXentLoss):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(temperature=temperature, **kwargs)
        self.weights = None

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            if self.weights is not None:
                neg_pairs = self.weights * (neg_pairs / self.temperature)
            else:
                neg_pairs = neg_pairs / self.temperature

            n_per_p = c_f.to_dtype(
                a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype
            )
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = (
                torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            )
            log_exp = torch.log(
                (numerator / denominator) + c_f.small_val(dtype)
            )
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()
