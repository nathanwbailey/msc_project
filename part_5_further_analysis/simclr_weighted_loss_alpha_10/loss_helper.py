import torch


def get_loss_weights(idx, batch_size, alpha_decay=10):
    """
    Computes loss weights for contrastive learning based on index differences.
    Excludes self and positive pairs.
    """
    idx_full = torch.cat((idx, idx), dim=0)
    diff = torch.abs(idx_full.unsqueeze(0) - idx_full.unsqueeze(1))

    # Mask for self-comparisons
    self_mask = torch.eye(len(idx_full), dtype=torch.bool, device=idx.device)
    # Mask for positive pairs
    pos_mask = torch.zeros_like(self_mask)
    pos_mask[range(batch_size), range(batch_size, 2 * batch_size)] = True
    pos_mask[range(batch_size, 2 * batch_size), range(batch_size)] = True

    # Exclude self and positive pairs
    exclude_mask = self_mask | pos_mask
    neg_mask = ~exclude_mask

    # Compute and normalize weights
    neg_diff = (
        diff[neg_mask]
        .view(2 * batch_size, 2 * batch_size - 2)
        .reshape(-1)
        .float()
    )
    weights = torch.exp(-alpha_decay * (1 / neg_diff).float())
    return weights
