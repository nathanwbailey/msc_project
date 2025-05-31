import torch

def get_loss_weights(idx, batch_size, max_val, alpha_decay=10):
    """
    Computes loss weights for contrastive learning based on index differences.
    Excludes self and positive pairs.
    """
    torch.set_printoptions(threshold=float('inf'))
    idx_full = torch.cat((idx, idx), dim=0)
    diff = torch.abs(idx_full.unsqueeze(0) - idx_full.unsqueeze(1))
    diff_vals = diff / max_val

    # Mask for self-comparisons
    self_mask = torch.eye(len(idx_full), dtype=torch.bool, device=idx.device)
    # Mask for positive pairs
    pos_mask = torch.zeros_like(self_mask)
    pos_mask[range(batch_size), range(batch_size, 2 * batch_size)] = True
    pos_mask[range(batch_size, 2 * batch_size), range(batch_size)] = True

    # Exclude self and positive pairs
    exclude_mask = self_mask | pos_mask
    neg_mask = ~exclude_mask
    
    # Compute weights
    # Confirm we have the right size before flatten
    neg_diff = diff_vals[neg_mask].view(2 * batch_size, 2 * batch_size - 2).reshape(-1).float()
    weights = torch.sigmoid(alpha_decay*neg_diff)
    return weights