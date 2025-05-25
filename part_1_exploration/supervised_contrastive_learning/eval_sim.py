import torch
import numpy as np
import torch.nn.functional as F

def negative_cosine_sim_matrix(embeddings):
    z = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(z, z.T)
    N = sim_matrix.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    negative_sims = sim_matrix[mask]
    return negative_sims

def eval_model(model, testloader, device):
    cos_sim_mean = []
    rand_cos_sim_mean = []
    mean_var = []
    with torch.no_grad():
        for batch in testloader:
            X = batch[0].to(device)
            X_prime = batch[1].to(device)
            embeddings_x = model(X).cpu()
            embeddings_x_prime = model(X_prime).cpu()

            cos_sim = F.cosine_similarity(embeddings_x, embeddings_x_prime, dim=1)
            cos_sim_mean.append(cos_sim.mean().item())
    
            # rand_indices = torch.randperm(embeddings_x.shape[0])
            # cos_sim = F.cosine_similarity(embeddings_x, embeddings_x_prime[rand_indices], dim=1)
            # rand_cos_sim_mean.append(cos_sim.mean().item())

            neg_cosine = negative_cosine_sim_matrix(embeddings_x)
            rand_cos_sim_mean.append(neg_cosine.mean().item())

            variance = embeddings_x.var(dim=0)
            mean_var.append(variance.mean().item())
    
    return np.mean(cos_sim_mean), np.mean(rand_cos_sim_mean), np.mean(mean_var)