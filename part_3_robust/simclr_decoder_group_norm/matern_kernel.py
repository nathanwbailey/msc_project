import torch


def matern_kernel(x, lengthscale=1.0, nu=2.5, sigma=1.0):
    """
    Closed form solutions for Matern Kernel.
    """
    # See : https://gaussianprocess.org/gpml/chapters/RW4.pdf
    # See : https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
    # Distance from each point to itself
    dist = torch.cdist(x, x, p=2) + 1e-6  # Avoid zero distance
    sqrt = (
        torch.sqrt(2 * torch.tensor(nu, device=x.device)) * dist / lengthscale
    )
    if nu == 0.5:
        K = torch.exp(-sqrt)
    elif nu == 1.5:
        K = (1 + sqrt) * torch.exp(-sqrt)
    elif nu == 2.5:
        K = (1 + sqrt + (sqrt**2) / 3.0) * torch.exp(-sqrt)
    else:
        raise NotImplementedError("Only ν = 0.5, 1.5, 2.5 are implemented")
    return sigma**2 * K


def matern_kernel_noise(sample, coords, lengthscale=1.0, nu=2.5, sigma=1.0):
    C, H, W = sample.shape
    N = H * W
    # Matern kernel gives the co-variance
    cov = matern_kernel(coords, lengthscale=lengthscale, nu=nu, sigma=sigma)
    # Normal
    z = torch.randn(C, N, device=cov.device)
    # Ensure the covariance matrix is positive definite
    L = torch.linalg.cholesky(cov + 1e-2 * torch.eye(N, device=sample.device))
    # Noise with a mean of 0
    # Covariance of Cov
    noise = (z @ L.T).reshape(C, H, W)
    return sample + noise


def matern_kernel_noise_batch(
    sample, coords, lengthscale=1.0, nu=2.5, sigma=1.0
):
    B, C, H, W = sample.shape
    N = H * W
    # Matern kernel gives the co-variance
    cov = matern_kernel(coords, lengthscale=lengthscale, nu=nu, sigma=sigma)
    # Normal
    z = torch.randn(B, C, N, device=cov.device)
    # Ensure the covariance matrix is positive definite
    L = torch.linalg.cholesky(cov + 1e-2 * torch.eye(N, device=sample.device))
    # Noise with a mean of 0
    # Covariance of Cov
    noise = (z @ L.T).reshape(B, C, H, W)
    return sample + noise


def matern_kernel_noise_time_batch(
    sample, coords, lengthscale=1.0, nu=2.5, sigma=1.0
):
    B, T, C, H, W = sample.shape
    N = H * W
    # Matern kernel gives the co-variance
    cov = matern_kernel(coords, lengthscale=lengthscale, nu=nu, sigma=sigma)
    # Normal
    z = torch.randn(B, T, C, N, device=cov.device)
    # Ensure the covariance matrix is positive definite
    L = torch.linalg.cholesky(cov + 1e-2 * torch.eye(N, device=sample.device))
    # Noise with a mean of 0
    # Covariance of Cov
    noise = (z @ L.T).reshape(B, T, C, H, W)
    return sample + noise


# C, H, W = 5, 64, 32
# sample = torch.zeros(C, H, W)

# x = torch.linspace(0, 1, W)
# y = torch.linspace(0, 1, H)
# xx, yy = torch.meshgrid(x, y, indexing='xy')

# print(xx.shape)
# print(yy.shape)
# print(xx.flatten().shape)
# coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
# print(coords.shape)

# noisy_sample = matern_kernel_noise(sample, coords)
# print(noisy_sample.shape)
