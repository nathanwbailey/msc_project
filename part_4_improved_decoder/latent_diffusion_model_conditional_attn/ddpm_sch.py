import matplotlib.pyplot as plt
import torch


def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    beta_t = torch.tensor(
        [beta1 + ((beta2 - beta1) * t / T) for t in range(0, T + 1)]
    )
    alpha_t = 1 - beta_t
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrt_beta_t = torch.sqrt(beta_t)
    alphabar_t = torch.cumprod(alpha_t, dim=0)
    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / torch.sqrt(1 - alphabar_t)

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


def plot_ddpm_schedules(beta1, beta2, T):

    schedules = ddpm_schedules(beta1, beta2, T)

    explanations = {
        "alpha_t": "Complement of noise level (1-β)\nControls signal preservation",
        "oneover_sqrta": "Scaling factor 1/√α\nUsed in denoising prediction",
        "sqrt_beta_t": "√β: Standard deviation\nof noise added per step",
        "alphabar_t": "ᾱ: Cumulative product of (1-β)\nTotal signal remaining",
        "sqrtab": "√ᾱ: Coefficient for x₀\nScales original image",
        "sqrtmab": "√(1-ᾱ): Coefficient for ε\nScales noise component",
        "mab_over_sqrtmab": "(1-α)/√(1-ᾱ)\nPosterior variance scaling",
    }

    plt.figure(figsize=(15, 10))

    for idx, (name, values) in enumerate(schedules.items(), 1):
        plt.subplot(3, 3, idx)
        plt.plot(values.numpy())
        plt.title(f"{name}\n{explanations[name]}", fontsize=10)
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("ddpm_sch.png")
