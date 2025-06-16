import torch
from torch import nn

from .ddpm_sch import ddpm_schedules


class DDPM(nn.Module):
    def __init__(
        self, eps_model, betas, n_T, loss_fn, device, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eps_model = eps_model.to(device)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            v = v.to(device)
            self.register_buffer(k, v)
        self.n_T = n_T
        self.loss_fn = loss_fn()
        self.device = device

    def forward(self, x, condition, t=None):
        if t is None:
            t = torch.randint(
                low=0, high=self.n_T + 1, size=(x.shape[0],)
            ).to(self.device)

        random_noise = torch.randn_like(x).to(self.device)

        sqrtab = self.sqrtab[t].view(-1, 1)
        sqrtmab = self.sqrtmab[t].view(-1, 1)
        x_t = sqrtab * x + sqrtmab * random_noise

        if isinstance(t, torch.Tensor):
            t = t.float()

        pred_noise = self.eps_model(x_t, t / self.n_T, condition)
        loss = self.loss_fn(pred_noise, random_noise)
        return (loss, random_noise, x_t)

    def sample(self, n_sample, size, condition, t=0):
        x_i = torch.randn(n_sample, size).to(self.device)
        for i in range(self.n_T, t, -1):
            time_tensor = (
                torch.full((n_sample, 1), i).float() / (self.n_T)
            ).to(self.device)
            if i > 1:
                rand_noise = torch.randn_like(x_i)
            else:
                rand_noise = torch.zeros_like(x_i)
            rand_noise = rand_noise.to(self.device)
            x_i = (
                self.oneover_sqrta[i]
                * (
                    x_i
                    - self.mab_over_sqrtmab[i]
                    * self.eps_model(x_i, time_tensor, condition)
                )
                + self.sqrt_beta_t[i] * rand_noise
            )

        return x_i
    
    def sample_and_log(self, n_sample, size, condition, t=0):
        trajectory_log = []
        x_i = torch.randn(n_sample, size).to(self.device)
        trajectory_log.append(x_i.clone().detach())
        for i in range(self.n_T, t, -1):
            time_tensor = (
                torch.full((n_sample, 1), i).float() / (self.n_T)
            ).to(self.device)
            if i > 1:
                rand_noise = torch.randn_like(x_i)
            else:
                rand_noise = torch.zeros_like(x_i)
            rand_noise = rand_noise.to(self.device)
            x_i = (
                self.oneover_sqrta[i]
                * (
                    x_i
                    - self.mab_over_sqrtmab[i]
                    * self.eps_model(x_i, time_tensor, condition)
                )
                + self.sqrt_beta_t[i] * rand_noise
            )
            trajectory_log.append(x_i.clone().detach())
        trajectory_log.reverse()
        trajectory_log = torch.stack(trajectory_log)
        return trajectory_log
