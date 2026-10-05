import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def rms_norm(x, target_rms=1.0, eps=1e-8):
    # x: [B, N, F, 2] float
    cur_rms = torch.sqrt(torch.mean(x**2, dim=(1,2,3), keepdim=True) + eps)  # [B,1,1,1]
    return x * (target_rms / cur_rms)

class SignalDiffusion(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params

        self.latent_channels = 4
        self.N = params.latent_dims // self.latent_channels

        self.input_dim = self.N
        self.max_step = params.max_step

        beta = np.array(params.noise_schedule)

        self.alpha = torch.tensor((1 - beta).astype(np.float32))
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.var_blur = torch.tensor(
            np.array(params.blur_schedule).astype(np.float32)
        )

        self.var_blur_bar = torch.cumsum(
            self.var_blur,
            dim=0
        )

        self.var_kernel = (
            self.input_dim / self.var_blur
        ).unsqueeze(1)

        self.var_kernel_bar = (
            self.input_dim / self.var_blur_bar
        ).unsqueeze(1)

        self.gaussian_kernel = self.get_kernel(
            self.var_kernel
        )

        self.gaussian_kernel_bar = self.get_kernel(
            self.var_kernel_bar
        )

        self.info_weights = (
            self.gaussian_kernel_bar *
            torch.sqrt(self.alpha_bar).unsqueeze(-1)
        )

        self.noise_weights = self.get_noise_weights()
      
    def get_kernel(self, var_kernel):
        samples = torch.arange(0, self.input_dim) # [N]
        gaussian_kernel = torch.exp(-((samples - self.input_dim // 2)**2) / (2 * var_kernel)) / torch.sqrt(2 * torch.pi * var_kernel) # G_t, [T, N]
        gaussian_kernel = self.input_dim * gaussian_kernel / torch.sum(gaussian_kernel, dim=1, keepdim=True) # Normalized G_t, [T, N]
        return gaussian_kernel

    def get_noise_weights(self):
        noise_weights = []
        for t in range(self.max_step):
            upper_bound = t + 1
            one_minus_alpha_sqrt = torch.sqrt(1 - self.alpha[0:upper_bound]) # \sqrt(1-\bar{\alpha_s}), for s in [1, t], [t]
            rev_one_minus_alpha_sqrt = torch.flipud(one_minus_alpha_sqrt) # \sqrt(1-\bar{\alpha_s}), for s in [t, 1], [t]
            rev_alpha = torch.flipud(self.alpha[0:upper_bound]) # alpha_s, for s in [t, 1], [t]
            rev_alpha_bar_sqrt = torch.sqrt(torch.cumprod(rev_alpha, dim=0) / rev_alpha[-1]) # \sqrt{\bar{\alpha_t} / \bar{\alpha_s}}, for s in [t, 1], [t]
            rev_var_blur = torch.flipud(self.var_blur[:upper_bound]) # [t] 
            rev_var_blur_bar = torch.cumsum(rev_var_blur, dim=0) - rev_var_blur[-1] # [t]
            rev_var_kernel_bar = (self.input_dim / rev_var_blur_bar).unsqueeze(1) # [t, 1]
            rev_kernel_bar = self.get_kernel(rev_var_kernel_bar) # \bar{G_t} / \bar{G_s}, for s in [t, 1], [t, N]
            rev_kernel_bar[0, :] = torch.ones(self.input_dim) 
            noise_weights.append(torch.mv((rev_alpha_bar_sqrt.unsqueeze(-1) * rev_kernel_bar).transpose(0, 1), rev_one_minus_alpha_sqrt)) # [t, N]
        return torch.stack(noise_weights, dim=0) # [T, N] 

    def get_noise_weights_stats(self):
        noise_weights = []
        one_minus_alpha_sqrt = torch.sqrt(1 - self.alpha[0])
        for t in range(self.max_step):
            noise_weights.append((1 - torch.sqrt(self.alpha_bar[t])*self.gaussian_kernel_bar[t, :]) / (1 - torch.sqrt(self.alpha[0]) * self.gaussian_kernel[0, :]))
        return one_minus_alpha_sqrt * torch.stack(noise_weights, dim=0) # [T, N]

    ## Depracated: numerical instable when params.blur_schedule is high, kernel may divided by 0.
    def get_noise_weights_div(self):
        noise_weights = []
        for t in range(self.max_step):
            upper_bound = t + 1
            one_minus_alpha_sqrt = torch.sqrt(1 - self.alpha[:upper_bound]) # \sqrt(1-\bar{\alpha_s}), for s in [1, t], [t]
            ratio_alpha_bar_sqrt = torch.sqrt(self.alpha_bar[t] / self.alpha_bar[:upper_bound]) # \sqrt(\bar{\alpha_t} / \bar{\alpha_s}), for s in [1, t], [t]
            ratio_kernel_bar = self.gaussian_kernel_bar[t, :] / self.gaussian_kernel_bar[:upper_bound, :] # \bar{G_t} / \bar{G_s}, for s in [1, t], [t, N]
            noise_weights.append(torch.mv((ratio_alpha_bar_sqrt.unsqueeze(-1) * ratio_kernel_bar).transpose(0, 1), one_minus_alpha_sqrt)) # [t, N]
        return torch.stack(noise_weights, dim=0) # [T, N]
    
    ## Depracated: numerical instable when params.blur_schedule is high, amplitude of kernel may overflow.
    def get_noise_weights_prod(self):
        noise_weights = []
        for t in range(self.max_step):
            upper_bound = t + 1
            one_minus_alpha_sqrt = torch.sqrt(1 - self.alpha[0:upper_bound]) # \sqrt(1-\bar{\alpha_s}), for s in [1, t], [t]
            rev_one_minus_alpha_sqrt = torch.flipud(one_minus_alpha_sqrt) # \sqrt(1-\bar{\alpha_s}), for s in [t, 1], [t]
            rev_alpha = torch.flipud(self.alpha[0:upper_bound]) # alpha_s, for s in [t, 1], [t]
            rev_alpha_bar_sqrt = torch.sqrt(torch.cumprod(rev_alpha, dim=0) / rev_alpha[-1]) # \sqrt{\bar{\alpha_t} / \bar{\alpha_s}}, for s in [t, 1], [t]
            rev_kernel = torch.flipud(self.gaussian_kernel[:upper_bound, :]) # G_s, for s in [t, 1], [t, N]
            rev_kernel_bar = torch.cumprod(rev_kernel, dim=0) / rev_kernel[-1, :] # \bar{G_t} / \bar{G_s}, for s in [t, 1], [t, N]
            noise_weights.append(torch.mv((rev_alpha_bar_sqrt.unsqueeze(-1) * rev_kernel_bar).transpose(0, 1), rev_one_minus_alpha_sqrt)) # [t, N]
        return torch.stack(noise_weights, dim=0) # [T, N] 

    def degrade_fn(self, x_0, t):
        """
        x_0: [B, C, N] complex64
        return:
            x_t:   [B, C, N] complex64
            noise: [B, C, N] complex64
        """

        device = x_0.device

        if self.noise_weights.device != device:
            self.noise_weights = self.noise_weights.to(device)

        if self.info_weights.device != device:
            self.info_weights = self.info_weights.to(device)

        noise_weight = self.noise_weights[t, :].unsqueeze(1).to(device)  # [B,1,N]
        info_weight = self.info_weights[t, :].unsqueeze(1).to(device)    # [B,1,N]

        noise_real = torch.randn_like(x_0.real, dtype=torch.float32, device=device)
        noise_imag = torch.randn_like(x_0.real, dtype=torch.float32, device=device)

        noise = torch.complex(noise_real, noise_imag) / np.sqrt(2.0)

        noise = noise_weight * noise
        x_t = info_weight * x_0 + noise

        return x_t, noise


    def sampling(self, restore_fn, cond, device):

        if isinstance(cond, dict):
            cond_list = cond.get('prompt')

            if isinstance(cond_list, str):
                cond_list = [cond_list]
            elif isinstance(cond_list, (list, tuple)):
                cond_list = list(cond_list)
            else:
                raise TypeError("cond['prompt'] must be str or list[str].")

        else:
            if isinstance(cond, str):
                cond_list = [cond]
            elif isinstance(cond, (list, tuple)):
                cond_list = list(cond)
            else:
                raise TypeError("cond must be str, list[str], or dict.")

        batch_size = len(cond_list)

        N = self.N
        C = self.latent_channels

        noise_real = torch.randn(batch_size, C, N, device=device)
        noise_imag = torch.randn(batch_size, C, N, device=device)

        noise = torch.complex(noise_real, noise_imag) / np.sqrt(2.0)

        batch_max = (self.max_step - 1) * torch.ones(
            batch_size,
            dtype=torch.int64,
            device=device
        )

        info_w = self.info_weights[batch_max, :].to(device)      # [B,N]
        noise_w = self.noise_weights[batch_max, :].to(device)   # [B,N]

        inf_weight = (noise_w + info_w).unsqueeze(1)             # [B,1,N]

        x_s = inf_weight * noise                                 # [B,C,N]

        for s in range(self.max_step - 1, -1, -1):

            t = s * torch.ones(
                batch_size,
                dtype=torch.int64,
                device=device
            )

            call_cond = cond if isinstance(cond, dict) else cond_list

            x_0_hat = restore_fn(x_s, t, call_cond)

            if s > 0:
                t_prev = (s - 1) * torch.ones(
                    batch_size,
                    dtype=torch.int64,
                    device=device
                )

                x_s, _ = self.degrade_fn(x_0_hat, t_prev)

        return x_0_hat