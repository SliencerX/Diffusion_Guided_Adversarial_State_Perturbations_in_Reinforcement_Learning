from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from .denoiser_ddpm import Denoiser_ddpm

import numpy as np
import argparse


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1


class DiffusionSampler_ddpm:
    def __init__(self, denoiser: Denoiser_ddpm, cfg: DiffusionSamplerConfig):
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)
        # parser = argparse.ArgumentParser(description='test for diffusion model')

        # parser.add_argument('--genbatch',type=int,default=5600,help='batch size for sampling process')
        # parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
        # parser.add_argument('--dtype',default=torch.float32)
        # parser.add_argument('--w',type=float,default=3.0,help='hyperparameters for classifier-free guidance strength')
        # parser.add_argument('--v',type=float,default=1.0,help='hyperparameters for the variance of posterior distribution')
        # parser.add_argument('--epoch',type=int,default=1000,help='epochs for loading models')
        # parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
        # # parser.add_argument('--device',default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),help='devices for training Unet model')
        # parser.add_argument('--label',type=str,default='range',help='labels of generated images')
        # parser.add_argument('--moddir',type=str,default='model_backup',help='model addresses')
        # parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
        # parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
        # parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
        # parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
        # parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
        # parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
        # parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
        # parser.add_argument('--droprate',type=float,default=0,help='dropout rate for model')
        # parser.add_argument('--clsnum',type=int,default=10,help='num of label classes')
        # parser.add_argument('--fid',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='generate samples used for quantative evaluation')
        # parser.add_argument('--genum',type=int,default=5600,help='num of generated samples')
        # parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
        # parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
        # parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
        # parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
        # parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
        # args = parser.parse_args()
        # self.args = args
        self.sample_steps = 20
        self.eta = 0
        self.select = 'linear'


    @torch.no_grad()
    def sample_next_obs(self, obs: Tensor, act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        device = obs.device
        b, t, c, h, w = obs.size()
        obs = obs.reshape(b, t * c, h, w)
        x = torch.randn(b, c, h, w, device=device)
        trajectory = [x]
        if self.sample_steps < self.denoiser.T:
            x = self.denoiser.ddim_sample((1,1,84,84), self.sample_steps, self.eta, self.select, obs, act)
        else:
            x = self.denoiser.sample((1,1,84,84), obs, act)

        return x, None
    
    # def sample_next_obs_classifier(self, obs: Tensor, act: Tensor, target_act: Tensor, policy) -> Tuple[Tensor, List[Tensor]]:
    #     max_guidance = 5
    #     add_factor = 0.3
    #     device = obs.device
    #     b, t, c, h, w = obs.size()
    #     x = torch.randn(b, c, h, w, device=device)
    #     #x = obs[0,-1].unsqueeze(0)
    #     obs = obs.reshape(b, t * c, h, w)
    #     s_in = torch.ones(b, device=device)
    #     gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        
    #     trajectory = [x]
    #     softmax = torch.nn.Softmax(dim = -1)
    #     #print(target_act)
    #     for time, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
    #         gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
    #         sigma_hat = sigma * (gamma + 1)
    #         with torch.enable_grad():
    #             # x_norm = (((x**2).mean((1,2,3)))**0.5)
    #             # print(x.shape)
    #             # print(x_norm)
    #             # x_in = x*0
    #             # for index in range(x_norm.shape[0]):
    #             #     if x_norm[index]>1:
    #             #         x_in[index] = x[index]/x_norm[index]
    #             #     else:
    #             #         x_in[index] = x[index]
    #             x_in = x.detach().requires_grad_(True).float()
    #             x_in = torch.permute(x_in, (0,1,3,2))
    #             # x_in = x_in.detach().requires_grad_(True).float()
    #             logits = policy.forward(x_in)
    #             logits = softmax(logits)
    #             # print(logits)
    #             # numerator = torch.exp(logits[0])[target_act]
    #             # #print(numerator)
    #             # denominator = torch.exp(logits[0]).sum(0, keepdim= True)
    #             #print(denominator)
    #             numerator = torch.exp(logits[0]*1)[target_act]
    #             denominator = torch.exp(logits[0]*0).sum(0, keepdim = True)
    #             # print(numerator)
    #             # print(denominator)
    #             selected = torch.log(numerator/denominator)

    #             current_time = time
    #             #current_guidance = (max_guidance/len(self.sigmas)) * (len(self.sigmas) - current_time)
    #             current_guidance = max_guidance
    #             current_guidance = max(current_guidance, 0.00001)

    #             interval = len(self.sigmas) - 1
    #             add_value = np.sin( current_time/interval * (1*np.pi) ) * max_guidance * add_factor
    #             current_guidance = current_guidance + add_value

    #             grads = torch.autograd.grad(selected.sum(), x_in)[0]
    #             grads = torch.permute(grads, (0,1,3,2))
    #             #grads = torch.clamp(grads, -1, 1)
    #             #grads_norm = ( ((grads**2).mean((1,2,3)))**0.5 )
    #             grads_norm = torch.norm(grads).unsqueeze(0)
    #             #print(grads_norm)
    #             #print(current_guidance)
    #             for index in range(x.shape[0]):
    #                 grads[index] = (grads[index]/grads_norm[index]) * current_guidance
    #             # print(gamma)
    #             if gamma > 0:
    #                 eps = torch.randn_like(x) * self.cfg.s_noise
    #                 x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
    #             x = x + grads
    #             denoised = self.denoiser.denoise(x, sigma, obs, act)
    #             d = (x - denoised) / sigma_hat
    #             dt = next_sigma - sigma_hat
    #             if self.cfg.order == 1 or next_sigma == 0:
    #                 # Euler method
    #                 x = x + d * dt
    #             else:
    #                 # Heun's method
    #                 x_2 = x + d * dt
    #                 denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
    #                 d_2 = (x_2 - denoised_2) / next_sigma
    #                 d_prime = (d + d_2) / 2
    #                 x = x + d_prime * dt
    #             # trajectory.append(x)
    #             # check_logit = torch.permute(x.detach(), (0,1,3,2))
    #             # logits = policy.forward(check_logit)
    #             # logits = softmax(logits)
    #             #print(logits)
    #     return x, trajectory
    
    # def sample_next_obs_partial(self,x_in, obs: Tensor, act: Tensor, sigmas_1, sigmas_2) -> Tuple[Tensor, List[Tensor]]:
    #     device = obs.device
    #     b, t, c, h, w = obs.size()
    #     obs = obs.reshape(b, t * c, h, w)
    #     s_in = torch.ones(b, device=device)
    #     gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
    #     x = x_in
    #     trajectory = [x]
    #     for sigma, next_sigma in zip(sigmas_1[:-1], sigmas_2[1:]):
    #         gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
    #         sigma_hat = sigma * (gamma + 1)
    #         if gamma > 0:
    #             eps = torch.randn_like(x) * self.cfg.s_noise
    #             x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
    #         denoised = self.denoiser.denoise(x, sigma, obs, act)
    #         d = (x - denoised) / sigma_hat
    #         dt = next_sigma - sigma_hat
    #         if self.cfg.order == 1 or next_sigma == 0:
    #             # Euler method
    #             x = x + d * dt
    #         else:
    #             # Heun's method
    #             x_2 = x + d * dt
    #             denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
    #             d_2 = (x_2 - denoised_2) / next_sigma
    #             d_prime = (d + d_2) / 2
    #             x = x + d_prime * dt
    #         trajectory.append(x)
    #     return x, trajectory
    
    # def sample_next_obs_classifier_guide(self, obs: Tensor, act: Tensor, target_act: Tensor, policy) -> Tuple[Tensor, List[Tensor]]:
    #     max_guidance = 8
    #     add_factor = 0.0
    #     device = obs.device
    #     ori_ob = torch.clone(obs).to(device)
    #     b, t, c, h, w = obs.size()
    #     x = torch.randn(b, c, h, w, device=device)
    #     obs = obs.reshape(b, t * c, h, w)
    #     s_in = torch.ones(b, device=device)
    #     gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        
    #     trajectory = [x]
    #     softmax = torch.nn.Softmax(dim = -1)
    #     for time, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
    #         gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
    #         sigma_hat = sigma * (gamma + 1)
    #         with torch.enable_grad():
    #             x_in = x.detach().requires_grad_(True).float()
    #             #x_in = torch.permute(x_in, (0,1,3,2))
    #             logits = policy.forward(torch.permute(x_in.add(1).div(2), (0,1,3,2)))
    #             logits = softmax(logits) #f(x)
    #             #logits_y = logits+torch.normal(0, 0.001, size = logits.shape).to(device) #y
    #             proposed_out,_ = self.sample_next_obs_partial(x_in, ori_ob, act, self.sigmas[time:-1], self.sigmas[1+time:])
    #             g_t = torch.autograd.grad(logits[0][target_act], x_in, retain_graph= True)[0]
    #             g_t = torch.permute(g_t, (0,1,3,2))
    #             g_t = torch.reshape(g_t, (1,-1))
    #             y = torch.matmul(g_t, torch.reshape(x_in, (-1,1))) + 0.1
    #             # proposed_out = proposed_out
    #             #proposed_out = torch.permute(proposed_out, (0,1,3,2))
    #             target_logits = policy.forward(torch.permute(proposed_out, (0,1,3,2)).add(1).div(2))
    #             target_logits = softmax(target_logits)
    #             square_error = torch.square(y - torch.matmul(g_t, torch.reshape(proposed_out, (-1,1))))
    #             # numerator = torch.exp(target_logits[0]*1)[target_act]
    #             # denominator = torch.exp(target_logits[0]*0).sum(0, keepdim = True)
    #             # # print(numerator)
    #             # # print(denominator)
    #             # selected = torch.log(numerator/denominator)
    #             grads = torch.autograd.grad(square_error, x_in)[0]
    #             #grads = torch.autograd.grad(selected, x_in)[0]

    #             # numerator = torch.exp(target_logits[0]*1)[target_act]
    #             # denominator = torch.exp(target_logits[0]*0).sum(0, keepdim = True)
    #             # # print(numerator)
    #             # # print(denominator)
    #             # selected = torch.log(numerator/denominator)

    #             current_time = time
    #             #current_guidance = (max_guidance/len(self.sigmas)) * (len(self.sigmas) - current_time)
    #             current_guidance = max_guidance
    #             current_guidance = max(current_guidance, 0.00001)
    #             interval = len(self.sigmas) - 1
    #             add_value = np.sin( current_time/interval * (1*np.pi) ) * max_guidance * add_factor
    #             current_guidance = current_guidance + add_value
    #             # grads = torch.autograd.grad(selected.sum(), x_in)[0]
    #             # grads = torch.permute(grads, (0,1,3,2))
    #             grads_norm = torch.norm(grads).unsqueeze(0) + 0.001
    #             for index in range(x.shape[0]):
    #                 grads[index] = (grads[index]/grads_norm[index])* (current_guidance)
    #             # grads = grads * (current_guidance)
    #             # print(gamma)
    #             # print(grads)
    #             if gamma > 0:
    #                 eps = torch.randn_like(x) * self.cfg.s_noise
    #                 x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
    #             x = x + grads
    #             denoised = self.denoiser.denoise(x, sigma, obs, act)
    #             d = (x - denoised) / sigma_hat
    #             dt = next_sigma - sigma_hat
    #             if self.cfg.order == 1 or next_sigma == 0:
    #                 # Euler method
    #                 x = x + d * dt
    #             else:
    #                 # Heun's method
    #                 x_2 = x + d * dt
    #                 denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
    #                 d_2 = (x_2 - denoised_2) / next_sigma
    #                 d_prime = (d + d_2) / 2
    #                 x = x + d_prime * dt
    #             # trajectory.append(x)
    #             # check_logit = torch.permute(x.detach(), (0,1,3,2))
    #             # logits = policy.forward(check_logit)
    #             # logits = softmax(logits)
    #             #print(logits)
    #     return x, trajectory


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))

