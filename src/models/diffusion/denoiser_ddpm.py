from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import Batch
from .inner_model_ddpm import InnerModel_ddpm, InnerModelConfig
from utils import ComputeLossOutput
import numpy as np
import math
import argparse
import tqdm

# parser = argparse.ArgumentParser(description='test for diffusion model')

# parser.add_argument('--batchsize',type=int,default=256,help='batch size per device for training Unet model')
# parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
# parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
# parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
# parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
# parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
# parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
# parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
# parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
# parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
# parser.add_argument('--droprate',type=float,default=0.1,help='dropout rate for model')
# parser.add_argument('--dtype',default=torch.float32)
# parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
# parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
# parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
# parser.add_argument('--epoch',type=int,default=1500,help='epochs for training')
# parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
# parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
# parser.add_argument('--interval',type=int,default=20,help='epoch interval between two evaluations')
# parser.add_argument('--moddir',type=str,default='model',help='model addresses')
# parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
# parser.add_argument('--genbatch',type=int,default=80,help='batch size for sampling process')
# parser.add_argument('--clsnum',type=int,default=10,help='num of label classes')
# parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
# parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
# parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
# parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
# parser.add_argument('--local_rank',default=0,type=int,help='node rank for distributed training')


# args = parser.parse_args()

def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float

def get_named_beta_schedule(schedule_name='linear', num_diffusion_timesteps=1000) -> np.ndarray:
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return  betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
        
def betas_for_alpha_bar(num_diffusion_timesteps:int, alpha_bar, max_beta=0.999) -> np.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class Denoiser_ddpm(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()

        self.dtype = torch.float32
        self.T = 500
        self.cfg = cfg
        self.inner_model = InnerModel_ddpm(cfg.inner_model)
        device = self.inner_model.noise_emb.weight.device
        betas = get_named_beta_schedule(num_diffusion_timesteps=self.T)
        self.betas = torch.tensor(betas,dtype=self.dtype)
        self.w = 1.8
        self.v = 0.3
        self.T = len(betas)
        self.alphas = 1 - self.betas
        self.log_alphas = torch.log(self.alphas)
        
        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim = 0)
        self.alphas_bar = torch.exp(self.log_alphas_bar)
        # self.alphas_bar = torch.cumprod(self.alphas, dim = 0)
        
        self.log_alphas_bar_prev = F.pad(self.log_alphas_bar[:-1],[1,0],'constant', 0)
        self.alphas_bar_prev = torch.exp(self.log_alphas_bar_prev)
        self.log_one_minus_alphas_bar_prev = torch.log(1.0 - self.alphas_bar_prev)
        # self.alphas_bar_prev = F.pad(self.alphas_bar[:-1],[1,0],'constant',1)

        # calculate parameters for q(x_t|x_{t-1})
        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas)
        # self.sqrt_alphas = torch.sqrt(self.alphas)

        # calculate parameters for q(x_t|x_0)
        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar
        self.sqrt_alphas_bar = torch.exp(self.log_sqrt_alphas_bar)
        # self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.exp(0.5 * self.log_one_minus_alphas_bar)
        
        # calculate parameters for q(x_{t-1}|x_t,x_0)
        # log calculation clipped because the \tilde{\beta} = 0 at the beginning
        self.tilde_betas = self.betas * torch.exp(self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.log_tilde_betas_clipped = torch.log(torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0))
        self.mu_coef_x0 = self.betas * torch.exp(0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.mu_coef_xt = torch.exp(0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.vars = torch.cat((self.tilde_betas[1:2],self.betas[1:]), 0)
        self.coef1 = torch.exp(-self.log_sqrt_alphas)
        self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar
        # calculate parameters for predicted x_0
        self.sqrt_recip_alphas_bar = torch.exp(-self.log_sqrt_alphas_bar)
        # self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.exp(self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar)
        # self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)

        self.uncond_pos = 0.1

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        None

    @staticmethod
    def _extract(coef:torch.Tensor, t:torch.Tensor, x_shape:tuple) -> torch.Tensor:
        """
        input:

        coef : an array
        t : timestep
        x_shape : the shape of tensor x that has K dims(the value of first dim is batch size)

        output:

        a tensor of shape [batchsize,1,...] where the length has K dims.
        """
        assert t.shape[0] == x_shape[0]

        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        coef = coef.to(t.device)
        chosen = coef[t]
        chosen = chosen.to(t.device)
        return chosen.reshape(neo_shape)
    def q_mean_variance(self, x_0:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of q(x_t|x_0)
        """
        mean = self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        var = self._extract(1.0 - self.sqrt_alphas_bar, t, x_0.shape)
        return mean, var
    
    def q_sample(self, x_0:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sample from q(x_t|x_0)
        """
        eps = torch.randn_like(x_0, requires_grad=False)
        return self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 \
            + self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps, eps
    
    def q_posterior_mean_variance(self, x_0:torch.Tensor, x_t:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of q(x_{t-1}|x_t,x_0)
        """
        posterior_mean = self._extract(self.mu_coef_x0, t, x_0.shape) * x_0 \
            + self._extract(self.mu_coef_xt, t, x_t.shape) * x_t
        posterior_var_max = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min = self._extract(self.log_tilde_betas_clipped, t, x_t.shape)
        log_posterior_var_max = self._extract(torch.log(self.betas), t, x_t.shape)
        log_posterior_var = self.v * log_posterior_var_max + (1 - self.v) * log_posterior_var_min
        neo_posterior_var = torch.exp(log_posterior_var)
        
        return posterior_mean, posterior_var_max, neo_posterior_var
    def p_mean_variance(self, x_t: Tensor, obs: Tensor, act: Tensor, t: Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        B, C = obs.shape[:2]
        assert t.shape == (B,)
        pred_eps = self.inner_model(x_t, obs, act, t)
        # if np.random.rand()< self.uncond_pos:
        #     zero_obs = torch.zeros(obs.shape, device= self.device)
        #     zero_act = torch.zeros(act.shape, device= self.device)
        #     pred_eps = self.model(x_t, zero_obs, zero_act, t)
        #pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        
        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"
        p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_eps)
        p_var = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
        return p_mean, p_var

    def _predict_x0_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.sqrt_recip_alphas_bar, t = t, x_shape = x_t.shape) \
            * x_t - self._extract(coef = self.sqrt_one_minus_alphas_bar, t = t, x_shape = x_t.shape) * eps

    def _predict_xt_prev_mean_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.coef1, t = t, x_shape = x_t.shape) * x_t - \
            self._extract(coef = self.coef2, t = t, x_shape = x_t.shape) * eps

    def p_sample(self, x_t: Tensor, obs: Tensor, act: Tensor, t: Tensor) -> torch.Tensor:
        """
        sample x_{t-1} from p_{theta}(x_{t-1}|x_t)
        """
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.p_mean_variance(x_t, obs, act, t)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise
    
    def sample(self, shape:tuple, obs: Tensor, act: Tensor) -> torch.Tensor:
        """
        sample images from p_{theta}
        """
        local_rank = 0
        if local_rank == 0:
            print('Start generating...')
        x_t = torch.randn(shape, device = self.device)
        tlist = torch.ones([x_t.shape[0]], device = self.device) * self.T
        for _ in tqdm(range(self.T),dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            tlist -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, obs, act, tlist)
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process...')
        return x_t
    
    def ddim_p_mean_variance(self, x_t:torch.Tensor, prevt:torch.Tensor, eta:float, obs: Tensor, act: Tensor, t: Tensor) -> torch.Tensor:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        pred_eps = self.inner_model(x_t, obs, act, t)
        # if np.random.rand()< self.uncond_pos:
        #     zero_obs = torch.zeros(obs.shape, device= self.device)
        #     zero_act = torch.zeros(act.shape, device= self.device)
        #     pred_eps = self.model(x_t, zero_obs, zero_act, t)
        # model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        # pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"

        alphas_bar_t = self._extract(coef = self.alphas_bar, t = t, x_shape = x_t.shape)
        alphas_bar_prev = self._extract(coef = self.alphas_bar_prev, t = prevt + 1, x_shape = x_t.shape)
        sigma = eta * torch.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar_t) * (1 - alphas_bar_t / alphas_bar_prev))
        p_var = sigma ** 2
        coef_eps = 1 - alphas_bar_prev - p_var
        coef_eps[coef_eps < 0] = 0
        coef_eps = torch.sqrt(coef_eps)
        p_mean = torch.sqrt(alphas_bar_prev) * (x_t - torch.sqrt(1 - alphas_bar_t) * pred_eps) / torch.sqrt(alphas_bar_t) + \
            coef_eps * pred_eps
        return p_mean, p_var
    
    def ddim_p_sample(self, x_t:torch.Tensor, prevt:torch.Tensor, eta:float, obs: Tensor, act: Tensor, t: Tensor) -> torch.Tensor: 
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.ddim_p_mean_variance(x_t, prevt.type(dtype=torch.long), eta, obs, act, t.type(dtype=torch.long))
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise
    
    def ddim_sample(self, shape:tuple, num_steps:int, eta:float, select:str, obs: Tensor, act: Tensor) -> torch.Tensor:
        local_rank = 0
        if local_rank == 0:
            print('Start generating(ddim)...')
        # a subsequence of range(0,1000)
        if select == 'linear':
            tseq = list(np.linspace(0, self.T-1, num_steps).astype(int))
        elif select == 'quadratic':
            tseq = list((np.linspace(0, np.sqrt(self.T), num_steps-1)**2).astype(int))
            tseq.insert(0, 0)
            tseq[-1] = self.T - 1
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{select}"')
        
        x_t = torch.randn(shape, device = self.device)
        tlist = torch.zeros([x_t.shape[0]], device = self.device)
        for i in range(num_steps):
            with torch.no_grad():
                tlist = tlist * 0 + tseq[-1-i]
                if i != num_steps - 1:
                    prevt = torch.ones_like(tlist, device = self.device) * tseq[-2-i]
                else:
                    prevt = - torch.ones_like(tlist, device = self.device) 
                x_t = self.ddim_p_sample(x_t, prevt, eta, obs, act, tlist)
                torch.cuda.empty_cache()
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process(ddim)...')
        return x_t
    
    # def forward(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tuple[Tensor, Tensor]:
    #     c_in, c_out, c_skip, c_noise = self._compute_conditioners(sigma)
    #     rescaled_obs = obs / self.cfg.sigma_data
    #     rescaled_noise = noisy_next_obs * c_in
    #     model_output = self.inner_model(rescaled_noise, c_noise, rescaled_obs, act)
    #     denoised = model_output * c_out + noisy_next_obs * c_skip
    #     return model_output, denoised

    def forward(self, noisy_next_obs: Tensor, obs: Tensor, act: Tensor, time: Tensor) -> Tuple[Tensor, Tensor]:
        if np.random.rand()< self.uncond_pos:
            print("drop condition")
            zero_obs = torch.zeros(obs.shape, device= self.device)
            zero_act = torch.zeros(act.shape, dtype=int, device= self.device)
            predicted_eps = self.inner_model(noisy_next_obs, zero_obs, zero_act, time)
        else:
            predicted_eps = self.inner_model(noisy_next_obs, obs, act, time)
        predicted_x0 = self._predict_x0_from_eps(noisy_next_obs, time, predicted_eps)
        return predicted_eps, predicted_x0


    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        _, d = self(noisy_next_obs, obs, act)
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d


    def compute_loss(self, batch: Batch) -> ComputeLossOutput:
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = batch.obs.size(1) - n

        all_obs = batch.obs.clone()
        loss = 0


        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            next_obs = all_obs[:, n + i]
            act = batch.act[:, i : n + i]
            mask = batch.mask_padding[:, n + i]
            # print(mask)

            b, t, c, h, w = obs.shape
            obs = obs.reshape(b, t * c, h, w)

            t = torch.randint(self.T, size = (obs.shape[0],), device=self.device)
            x_t, eps = self.q_sample(next_obs, t)
            pred_eps, predicted_x0 = self(x_t, obs, act, t)

            # model_output, denoised = self(noisy_next_obs, sigma, obs, act)

            # target = (next_obs - c_skip * noisy_next_obs) / c_out
            loss += F.mse_loss(pred_eps, eps)
            #loss += F.mse_loss(model_output, target)

            all_obs[:, n + i] = predicted_x0.detach().clamp(-1, 1)

        loss /= seq_length
        return loss, {"loss_denoising": loss.detach()}
