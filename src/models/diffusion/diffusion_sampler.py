from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from .denoiser import Denoiser

import numpy as np

from stable_baselines3 import DQN,PPO

def norm_neg_pos(x):
    return torch.tensor(x, device=x.device).mul(2).sub(1).contiguous()

def norm_zero_pos(x):
    return torch.tensor(x, device=x.device).add(1).div(2).contiguous()



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

def strength_scheduler(steps: int, current_step: int, true_obs = False):
    if true_obs:
        if current_step > steps:
            return 0 , 1
        return max(1-current_step/steps, 0.3), min(current_step/steps, 0.7)
    else:
        return 1, 0

class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig):
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)

    @torch.no_grad()
    def sample_next_obs(self, obs: Tensor, act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        device = obs.device
        b, t, c, h, w = obs.size()
        obs = obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        x = torch.randn(b, c, h, w, device=device)
        trajectory = [x]
        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
            denoised = self.denoiser.denoise(x, sigma, obs, act)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            trajectory.append(x)
        return x, trajectory
    
    def sample_next_obs_classifier(self, obs: Tensor, act: Tensor, target_act: Tensor, policy) -> Tuple[Tensor, List[Tensor]]:
        max_guidance = 5
        add_factor = 0.3
        device = obs.device
        b, t, c, h, w = obs.size()
        x = torch.randn(b, c, h, w, device=device)
        #x = obs[0,-1].unsqueeze(0)
        obs = obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        
        trajectory = [x]
        softmax = torch.nn.Softmax(dim = -1)
        #print(target_act)
        for time, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            with torch.enable_grad():
                # x_norm = (((x**2).mean((1,2,3)))**0.5)
                # print(x.shape)
                # print(x_norm)
                # x_in = x*0
                # for index in range(x_norm.shape[0]):
                #     if x_norm[index]>1:
                #         x_in[index] = x[index]/x_norm[index]
                #     else:
                #         x_in[index] = x[index]
                x_in = x.detach().requires_grad_(True).float()
                x_in = torch.permute(x_in, (0,1,3,2))
                # x_in = x_in.detach().requires_grad_(True).float()
                logits = policy.forward(x_in)
                logits = softmax(logits)
                # print(logits)
                # numerator = torch.exp(logits[0])[target_act]
                # #print(numerator)
                # denominator = torch.exp(logits[0]).sum(0, keepdim= True)
                #print(denominator)
                numerator = torch.exp(logits[0]*1)[target_act]
                denominator = torch.exp(logits[0]*0).sum(0, keepdim = True)
                # print(numerator)
                # print(denominator)
                selected = torch.log(numerator/denominator)

                current_time = time
                #current_guidance = (max_guidance/len(self.sigmas)) * (len(self.sigmas) - current_time)
                current_guidance = max_guidance
                current_guidance = max(current_guidance, 0.00001)

                interval = len(self.sigmas) - 1
                add_value = np.sin( current_time/interval * (1*np.pi) ) * max_guidance * add_factor
                current_guidance = current_guidance + add_value

                grads = torch.autograd.grad(selected.sum(), x_in)[0]
                grads = torch.permute(grads, (0,1,3,2))
                #grads = torch.clamp(grads, -1, 1)
                #grads_norm = ( ((grads**2).mean((1,2,3)))**0.5 )
                grads_norm = torch.norm(grads).unsqueeze(0)
                #print(grads_norm)
                #print(current_guidance)
                for index in range(x.shape[0]):
                    grads[index] = (grads[index]/grads_norm[index]) * current_guidance
                # print(gamma)
                if gamma > 0:
                    eps = torch.randn_like(x) * self.cfg.s_noise
                    x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
                x = x + grads
                denoised = self.denoiser.denoise(x, sigma, obs, act)
                d = (x - denoised) / sigma_hat
                dt = next_sigma - sigma_hat
                if self.cfg.order == 1 or next_sigma == 0:
                    # Euler method
                    x = x + d * dt
                else:
                    # Heun's method
                    x_2 = x + d * dt
                    denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
                    d_2 = (x_2 - denoised_2) / next_sigma
                    d_prime = (d + d_2) / 2
                    x = x + d_prime * dt
                # trajectory.append(x)
                # check_logit = torch.permute(x.detach(), (0,1,3,2))
                # logits = policy.forward(check_logit)
                # logits = softmax(logits)
                #print(logits)
        return x, trajectory
    
    def sample_next_obs_classifier_fade(self, obs: Tensor, act: Tensor, target_act: Tensor, policy) -> Tuple[Tensor, List[Tensor]]:
        max_guidance = 6
        add_factor = 0.01
        device = obs.device
        b, t, c, h, w = obs.size()
        x = torch.randn(b, c, h, w, device=device)
        #x = obs[0,-1].unsqueeze(0)
        obs = obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        
        trajectory = [x]
        softmax = torch.nn.Softmax(dim = -1)
        obs_zeros = torch.zeros(size= obs.shape, device= obs.device)
        act_zeros = torch.zeros(size= act.shape, dtype=int, device= act.device)
        #print(target_act)
        total_steps = int(len(self.sigmas[:-1]))
        for time, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True).float()
                x_in = torch.permute(x_in, (0,1,3,2))
                # x_in = x_in.detach().requires_grad_(True).float()
                logits = policy.forward(x_in)
                logits = softmax(logits)

                numerator = torch.exp(logits[0]*1)[target_act]
                denominator = torch.exp(logits[0]*0).sum(0, keepdim = True)

                selected = torch.log(numerator/denominator)

                current_time = time

                current_guidance = max_guidance
                #current_guidance = (max_guidance/len(self.sigmas)) * (len(self.sigmas) - time)
                #current_guidance = max(current_guidance, 2.7)

                interval = len(self.sigmas) - 1
                add_value = np.sin( current_time/interval * (1*np.pi) ) * max_guidance * add_factor
                current_guidance = current_guidance + add_value

                grads = torch.autograd.grad(selected.sum(), x_in)[0]
                grads = torch.permute(grads, (0,1,3,2))

                grads_norm = torch.norm(grads).unsqueeze(0)

                for index in range(x.shape[0]):
                    grads[index] = (grads[index]/grads_norm[index]) * current_guidance

                if gamma > 0:
                    eps = torch.randn_like(x) * self.cfg.s_noise
                    x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
                x = x + grads

                denoised = self.denoiser.denoise(x, sigma, obs, act)
                denoised_zero = self.denoiser.denoise(x, sigma, obs_zeros, act_zeros)
                d = (x - denoised) / sigma_hat
                dt = next_sigma - sigma_hat
                d_zero = (x - denoised_zero)/ sigma_hat
                alpha_1, alpha_2 = strength_scheduler(total_steps, time)
                if self.cfg.order == 1 or next_sigma == 0:
                    # Euler method
                    x = x + alpha_1 * d * dt + alpha_2 * d_zero * dt
                else:
                    # Heun's method
                    x_2 = x + d * dt
                    denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
                    d_2 = (x_2 - denoised_2) / next_sigma
                    d_prime = (d + d_2) / 2
                    x = x + d_prime * dt


                
                # trajectory.append(x)
                # check_logit = torch.permute(x.detach(), (0,1,3,2))
                # logits = policy.forward(check_logit)
                # logits = softmax(logits)
                #print(logits)
        return x, trajectory
    
    def sample_next_obs_partial(self,x_in, obs: Tensor, act: Tensor, sigmas_1, sigmas_2) -> Tuple[Tensor, List[Tensor]]:
        device = obs.device
        b, t, c, h, w = obs.size()
        obs = obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        x = x_in
        trajectory = [x]
        for sigma, next_sigma in zip(sigmas_1[:-1], sigmas_2[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
            denoised = self.denoiser.denoise(x, sigma, obs, act)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            trajectory.append(x)
        return x, trajectory
    
    def sample_next_obs_classifier_guide(self, obs: Tensor, act: Tensor, target_act: Tensor, policy) -> Tuple[Tensor, List[Tensor]]:
        max_guidance = 15
        add_factor = 0.0
        device = obs.device
        ori_ob = torch.clone(obs).to(device)
        b, t, c, h, w = obs.size()
        x = torch.randn(b, c, h, w, device=device)
        obs = obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        
        trajectory = [x]
        softmax = torch.nn.Softmax(dim = -1)
        for time, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True).float()
                #x_in = torch.permute(x_in, (0,1,3,2))
                logits = policy.forward(torch.permute(x_in.add(1).div(2), (0,1,3,2)))
                logits = softmax(logits) #f(x)
                #logits_y = logits+torch.normal(0, 0.001, size = logits.shape).to(device) #y
                proposed_out,_ = self.sample_next_obs_partial(x_in, ori_ob, act, self.sigmas[time:-1], self.sigmas[1+time:])
                g_t = torch.autograd.grad(logits[0][target_act], x_in, retain_graph= True)[0]
                g_t = torch.permute(g_t, (0,1,3,2))
                g_t = torch.reshape(g_t, (1,-1))
                y = torch.matmul(g_t, torch.reshape(x_in, (-1,1))) + 0.1
                # proposed_out = proposed_out
                #proposed_out = torch.permute(proposed_out, (0,1,3,2))
                target_logits = policy.forward(torch.permute(proposed_out, (0,1,3,2)).add(1).div(2))
                target_logits = softmax(target_logits)
                square_error = torch.square(y - torch.matmul(g_t, torch.reshape(proposed_out, (-1,1))))
                # numerator = torch.exp(target_logits[0]*1)[target_act]
                # denominator = torch.exp(target_logits[0]*0).sum(0, keepdim = True)
                # # print(numerator)
                # # print(denominator)
                # selected = torch.log(numerator/denominator)
                grads = torch.autograd.grad(square_error, x_in)[0]
                #grads = torch.autograd.grad(selected, x_in)[0]

                # numerator = torch.exp(target_logits[0]*1)[target_act]
                # denominator = torch.exp(target_logits[0]*0).sum(0, keepdim = True)
                # # print(numerator)
                # # print(denominator)
                # selected = torch.log(numerator/denominator)

                current_time = time
                #current_guidance = (max_guidance/len(self.sigmas)) * (len(self.sigmas) - current_time)
                current_guidance = max_guidance
                current_guidance = max(current_guidance, 0.00001)
                interval = len(self.sigmas) - 1
                add_value = np.sin( current_time/interval * (1*np.pi) ) * max_guidance * add_factor
                current_guidance = current_guidance + add_value
                # grads = torch.autograd.grad(selected.sum(), x_in)[0]
                # grads = torch.permute(grads, (0,1,3,2))
                grads_norm = torch.norm(grads).unsqueeze(0) + 0.000001
                for index in range(x.shape[0]):
                    grads[index] = (grads[index]/grads_norm[index])* (current_guidance)
                # grads = grads * (current_guidance)
                # print(gamma)
                # print(grads)
                if gamma > 0:
                    eps = torch.randn_like(x) * self.cfg.s_noise
                    x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
                x = x + grads
                denoised = self.denoiser.denoise(x, sigma, obs, act)
                d = (x - denoised) / sigma_hat
                dt = next_sigma - sigma_hat
                if self.cfg.order == 1 or next_sigma == 0:
                    # Euler method
                    x = x + d * dt
                else:
                    # Heun's method
                    x_2 = x + d * dt
                    denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
                    d_2 = (x_2 - denoised_2) / next_sigma
                    d_prime = (d + d_2) / 2
                    x = x + d_prime * dt
                # trajectory.append(x)
                # check_logit = torch.permute(x.detach(), (0,1,3,2))
                # logits = policy.forward(check_logit)
                # logits = softmax(logits)
                #print(logits)
        return x, trajectory
    
    def sample_next_obs_classifier_guide_fade(self, obs: Tensor, act: Tensor, target_act: Tensor, policy, max_guidance, ae = None) -> Tuple[Tensor, List[Tensor]]:
        diff = torch.nn.MSELoss(reduce= 'sum')
        add_factor = 0.00
        device = obs.device
        ori_ob = torch.clone(obs).to(device)
        b, t, c, h, w = obs.size()
        x = torch.randn(b, c, h, w, device=device)
        obs = obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        total_steps = len(self.sigmas[:-1])
        trajectory = [x]
        softmax = torch.nn.Softmax(dim = -1)
        obs_zeros = torch.zeros(size= obs.shape, device= obs.device)
        act_zeros = torch.full(size= act.shape, fill_value = self.denoiser.cfg.inner_model.num_actions, dtype=int, device= act.device)
        for time, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True).float()
                # logits = policy.forward(torch.permute(x_in.add(1).div(2), (0,1,3,2)))
                # logits = softmax(logits) #f(x)
                proposed_out,_ = self.sample_next_obs_partial(x_in, ori_ob, act, self.sigmas[time:-1], self.sigmas[1+time:])
                # g_t = torch.autograd.grad(logits[0][target_act], x_in, retain_graph= True)[0]
                # g_t = torch.permute(g_t, (0,1,3,2))
                # g_t = torch.reshape(g_t, (1,-1))
                # y = torch.matmul(g_t, torch.reshape(x_in, (-1,1))) + 0.1
                if isinstance(policy, DQN):
                    target_logits = policy.q_net(proposed_out.add(1).div(2)*255)
                else:
                    target_logits = policy.forward(torch.permute(proposed_out, (0,1,3,2)).add(1).div(2))
                target_logits = softmax(target_logits)
                numerator = torch.exp(target_logits[0]*1)[target_act]
                denominator = torch.exp(target_logits[0]*0).sum(0, keepdim = True)
                selected = torch.log(numerator/denominator)
                # grads = torch.autograd.grad(selected, x_in)[0]
                # square_error = torch.square(y - torch.matmul(g_t, torch.reshape(proposed_out, (-1,1))))
                grads = torch.autograd.grad(selected, proposed_out)[0]
                current_time = time
                #current_guidance = (max_guidance/len(self.sigmas)) * (len(self.sigmas) - current_time)
                current_guidance = max_guidance
                current_guidance = max(current_guidance, 0.00001)
                interval = len(self.sigmas) - 1
                add_value = np.sin( current_time/interval * (1*np.pi) ) * max_guidance * add_factor
                current_guidance = current_guidance + add_value

                grads_norm = torch.norm(grads).unsqueeze(0) + 0.000001
                for index in range(x.shape[0]):
                    grads[index] = (grads[index]/grads_norm[index])* (current_guidance)

                if gamma > 0:
                    eps = torch.randn_like(x) * self.cfg.s_noise
                    x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
                if next_sigma!=0:
                    to_check = norm_zero_pos(x_in).requires_grad_()
                    loss = diff(to_check,ae(to_check))
                    grad_1 = torch.autograd.grad(loss, to_check)[0]
                    grad_1_norm = torch.norm(grad_1).unsqueeze(0) + 0.000001
                    for index in range(x.shape[0]):
                        grad_1[index] = (grad_1[index]/grad_1_norm[index])
                    #print(grad_1)
                    x = x - 1 * grad_1
                x = x + grads
                denoised = self.denoiser.denoise(x, sigma, obs, act)
                alpha_1, alpha_2 = strength_scheduler(total_steps, time) 
                denoised_zero = self.denoiser.denoise(x, sigma, obs_zeros, act_zeros)
                d = (x - denoised) / sigma_hat
                dt = next_sigma - sigma_hat
                d_zero = (x-denoised_zero)/ sigma_hat
                alpha_1, alpha_2 = strength_scheduler(total_steps, time)
                if isinstance(policy, DQN):
                    alpha_1 = 1
                    alpha_2 = 0.00
                if self.cfg.order == 1 or next_sigma == 0:
                    # Euler method
                    x = x + alpha_1 * d * dt + alpha_2 * d_zero * dt
                else:
                    # Heun's method
                    x_2 = x + d * dt
                    denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
                    d_2 = (x_2 - denoised_2) / next_sigma
                    d_prime = (d + d_2) / 2
                    x = x + d_prime * dt
                # if next_sigma!=0:
                #     to_check = norm_zero_pos(x).requires_grad_()
                #     loss = diff(to_check,ae(to_check))
                #     grad_1 = torch.autograd.grad(loss, to_check)[0]
                #     grad_1_norm = torch.norm(grad_1).unsqueeze(0) + 0.000001
                #     for index in range(x.shape[0]):
                #         grad_1[index] = (grad_1[index]/grad_1_norm[index])
                #     #print(grad_1)
                #     x = x - grad_1
                # trajectory.append(x)
                # check_logit = torch.permute(x.detach(), (0,1,3,2))
                # logits = policy.forward(check_logit)
                # logits = softmax(logits)
                #print(logits)
        return x, trajectory
    
    def sample_next_obs_classifier_guide_fade_ppo(self, obs: Tensor, act: Tensor, target_act: Tensor, policy, max_guidance, true_state, ae = None, true_obs = False) -> Tuple[Tensor, List[Tensor]]:
        diff = torch.nn.MSELoss(reduce= 'sum')
        add_factor = 0.00
        device = obs.device
        ori_ob = torch.clone(obs).to(device)
        b, t, c, h, w = obs.size()
        x = torch.randn(b, c, h, w, device=device)
        obs = obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        total_steps = len(self.sigmas[:-1])
        trajectory = [x]
        softmax = torch.nn.Softmax(dim = -1)
        obs_zeros = torch.zeros(size= obs.shape, device= obs.device)
        act_zeros = act
        for time, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            with torch.enable_grad():
                x_in = x.detach().float()

                proposed_out,_ = self.sample_next_obs_partial(x_in, ori_ob, act, self.sigmas[time:-1], self.sigmas[1+time:])
                proposed_out.requires_grad_(True)
                # target_act_t = torch.tensor(target_act).cuda()
                # _, log_prob,_ = policy.policy.evaluate_actions(proposed_out, target_act_t)

                proposed_action = policy.policy._predict(proposed_out)
                true_action = policy.policy._predict(true_state)
                # if isinstance(policy, DQN):
                #selected,p,_ = policy.policy.evaluate_actions(true_state, proposed_action)
                # elif isinstance(policy, PPO):
                _,p,_ = policy.policy.evaluate_actions(proposed_out, true_action)
                selected = p

                # selected = policy.policy.predict_values(proposed_out*255)

                # grads = torch.autograd.grad(selected, x_in)[0]
                # square_error = torch.square(y - torch.matmul(g_t, torch.reshape(proposed_out, (-1,1))))
                grads = torch.autograd.grad(selected, proposed_out)[0]
                current_time = time
                #current_guidance = (max_guidance/len(self.sigmas)) * (len(self.sigmas) - current_time)
                current_guidance = max_guidance
                current_guidance = max(current_guidance, 0.00001)
                interval = len(self.sigmas) - 1
                add_value = np.sin( current_time/interval * (1*np.pi) ) * max_guidance * add_factor
                current_guidance = current_guidance + add_value

                grads_norm = torch.norm(grads).unsqueeze(0) + 0.000001
                for index in range(x.shape[0]):
                    grads[index] = (grads[index]/grads_norm[index])* (current_guidance)

                if gamma > 0:
                    eps = torch.randn_like(x) * self.cfg.s_noise
                    x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
                if next_sigma!=0:
                    to_check = norm_zero_pos(x_in).requires_grad_()
                    loss = diff(to_check,ae(to_check))
                    grad_1 = torch.autograd.grad(loss, to_check)[0]
                    grad_1_norm = torch.norm(grad_1).unsqueeze(0) + 0.000001
                    for index in range(x.shape[0]):
                        grad_1[index] = (grad_1[index]/grad_1_norm[index])
                    #print(grad_1)
                    x = x - 0.1 * grad_1
                x = x - 0.95*grads
                denoised = self.denoiser.denoise(x, sigma, obs, act)
                alpha_1, alpha_2 = strength_scheduler(total_steps, time, true_obs= true_obs) 
                denoised_zero = self.denoiser.denoise(x, sigma, obs_zeros, act_zeros)
                d = (x - denoised) / sigma_hat
                dt = next_sigma - sigma_hat
                d_zero = (x-denoised_zero)/ sigma_hat
                alpha_1, alpha_2 = strength_scheduler(total_steps, time)
                if self.cfg.order == 1 or next_sigma == 0:
                    # Euler method
                    x = x + alpha_1 * d * dt + alpha_2 * d_zero * dt
                else:
                    # Heun's method
                    x_2 = x + d * dt
                    denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
                    d_2 = (x_2 - denoised_2) / next_sigma
                    d_prime = (d + d_2) / 2
                    x = x + d_prime * dt
                # if next_sigma!=0:
                #     to_check = norm_zero_pos(x).requires_grad_()
                #     loss = diff(to_check,ae(to_check))
                #     grad_1 = torch.autograd.grad(loss, to_check)[0]
                #     grad_1_norm = torch.norm(grad_1).unsqueeze(0) + 0.000001
                #     for index in range(x.shape[0]):
                #         grad_1[index] = (grad_1[index]/grad_1_norm[index])
                #     #print(grad_1)
                #     x = x - grad_1
                # trajectory.append(x)
                # check_logit = torch.permute(x.detach(), (0,1,3,2))
                # logits = policy.forward(check_logit)
                # logits = softmax(logits)
                #print(logits)
        return x, trajectory
    
    def sample_next_obs_classifier_guide_fade_v(self, obs: Tensor, act: Tensor, target_act: Tensor, policy, max_guidance, true_state, ae = None, true_obs = False) -> Tuple[Tensor, List[Tensor]]:
        diff = torch.nn.MSELoss(reduce= 'sum')
        add_factor = 0.00
        device = obs.device
        ori_ob = torch.clone(obs).to(device)
        b, t, c, h, w = obs.size()
        x = torch.randn(b, c, h, w, device=device)
        obs = obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        total_steps = len(self.sigmas[:-1])
        trajectory = [x]
        softmax = torch.nn.Softmax(dim = -1)
        obs_zeros = torch.zeros(size= obs.shape, device= obs.device)
        act_zeros = torch.full(size= act.shape, fill_value = self.denoiser.cfg.inner_model.num_actions, dtype=int, device= act.device)
        for time, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True).float()
                # logits = policy.forward(torch.permute(x_in.add(1).div(2), (0,1,3,2)))
                # logits = softmax(logits) #f(x)
                proposed_out,_ = self.sample_next_obs_partial(x_in, ori_ob, act, self.sigmas[time:-1], self.sigmas[1+time:])
                # g_t = torch.autograd.grad(logits[0][target_act], x_in, retain_graph= True)[0]
                # g_t = torch.permute(g_t, (0,1,3,2))
                # g_t = torch.reshape(g_t, (1,-1))
                # y = torch.matmul(g_t, torch.reshape(x_in, (-1,1))) + 0.1
                if isinstance(policy, DQN):
                    target_logits = policy.q_net(proposed_out.add(1).div(2)*255)
                else:
                    target_logits = policy.forward(torch.permute(proposed_out, (0,1,3,2)).add(1).div(2))
                tmp_act = softmax(target_logits[0])
                if isinstance(policy, DQN):
                    true_state_in = (torch.permute(true_state,(0,2,1))*255).unsqueeze(0)
                    selected = torch.dot(tmp_act, policy.q_net(true_state_in)[0])
                else:
                    selected = torch.dot(tmp_act,policy.forward(true_state)[0])
                #target_logits = softmax(target_logits)
                #selected = torch.max(target_logits[0])
                grads = torch.autograd.grad(selected, proposed_out)[0]
                current_time = time
                #current_guidance = (max_guidance/len(self.sigmas)) * (len(self.sigmas) - current_time)
                current_guidance = max_guidance
                current_guidance = max(current_guidance, 0.00001)
                interval = len(self.sigmas) - 1
                add_value = np.sin( current_time/interval * (1*np.pi) ) * max_guidance * add_factor
                current_guidance = current_guidance + add_value

                grads_norm = torch.norm(grads).unsqueeze(0) + 0.000001
                for index in range(x.shape[0]):
                    grads[index] = (grads[index]/grads_norm[index])* (current_guidance)

                if gamma > 0:
                    eps = torch.randn_like(x) * self.cfg.s_noise
                    x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
                if next_sigma!=0:
                    to_check = norm_zero_pos(x_in).requires_grad_()
                    loss = diff(to_check,ae(to_check))
                    grad_1 = torch.autograd.grad(loss, to_check)[0]
                    grad_1_norm = torch.norm(grad_1).unsqueeze(0) + 0.000001
                    for index in range(x.shape[0]):
                        grad_1[index] = (grad_1[index]/grad_1_norm[index])
                    #print(grad_1)
                    x = x - 1 * grad_1
                x = x - grads
                denoised = self.denoiser.denoise(x, sigma, obs, act)
                alpha_1, alpha_2 = strength_scheduler(total_steps, time) 
                denoised_zero = self.denoiser.denoise(x, sigma, obs_zeros, act_zeros)
                d = (x - denoised) / sigma_hat
                dt = next_sigma - sigma_hat
                d_zero = (x-denoised_zero)/ sigma_hat
                alpha_1, alpha_2 = strength_scheduler(total_steps, time, true_obs)
                # if isinstance(policy, DQN):
                #     alpha_1 = 0.9
                #     alpha_2 = 0.1
                if self.cfg.order == 1 or next_sigma == 0:
                    # Euler method
                    x = x + alpha_1 * d * dt + alpha_2 * d_zero * dt
                else:
                    # Heun's method
                    x_2 = x + d * dt
                    denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
                    d_2 = (x_2 - denoised_2) / next_sigma
                    d_prime = (d + d_2) / 2
                    x = x + d_prime * dt
                # if next_sigma!=0:
                #     to_check = norm_zero_pos(x).requires_grad_()
                #     loss = diff(to_check,ae(to_check))
                #     grad_1 = torch.autograd.grad(loss, to_check)[0]
                #     grad_1_norm = torch.norm(grad_1).unsqueeze(0) + 0.000001
                #     for index in range(x.shape[0]):
                #         grad_1[index] = (grad_1[index]/grad_1_norm[index])
                #     #print(grad_1)
                #     x = x - grad_1
                # trajectory.append(x)
                # check_logit = torch.permute(x.detach(), (0,1,3,2))
                # logits = policy.forward(check_logit)
                # logits = softmax(logits)
                #print(logits)
        return x, trajectory


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))

