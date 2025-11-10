import argparse
from pathlib import Path
from typing import Tuple

# from huggingface_hub import hf_hub_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from agent import Agent
from agent_ddpm import Agent_DDPM
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, collate_segments_to_batch, Dataset
from envs import make_atari_env, WorldModelEnv, make_atari_env_test, WorldModelEnv_DDPM
#from game import ActionNames, DatasetEnv, Game, get_keymap_and_action_names, Keymap, NamedEnv, PlayEnv
from utils import get_path_agent_ckpt, prompt_atari_game
from model import *
import cv2
from utils import *
import sys
sys.path.append(".")
from autoencoder_models import *
import argparse
import csv
from attacks import *
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import time
from gym.spaces import Box
from paad_rl.a2c_ppo_acktr import algo, utils
from paad_rl.a2c_ppo_acktr.algo import gail
from paad_rl.a2c_ppo_acktr.arguments import get_args
from paad_rl.a2c_ppo_acktr.envs import make_vec_envs
from paad_rl.a2c_ppo_acktr.model import Policy
from paad_rl.a2c_ppo_acktr.storage import RolloutStorage
# from evaluation import evaluate
from paad_rl.attacker.attacker import *
from paad_rl.utils.dqn_core import DQN_Agent, Q_Atari,model_get
from paad_rl.utils.param import Param
from paad_rl.a2c_ppo_acktr.algo.kfac import KFACOptimizer
import torchvision.transforms.functional as TF
from stable_baselines3 import PPO, DQN


def normalize_to_reference(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Normalize source image to match reference's channel-wise mean/std.
    Inputs: source/reference tensors of shape [C, H, W]
    """
    # Compute reference statistics
    ref_mean = reference.view(reference.shape[0], -1).mean(dim=1)
    ref_std = reference.view(reference.shape[0], -1).std(dim=1)

    # Compute source statistics
    src_mean = source.view(source.shape[0], -1).mean(dim=1)
    src_std = source.view(source.shape[0], -1).std(dim=1)

    # Avoid division by zero (add epsilon=1e-8 if needed)
    normalized = (source - src_mean[:, None, None]) * (ref_std[:, None, None] / src_std[:, None, None]) + ref_mean[:, None, None]
    
    return normalized

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def brightness(image, off1, off2):
    return np.clip(image*off1 + off2/255,-1,1)

def shift_img(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted_image

from scipy.stats import wasserstein_distance

def calculate_wasserstein_distance_tensor(image1_tensor, image2_tensor):
    # Convert PyTorch tensors to numpy arrays and flatten them
    image1_flat = image1_tensor.cpu().numpy().flatten()
    image2_flat = image2_tensor.cpu().numpy().flatten()
    
    # Calculate Wasserstein distance
    distance = wasserstein_distance(image1_flat, image2_flat)
    return distance

def calculate_l2_distance(image1_tensor, image2_tensor):
    # Ensure both images are of the same shape
    assert image1_tensor.shape == image2_tensor.shape, "Images must have the same shape"
    
    # Calculate the L2 (Euclidean) distance
    distance = torch.norm(image1_tensor - image2_tensor, p=2)
    return distance.item()  # Convert to Python float

OmegaConf.register_new_resolver("eval", eval)

def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
            
        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)


@torch.no_grad()
def main(args):
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    U_net = Unet(
        dim = 64,
        dim_mults = (1, 2),
        channels = 1
    )

    denoiser = GaussianDiffusion(
        U_net,
        image_size = 84,
        # channels = 1,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    )

    with initialize(version_base="1.3", config_path="../config"):
        if 'airsim' in args.env:
            cfg = compose(config_name="airsim_trainer")
        else:
            cfg = compose(config_name="trainer")
        OmegaConf.resolve(cfg)

    #test_env = make_atari_env(num_envs=, device=device, **cfg.env.train)
    cfg.env.test.id = args.env
    cfg.env.train.id = args.env
    test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=device, **cfg.env.test)
    print("obs_space", test_env.observation_space)

    #Freeway Model
    if "Freeway" in args.env:
        agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
        path_ckpt = get_path_agent_ckpt("/outputs/2025-05-01/17-00-25/checkpoints", epoch = -1)
        agent.load(path_ckpt)

    if "Pong" in args.env:
    # Pong Model
        agent = Agent(instantiate(cfg.agent, num_actions=4)).to(device).eval()
        path_ckpt = get_path_agent_ckpt("/outputs/2024-08-12/17-44-08/checkpoints", epoch = 50)
        agent.load(path_ckpt)

    n = 4
    dataset = Dataset(Path(f"dataset/{path_ckpt.stem}_{n}"))
    dataset.load_from_default_path()

    # World model environment
    bs = BatchSampler(dataset, 1, cfg.agent.denoiser.inner_model.num_steps_conditioning, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)

    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg, return_denoising_trajectory=True)


    if "Pong" in args.env and (args.model == "natural" or args.model == "diffusion_history"):
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong-natural.model'))
    if "Pong" in args.env and args.model == "sa-dqn-convex":
        policy = model_setup(cfg.env.train.id, test_env, True, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong-convex.model'))
    if "Pong" in args.env and args.model == "sa-dqn-pgd":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, False, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong-pgd.model'))
    if "Pong" in args.env and args.model == "wocar":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, False, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong-wocar-pgd.pth'))
    if "Pong" in args.env and args.model == "car-dqn-pgd":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong_car_pgd.pth'))

    if "Pong" in args.env and "dp-dqn" in args.model:
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong-DP-DQN-O.pth'))
        #policy.features.load_state_dict(torch.load('src/pre_trained/Pong-natural.model'))
        denoiser.load_state_dict(torch.load("/results_Pong/model-150.pt")['model'])
        denoiser.to(device)

    if "Freeway" in args.env and (args.model == "natural" or args.model == "diffusion_history"):  
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-natural.model'))
        #policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-DP-DQN-O.pth'))
    if "Freeway" in args.env and args.model == "sa-dqn-convex":
        policy = model_setup(cfg.env.train.id, test_env, True, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-convex.model'))
    if "Freeway" in args.env and args.model == "sa-dqn-pgd":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, False, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-pgd.model'))
    if "Freeway" in args.env and args.model == "wocar":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, False, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-wocar-pgd.pth'))
    if "Freeway" in args.env and args.model == "car-dqn-pgd":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway_car_pgd.pth'))

    if "Freeway" in args.env and "dp-dqn" in args.model:
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-DP-DQN-O.pth'))
        denoiser.load_state_dict(torch.load("/results_Freeway/model-150.pt")['model'])
        denoiser.to(device)

    
    if "Pong" in args.env:
        net = torch.load("/ae/pong_autoencoder49").to(device)
    if "Freeway" in args.env:
        net = torch.load("/ae/freeway_autoencoder49").to(device)

    diff = torch.nn.MSELoss(reduce= 'sum')
    last_count = 0

    # for interval in [16,8,2]:
    total_rew_write = []
    total_mani_write = []
    total_dev_write = []
    total_invalid_write = []
    pgd_time = []
    attack_time = []
    pgd_1_error = []
    pgd_3_error = []
    pgd_15_error = []
    minbest_1_error = []
    minbest_3_error = []
    minbest_15_error = [] 
    minbest_time = []
    bc_error = []
    blur_error = []
    shift_error = []
    rotate_error = []

    ours_error_inf = []
    ours_error = []
    ours_error_was = []
    true_error_was = []
    ours_error_was_1 = []
    pgd_1_error_was = []
    pgd_15_error_was = []
    seed = 1001
    total_diff_count = 0

    # all_tests = [1,2,3,4,5,6]
    # for try_strength in all_tests:
    for i in range(5):
        ours_error = []
        end = False
        truc = False
        obs, _ = test_env.reset(seed = i+seed)
        #obs, _ = test_env.reset()
        count = 0
        obs_his = torch.zeros((4,1,84,84)).to(device)
        perturb_obs_his = torch.zeros((4,1,84,84)).to(device)
        perturb_obs_his_1 = torch.zeros((4,1,84,84)).to(device)
        perturb_obs_his_2 = torch.zeros((4,1,84,84)).to(device)
        print(obs.shape)
        for j in range(4):
            obs_his[j] = obs
            perturb_obs_his[j] = obs
            perturb_obs_his_1[j] = obs
            perturb_obs_his_2[j] = obs
        if 'highway' not in args.env:
            act_his = torch.zeros(4, dtype= torch.int).to(device)
        else:
            act_his = torch.zeros((4, 2), dtype=torch.float).to(device)
        obs_his[-1] = obs
        perturb_obs_his[-1] = obs
        perturb_obs_his_1[-1] = obs
        perturb_obs_his_2[-1] = obs
        generated = None 
        victim_gen = None
        total_rew = 0
        target_act = 0
        optimal_act = 1
        count_sec = 0
        count_div = 0
        tmp = obs.squeeze(0)
        # if 'highway' not in args.env and 'airsim' not in args.env:
        #     tmp = torch.permute(tmp,(0,2,1))
        if 'highway' not in args.env and (not isinstance(policy, DQN)):
            tmp = torch.permute(tmp,(0,2,1))
        l_infinite_diff = 0
        invalid_set = set()
        attacked_count = []
        diff_count = []
        last_perturb_1 = obs
        last_perturb_15 = obs
        state_imt = []
        total_attacked = 0
        attack_max = args.attack_rate

        while not (end or truc):
            act_his = act_his.roll(-1)
            if "highway" in args.env:
                if generated == None:
                    act,_ = policy.predict(tmp.cpu())
                    act = np.round(act, decimals=1)
                elif args.model == "diffusion_history":
                    act,_ = policy.predict(victim_gen.cpu())
                    act = np.round(act, decimals=1)
                else:
                    act,_ = policy.predict(generated.cpu())
                    act = np.round(act, decimals=1)
            elif "airsim" in args.env:
                if generated == None:
                #if True:
                    if isinstance(policy, DQN):
                        act,_ = policy.predict((norm_zero_pos(tmp).cpu().numpy()*255).transpose(0,2,1))
                    else:
                        act = policy.act(norm_zero_pos(tmp), 0)
                elif args.model == "diffusion_history":
                    if victim_gen == None:
                        act,_ = policy.predict((norm_zero_pos(tmp).cpu().numpy()*255).transpose(0,2,1))
                    else:
                        act,_ = policy.predict(norm_zero_pos(victim_gen).cpu().numpy()*255)
                elif args.model == "natural" or "dp-dqn" in args.model:
                    act,_ = policy.predict((norm_zero_pos(generated).cpu().numpy()*255))
                else:
                    tmp1 = norm_zero_pos(torch.permute(generated, (0,2,1)))
                    act = policy.act(tmp1, 0)
            else:
                if generated == None:
                    act = policy.act(norm_zero_pos(tmp), 0)
                else:
                    if args.model == "diffusion_history":
                        tmp2 = norm_zero_pos(torch.permute(victim_gen, (0,2,1)))
                        act = policy.act(tmp2, 0)
                    elif args.model == "dp-dqn-history" or args.model == "dp-dqn":
                        if victim_gen == None:
                            act = policy.act(norm_zero_pos(tmp), 0)
                        else:
                            tmp2 = norm_zero_pos(torch.permute(victim_gen, (0,2,1)))
                            act = policy.act(tmp2, 0)
                    elif args.attack == "pgd":
                        act = policy.act(generated, 0)
                    else:
                        tmp1 = norm_zero_pos(torch.permute(generated, (0,2,1)))
                        act = policy.act(tmp1, 0)
                        #act = policy.act(norm_zero_pos(tmp), 0)
                    # act_1 = policy.act(norm_zero_pos(tmp),0)
                    # print(act, act_1)
            print("target ",target_act, "actual_act", act, "optimal_act", optimal_act, "current_reward", total_rew, "count", count-1, "diff", l_infinite_diff)
            if count > 0 and 'highway' not in args.env and 'airsim' not in args.env:    
                if (target_act.detach().cpu() == act.detach().cpu() and (target_act.detach().cpu()!=optimal_act.detach().cpu())):
                    count_sec  += 1 
                if (act.detach().cpu()!=optimal_act.detach().cpu()):
                    attacked_count.append(count-1)
                    count_div += 1
            elif 'airsim' in args.env:
                if (target_act == act and (target_act!=optimal_act)):
                    count_sec  += 1 
                if (act!=optimal_act):
                    attacked_count.append(count-1)
                    count_div += 1
            
            if "highway" in args.env:
                act_1 = torch.tensor(optimal_act).cuda()
                act = torch.tensor(act).cuda()
                act_his[-1] = act
            elif "airsim" in args.env:
                act = torch.tensor(act).cuda()
                act = act.unsqueeze(0)
                act_his[-1] = act
            else:
                #act_his[-1] = act
                act_his[-1] = act
            #print(act)

            obs_input = obs_his.unsqueeze(0)
            act_input = act_his.unsqueeze(0)
            perturb_obs_his_input = perturb_obs_his.unsqueeze(0)
            perturb_obs_his_input_1 = perturb_obs_his_1.unsqueeze(0)
            perturb_obs_his_input_2 = perturb_obs_his_2.unsqueeze(0)

            if "highway" in args.env:
                act = act.unsqueeze(0)
                act = act.cpu()
            obs, rew, end, truc, _ = test_env.step(act)
            if "Pong" in args.env:
                if rew!=0:
                    print("sync begin")
                    last_count = count
            tmp = obs.squeeze(0)
            if "highway" not in args.env:
                tmp = torch.permute(tmp,(0,2,1))
                if "airsim" in args.env:
                    if isinstance(policy, DQN):
                        optimal_act,_ = policy.predict((norm_zero_pos(tmp).cpu().numpy()*255).transpose(0,2,1))
                    else:
                        optimal_act = policy.act(norm_zero_pos(tmp), 0)
                else:
                    optimal_act = policy.act(norm_zero_pos(tmp), 0)
                target_act = torch.randint(low=0, high=test_env.num_actions, size= (obs.size(0),), device=obs.device)
                while target_act == optimal_act:
                    target_act = torch.randint(low=0, high=test_env.num_actions, size= (obs.size(0),), device=obs.device)
            else:
                optimal_act, _ = policy.predict(tmp.cpu())
                target_act = -optimal_act

            valid_gen = False
            if "Pong" in args.env:
                if args.model!="natural" and args.model!="diffusion_history" and args.model != "dp-dqn-history":
                    try_strength = 2
                else:
                    try_strength = 4
            if "Freeway" in args.env:
                if args.model!="natural" and args.model!="diffusion_history" and args.model != "dp-dqn-history":
                    try_strength = 6
                else:
                    try_strength = try_strength
            if "Bank" in args.env:
                try_strength = 4

            if "Road" in args.env:
                try_strength = 6

            if "doom" in args.env:
                if args.model!="natural" and args.model!="diffusion_history" and args.model != "dp-dqn-history":
                    try_strength = 2.5
                else:
                    try_strength = 4

            if "highway" in args.env:
                try_strength = 4

            if "airsim" in args.env:
                try_strength = 0.5

            while not valid_gen:    
                ours_start = time.time()
                if "highway" not in args.env:
                    if isinstance(policy, DQN):
                        true_state_in = (torch.permute(norm_zero_pos(tmp),(0,2,1))*255).unsqueeze(0)
                        current_q = policy.q_net(true_state_in)[0]
                    else:
                        current_q = policy.forward(norm_zero_pos(tmp))[0]
                    q_diff = (torch.max(current_q) - torch.min(current_q)).cpu().detach().numpy()
                    state_imt.append(q_diff)
                    cali_per = (1-attack_max)*100
                    if (q_diff > np.percentile(state_imt, cali_per) and total_attacked<int(len(state_imt)*attack_max)) or attack_max == 1.0:
                        print("Find important state to attack!")
                        num_actions = test_env.num_actions
                        rand_act_his = act_his.clone()
                        rand_act_his[-1] = num_actions
                        # rand_act_his[-2] = num_actions
                        # rand_act_his[-3] = num_actions
                        # rand_act_his[-4] = num_actions
                        rand_act_his_input = rand_act_his.unsqueeze(0)
                        if args.attack == "imagine":
                            generated, _ = wm_env.sampler.sample_next_obs_classifier_guide_fade_v(perturb_obs_his_input, rand_act_his_input, target_act, policy, try_strength, norm_zero_pos(tmp), net)
                        elif args.attack == "real":
                            generated,_ = wm_env.sampler.sample_next_obs_classifier_guide_fade_v(obs_input, act_input, target_act, policy, try_strength, norm_zero_pos(tmp) ,net, true_obs = True)
                        total_attacked+=1
                    else:
                        if args.attack == "imagine":
                            generated,_ = wm_env.sampler.sample_next_obs(perturb_obs_his_input, act_input)
                        else:
                            generated = obs
                else:
                    tmp = tmp.unsqueeze(0)
                    optimal_act = torch.tensor(optimal_act).cuda()
                    target_act = torch.tensor(target_act).cuda()
                    current_q,p1,_ = policy.policy.evaluate_actions(tmp, optimal_act)
                    estimate_worst,p2,_ = policy.policy.evaluate_actions(tmp, target_act)
                    q_diff = current_q*p1 - estimate_worst*p2
                    q_diff = q_diff.cpu().detach().numpy()
                    state_imt.append(q_diff)
                    cali_per = (1-attack_max)*100
                    if (q_diff > np.percentile(state_imt, cali_per) and total_attacked<int(len(state_imt)*attack_max)) or attack_max == 1.0:
                        print("Find important state to attack!")
                        rand_act_his = act_his.clone()
                        rand_act_his[-1] = torch.tensor([0.0,0.0])
                        rand_act_his_input = rand_act_his.unsqueeze(0)
                        if args.attack == "imagine":
                            generated, _ = wm_env.sampler.sample_next_obs_classifier_guide_fade_ppo(perturb_obs_his_input, rand_act_his_input, target_act, policy, try_strength, (tmp), net)
                        elif args.attack == "real":
                            generated,_ = wm_env.sampler.sample_next_obs_classifier_guide_fade_ppo(obs_input, act_input, target_act, policy, try_strength, (tmp) ,net, true_obs = True)
                        total_attacked+=1
                    else:
                        if args.attack == "imagine":
                            generated,_ = wm_env.sampler.sample_next_obs(perturb_obs_his_input, act_input)
                        else:
                            generated = obs

                ours_end = time.time()
                attack_time.append(ours_end - ours_start)
                to_check = norm_zero_pos((generated))


                ours_error.append(torch.norm(to_check - net(to_check), p =2).cpu().numpy())

                if False:
                #if loss >= 0.0003:
                #if loss >= 0.05:
                    print("invalid detected, retry with lower strength")
                    invalid_set.add(count)
                    try_strength = max(try_strength - 0.1, 0)
                    if try_strength <= 1:
                        break
                else:
                    valid_gen = True
            if args.model == 'diffusion_history' or args.model == "dp-dqn-history" or args.model == "dp-dqn":
            #if True:
                victim_gen,_ = wm_env.sampler.sample_next_obs(perturb_obs_his_input, act_input)
                #victim_gen,_ = wm_env.sampler.sample_next_obs(obs_input, act_input)
                victim_gen = victim_gen.squeeze(0)
            # generated,_ = wm_env.sampler.sample_next_obs(obs_input, act_input)
            generated = generated.squeeze(0)

            l_infinite_diff = torch.max(torch.abs(norm_zero_pos(torch.permute(generated, (0,2,1))) - norm_zero_pos(tmp)))
            ours_error_inf.append(l_infinite_diff.cpu().numpy())
            
            total_rew+=rew
            obs_his = obs_his.roll(-1,dims=0)
            obs_his[-1] = obs
            perturb_obs_his = perturb_obs_his.roll(-1, dims= 0)
            perturb_obs_his_1 = perturb_obs_his_1.roll(-1, dims = 0)
            perturb_obs_his_2 = perturb_obs_his_2.roll(-1, dims = 0)
            # last_perturb_1 = att_state_tensor_1
            # last_perturb_15 = att_state_tensor_15
            # if count % interval !=0:
            if True:
                if count<=5 and "airsim" not in args.env:
                    perturb_obs_his[-1] = obs
                    perturb_obs_his[0] = obs_his[0]
                    perturb_obs_his[1] = obs_his[1]
                    perturb_obs_his[2] = obs_his[2]
                    perturb_obs_his_1[-1] = obs
                    perturb_obs_his_1[0] = obs_his[0]
                    perturb_obs_his_1[1] = obs_his[1]
                    perturb_obs_his_1[2] = obs_his[2]
                else:
                    if count-last_count<=20 and "airsim" not in args.env and "Pong" in args.env:
                        perturb_obs_his[-1] = obs
                        perturb_obs_his[0] = obs_his[0]
                        perturb_obs_his[1] = obs_his[1]
                        perturb_obs_his[2] = obs_his[2]
                    else:
                        perturb_obs_his[-1] = generated
            count += 1
            if count >= 5000:
                break
        print("actual attack percent", total_attacked/count)
        print("end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default="imagine", help="attack types, random or minq")
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='environment types')
    parser.add_argument('--model', type=str, default='natural', help="defense types, natural, sa-dqn-pgd, sa-dqn-convex, wocar, diffsuion_history")
    parser.add_argument('--record_change', type= bool, default= False, help="record semantic change images")
    parser.add_argument('--attack_rate', type= float, default=0.15, help="max rate to attack")
    args = parser.parse_args()
    main(args)
