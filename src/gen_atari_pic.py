#!/usr/bin/env python3
"""
Atari trajectory dumper with policy network (minimal deps)
---------------------------------------------------------
- Hard-coded environment crop_shift and restrict_actions per env.
- Preserves original PNG format (cv2.imwrite with obs.transpose(2,1,0)).
- Uses your Atari wrappers and policy model (models.py).

Example:
  python gen_atari_pic.py --env PongNoFrameskip-v4 --num-trajs 3 --traj-len 150 \
    --policy-ckpt ./sa-dqn_models/models/Pong-natural.model --out ./airsim_pic_traj

"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import torch

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

from wrappers import make_atari, make_atari_cart, wrap_deepmind, wrap_pytorch, ObservationWrapper
from model import model_setup

# -----------------------------
# Hard-coded crop/restrict map
# -----------------------------
ENV_CFG = {
    "PongNoFrameskip-v4": dict(crop_shift=10, restrict_actions=4),
    "FreewayNoFrameskip-v4": dict(crop_shift=0, restrict_actions=0),
    "RoadRunnerNoFrameskip-v4": dict(crop_shift=20, restrict_actions=True),
    "AirsimCar-v0": dict(crop_shift=0, restrict_actions=0),
}

def _normalize_reset(env):
    out = env.reset()
    return out if isinstance(out, tuple) and len(out) == 2 else (out, {})

def _normalize_step(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        return out
    if len(out) == 4:
        obs, rew, done, info = out
        return obs, rew, bool(done), False, info
    if len(out) == 5:
        obs, rew, done, _unused, info = out
        return obs, rew, bool(done), False, info
    raise RuntimeError("Unexpected env.step() return format.")

def build_env(env_id: str):
    cfg = ENV_CFG.get(env_id, dict(crop_shift=0, restrict_actions=0))
    print(cfg)
    if "airsim" in env_id.lower():
        return gym.make(env_id)
    if "NoFrameskip" in env_id:
        env = make_atari(env_id)
        env = wrap_deepmind(env, episode_life=True, clip_rewards=False, frame_stack=False,
                            scale=False, color_image=False, central_crop=True,
                            crop_shift=cfg["crop_shift"], restrict_actions=cfg["restrict_actions"])
        env = wrap_pytorch(env)
        return env
    env = make_atari_cart(env_id)
    if "doom" in env_id.lower():
        env = ObservationWrapper(env)
        env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
    return env

def load_policy(env_id: str, env, ckpt: str | None, dueling: bool, model_width: int, use_cuda: bool):
    model = model_setup(env_id, env, robust_model=False, logger=None, use_cuda=use_cuda,
                        dueling=dueling, model_width=model_width)
    if ckpt:
        state = torch.load(ckpt, map_location=("cuda" if use_cuda and torch.cuda.is_available() else "cpu"))
        try:
            model.load_state_dict(state)
        except Exception:
            model.features.load_state_dict(state)
    model.eval()
    return model

def act(model, obs: np.ndarray, epsilon: float, use_cuda: bool) -> int:
    x = obs.astype(np.float32) / 255.0
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    if use_cuda and torch.cuda.is_available():
        t = t.cuda()
    with torch.no_grad():
        a = model.act(t, epsilon=epsilon)[0]
    return int(a)

def dump_trajectories(env_id: str, num_trajs: int, traj_len: int, out_dir: Path, policy_ckpt: str | None,
                      epsilon: float, dueling: bool, model_width: int, seed: int | None, use_cuda: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(env_id)
    if seed is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        except Exception:
            pass
    model = load_policy(env_id, env, policy_ckpt, dueling=dueling, model_width=model_width, use_cuda=use_cuda)
    total_steps = 0
    max_exp = max(1, num_trajs * traj_len)
    for traj_idx in range(num_trajs):
        traj_dir = out_dir / str(traj_idx)
        traj_dir.mkdir(parents=True, exist_ok=True)
        obs, _ = _normalize_reset(env)
        step_in_traj = 0
        while step_in_traj < traj_len:
            greedy_prob = max(1.0 - (total_steps*2 / max_exp), 0.0)
            eps_now = epsilon * greedy_prob
            action = act(model, obs, epsilon=eps_now, use_cuda=use_cuda)
            obs, reward, terminated, truncated, info = _normalize_step(env, action)
            img_path = traj_dir / f"{step_in_traj}.png"
            cv2.imwrite(str(img_path), obs.transpose(2, 1, 0))
            total_steps += 1
            step_in_traj += 1
            if terminated or truncated:
                obs, _ = _normalize_reset(env)
                break
    env.close()

def parse_args():
    ap = argparse.ArgumentParser(description="Dump Atari observations to PNG trajectories (hard-coded wrapper config)")
    ap.add_argument("--env", default="FreewayNoFrameskip-v4", help="Env id")
    ap.add_argument("--num-trajs", type=int, default=10)
    ap.add_argument("--traj-len", type=int, default=5000)
    ap.add_argument("--out", default="./freeway_pic_traj")
    ap.add_argument("--policy-ckpt", default="./src/pre_trained/Freeway-natural.model")
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--dueling", action="store_true", default= True)
    ap.add_argument("--model-width", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cuda", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    dump_trajectories(
        env_id=args.env,
        num_trajs=args.num_trajs,
        traj_len=args.traj_len,
        out_dir=Path(args.out),
        policy_ckpt=args.policy_ckpt,
        epsilon=args.epsilon,
        dueling=args.dueling,
        model_width=args.model_width,
        seed=args.seed,
        use_cuda=args.cuda,
    )

if __name__ == "__main__":
    main()