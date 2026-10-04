# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 15:00:06 2025

@author: TAKO
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, numpy as np, ray
from typing import Optional
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from envs.pistonball_env import env_creator
from models.cnn_model_v2 import CNNModelV2

def parse_args():
    p = argparse.ArgumentParser("RLlib PPO for PettingZoo Pistonball")
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--stop-timesteps", type=int, default=5_000_000)
    p.add_argument("--storage-path", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if (args.eval or args.resume) and not args.checkpoint_path:
        p.error("--checkpoint-path ist erforderlich bei --eval/--resume")
    return args

def build_config(num_workers: int, seed: int, num_gpus: int):
    env_name = "pistonball_v6"
    # Registrierung (muss vor Config-Aufbau passieren)
    register_env(env_name, lambda cfg: ParallelPettingZooEnv(env_creator(cfg)))
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    cfg = PPOConfig()
    cfg = cfg.environment(env=env_name,
                          clip_actions=True,
                          env_config={"seed": seed})
    # neue API-Namen in Ray 2.48.0
    cfg.num_env_runners = num_workers
    cfg.rollout_fragment_length = 128

    cfg = cfg.training(
        train_batch_size=512,
        lr=2e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        model={"custom_model": "CNNModelV2"},
    )
    # neue Felder statt sgd_minibatch_size/num_sgd_iter
    cfg.minibatch_size = 64
    cfg.num_epochs = 10

    cfg = cfg.framework("torch")
    cfg = cfg.resources(num_gpus=num_gpus)
    cfg = cfg.reporting(min_time_s_per_iteration=1)
    return cfg, env_name

def train(args):
    storage_path = args.storage_path or os.path.expanduser("~/ray_results/pistonball_v6")
    num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config, env_name = build_config(args.num_workers, args.seed, num_gpus)

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    run_kwargs = dict(
        name="PPO",
        stop={"timesteps_total": args.stop_timesteps},
        checkpoint_freq=10,
        storage_path=storage_path,       # statt local_dir
        config=config.to_dict(),
    )
    if args.resume and args.checkpoint_path:
        run_kwargs["restore"] = os.path.expanduser(args.checkpoint_path)
    tune.run("PPO", **run_kwargs)

def evaluate(checkpoint_path: str, episodes: int = 5):
    cp = os.path.expanduser(checkpoint_path)
    if not os.path.exists(cp):
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {cp}")

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    algo: PPO = PPO.from_checkpoint(cp)

    env = ParallelPettingZooEnv(env_creator({"render_mode": "human"}))
    obs = env.reset(seed=0)
    for ep in range(episodes):
        done = {agent: False for agent in env.possible_agents}
        ep_rewards = {agent: 0.0 for agent in env.possible_agents}
        while not all(done.values()):
            actions = {agent: algo.compute_single_action(ob, explore=False)
                       for agent, ob in obs.items()}
            obs, rewards, done, infos = env.step(actions)
            for k, r in rewards.items():
                ep_rewards[k] += float(r)
        print(f"Episode {ep+1}: mean reward = {np.mean(list(ep_rewards.values())):.3f}")
    env.close()
    algo.stop()
    ray.shutdown()

def main():
    args = parse_args()
    if args.eval:
        evaluate(args.checkpoint_path)
    else:
        train(args)

if __name__ == "__main__":
    main()
