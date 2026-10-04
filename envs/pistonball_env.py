# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:46:24 2025

@author: TAKO
"""

# envs/pistonball_env.py
import supersuit as ss
from pettingzoo.butterfly import pistonball_v6

def env_creator(config=None):
    config = config or {}
    render_mode = config.get("render_mode", None)

    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
        render_mode=render_mode,
    )
    # Nur SuperSuit/PettingZoo-Wrapper hier!
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.frame_stack_v1(env, 3)
    return env
