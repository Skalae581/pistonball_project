# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 15:32:44 2025

@author: TAKO
"""

"""
Basic code which shows what it's like to run PPO on the Pistonball env using the parallel API.
Inspired by CleanRL. Minimal logging/saving for clarity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical
from pettingzoo.butterfly import pistonball_v6
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# -------------------- Hyperparams --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ent_coef = 0.1
vf_coef = 0.1
clip_coef = 0.2
gamma = 0.99
batch_size = 64
stack_size = 4
frame_size = (64, 64)
max_cycles = 125
total_episodes = 500            # vorher 0 -> es würde nie trainieren
left_bonus = 0.15               # Zusatzbelohnung für "weiter nach links"

# -------------------- Agent --------------------
class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# -------------------- Utils --------------------
def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = obs.transpose(0, -1, 1, 2)  # NHWC -> NCHW
    obs = torch.tensor(obs).to(device)
    return obs

def batchify(x, device):
    """Converts PZ style dict -> torch batch."""
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x).to(device)
    return x

def unbatchify(x, env):
    """Converts torch batch -> PZ style dict."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x

def shape_rewards_left(rewards_dict, actions_tensor, env, left_bonus: float):
    """Belohnt diskrete Aktionen, die weiter links liegen, mit einem Bonus."""
    shaped = {}
    n = env.action_space(env.possible_agents[0]).n
    center = (n - 1) / 2.0  # z.B. n=5 -> center=2
    actions = actions_tensor.detach().cpu().numpy()  # [num_agents]
    for i, agent in enumerate(env.possible_agents):
        a = float(actions[i])
        leftness = max(0.0, (center - a) / max(center, 1.0))  # [0..1]
        shaped[agent] = float(rewards_dict[agent]) + left_bonus * leftness
    return shaped

# -------------------- Env setup --------------------
env = pistonball_v6.parallel_env(
    render_mode="rgb_array", continuous=False, max_cycles=max_cycles
)
env = color_reduction_v0(env)
env = resize_v1(env, frame_size[0], frame_size[1])
env = frame_stack_v1(env, stack_size=stack_size)

num_agents = len(env.possible_agents)
num_actions = env.action_space(env.possible_agents[0]).n
observation_size = env.observation_space(env.possible_agents[0]).shape

# -------------------- Learner setup --------------------
agent = Agent(num_actions=num_actions).to(device)
optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)
run_name = f"pistonball_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
global_step = 0  # optionaler Zähler in Umweltschritten

# -------------------- Load weights (optional) --------------------
try:
    agent.load_state_dict(torch.load("pistonball_agent.pth", map_location=device))
    print("Loaded previous agent weights.")
except Exception as e:
    print("No previous agent weights found, starting from scratch.")
    print(e)

# -------------------- Episode buffers --------------------
end_step = 0
total_episodic_return = 0
rb_obs = torch.zeros((max_cycles, num_agents, stack_size, *frame_size), device=device)
rb_actions = torch.zeros((max_cycles, num_agents), device=device)
rb_logprobs = torch.zeros((max_cycles, num_agents), device=device)
rb_rewards = torch.zeros((max_cycles, num_agents), device=device)
rb_terms = torch.zeros((max_cycles, num_agents), device=device)   # done-mask (0/1)
rb_values = torch.zeros((max_cycles, num_agents), device=device)

# -------------------- Training --------------------
for episode in range(total_episodes):
    with torch.no_grad():
        next_obs, info = env.reset(seed=None)
        total_episodic_return = 0
        end_step = max_cycles

        for step in range(0, max_cycles):
            obs = batchify_obs(next_obs, device)

            actions, logprobs, _, values = agent.get_action_and_value(obs)

            next_obs, rewards, terms, truncs, infos = env.step(
                unbatchify(actions, env)
            )

            # Reward shaping
            shaped_rewards = shape_rewards_left(rewards, actions, env, left_bonus)

            # Store
            rb_obs[step] = obs
            rb_rewards[step] = batchify(shaped_rewards, device)
            rb_terms[step] = batchify(terms, device).float()  # 1.0 bei done
            rb_actions[step] = actions
            rb_logprobs[step] = logprobs
            rb_values[step] = values.flatten()

            total_episodic_return += rb_rewards[step].mean().item()

            # Stop when any agent done/truncated
            if any(terms.values()) or any(truncs.values()):
                end_step = step + 1
                break

    # Returns & advantages (einfach, stabil)
    with torch.no_grad():
        rb_returns = torch.zeros_like(rb_rewards, device=device)
        rb_advantages = torch.zeros_like(rb_rewards, device=device)
        next_return = torch.zeros((num_agents,), device=device)

        for t in reversed(range(end_step)):
            done_mask = rb_terms[t].float()  # 1.0 wenn terminal
            next_return = rb_rewards[t] + gamma * next_return * (1.0 - done_mask)
            rb_returns[t] = next_return

        rb_advantages[:end_step] = rb_returns[:end_step] - rb_values[:end_step]

    # Flatten episode to transitions
    b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
    b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
    b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
    b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
    b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
    b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

    # PPO update
    b_index = np.arange(len(b_obs))
    clip_fracs = []
    for repeat in range(3):
        np.random.shuffle(b_index)
        for start in range(0, len(b_obs), batch_size):
            end = start + batch_size
            batch_index = b_index[start:end]

            _, newlogprob, entropy, value = agent.get_action_and_value(
                b_obs[batch_index], b_actions.long()[batch_index]
            )
            logratio = newlogprob - b_logprobs[batch_index]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_fracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

            # normalize advantages (und benutzen)
            advantages_norm = b_advantages[batch_index]
            advantages_norm = (advantages_norm - advantages_norm.mean()) / (advantages_norm.std() + 1e-8)

            # Policy loss
            pg_loss1 = -advantages_norm * ratio
            pg_loss2 = -advantages_norm * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss (clipped)
            value = value.flatten()
            v_loss_unclipped = (value - b_returns[batch_index]) ** 2
            v_clipped = b_values[batch_index] + torch.clamp(value - b_values[batch_index], -clip_coef, clip_coef)
            v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # simple metrics
    y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    mean_clip = float(np.mean(clip_fracs))
    curr_lr = optimizer.param_groups[0]["lr"]
    global_step += end_step  # oder end_step * num_agents, wenn du magst
    
    # per Episode loggen (x-Achse = Episode)
    writer.add_scalar("loss/policy", float(pg_loss.item()), episode)
    writer.add_scalar("loss/value", float(v_loss.item()), episode)
    writer.add_scalar("loss/entropy", float(entropy_loss.item()), episode)
    writer.add_scalar("kls/approx_kl", float(approx_kl.item()), episode)
    writer.add_scalar("kls/old_approx_kl", float(old_approx_kl.item()), episode)
    writer.add_scalar("ppo/clip_fraction", mean_clip, episode)
    writer.add_scalar("train/episodic_return", float(total_episodic_return), episode)
    writer.add_scalar("train/episode_length", int(end_step), episode)
    writer.add_scalar("value/explained_variance", float(explained_var), episode)
    writer.add_scalar("opt/lr", float(curr_lr), episode)
    
    # optional zusätzlich mit global_step (x-Achse = Schritte)
    writer.add_scalar("by_step/episodic_return", float(total_episodic_return), global_step)
    writer.flush()

    print(f"Training episode {episode}")
    print(f"Episodic Return: {total_episodic_return:.3f}")
    print(f"Episode Length: {end_step}")
    print("")
    print(f"Value Loss: {v_loss.item():.6f}")
    print(f"Policy Loss: {pg_loss.item():.6f}")
    print(f"Old Approx KL: {old_approx_kl.item():.6f}")
    print(f"Approx KL: {approx_kl.item():.6f}")
    print(f"Clip Fraction: {np.mean(clip_fracs):.3f}")
    print(f"Explained Variance: {explained_var:.3f}")
    print("\n-------------------------------------------\n")

# Save weights
torch.save(agent.state_dict(), "pistonball_agent.pth")
writer.close()
# -------------------- Render policy --------------------
env = pistonball_v6.parallel_env(render_mode="human", continuous=False)
env = color_reduction_v0(env)
env = resize_v1(env, 64, 64)
env = frame_stack_v1(env, stack_size=4)

agent.eval()
with torch.no_grad():
    for episode in range(5):
        obs, infos = env.reset(seed=None)
        obs_batch = batchify_obs(obs, device)
        while True:
            actions, logprobs, _, values = agent.get_action_and_value(obs_batch)
            obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
            if any(terms.values()) or any(truncs.values()):
                break
            obs_batch = batchify_obs(obs, device)
