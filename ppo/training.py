import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch.nn as nn
import torch

from torch.distributions import Categorical
import torch.nn.functional as F


# VARIABLES
OBSERVATION_DIM = 22
ACTION_DIM = 19

NUM_ENVS = 32
NUM_EPOCHS = 25
TRAJECTORY_WINDOW = 256

BATCH_SIZE = 32

CLIP = 0.2
VF_COEF = 0.75
ENT_COEF = 0.01
EPOCHS = 25
MB_SIZE = 256

GAMMA = 0.99
LAMBDA = 0.95

NUM_EPISODES = 25


from core.environment import Environment
from core.configs import EnvironmentConfig
from ppo.actor_critic_model import ActorCritic

torch.manual_seed(42)

cfg = EnvironmentConfig(num_agents=2, SEE_CARD_COUNTS=True)
env = Environment(cfg)

ppomodel = ActorCritic(obs_dim=OBSERVATION_DIM, act_dim=ACTION_DIM)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ppomodel.parameters(), lr=3e-4)

def obtain_trajectories(env):
    observations = []
    actions      = []
    log_probs    = []
    values       = []
    rewards_buf  = []
    dones        = []
    action_masks = []
    claim_sequences = []

    obs, _ = env.reset()
    for _ in range(TRAJECTORY_WINDOW):
        state       = obs["observation"]
        action_mask = obs["action_mask"]
        claim_seqs = obs["claim_seq"]

        with torch.no_grad():
            action, log_prob, _ = ppomodel.get_action(state, action_mask, claim_seqs)
            _, value         = ppomodel(state)

        observations.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value.squeeze())
        action_masks.append(action_mask)
        claim_sequences.append(claim_seqs)

        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        rewards_buf.append(torch.tensor(reward, dtype=torch.float32))
        dones.append(torch.tensor(float(done), dtype=torch.float32))

        if done:
            obs, _ = env.reset()

    # Stack into tensors for PPO update
    observations = torch.stack(observations)   # (T, obs_dim)
    actions      = torch.stack(actions)        # (T,)
    log_probs    = torch.stack(log_probs)      # (T,)
    values       = torch.stack(values)         # (T,)
    rewards_buf  = torch.stack(rewards_buf)    # (T,)
    dones        = torch.stack(dones)          # (T,)
    action_masks    = torch.stack(action_masks)     # (T, act_dim)
    claim_sequences = torch.stack(claim_sequences)  # (T, 52, 2)

    # GAE (Generalized Advantage Estimation)
    advantages = torch.zeros(TRAJECTORY_WINDOW, dtype=torch.float32)
    last_gae = 0.0

    with torch.no_grad():
        _, last_value = ppomodel(obs["observation"])
        last_value = last_value.squeeze()

    for t in reversed(range(TRAJECTORY_WINDOW)):
        next_value          = last_value if t == TRAJECTORY_WINDOW - 1 else values[t + 1]
        next_non_terminal   = 1.0 - dones[t]
        delta               = rewards_buf[t] + GAMMA * next_value * next_non_terminal - values[t]
        last_gae            = delta + GAMMA * LAMBDA * next_non_terminal * last_gae
        advantages[t]       = last_gae

    returns = advantages + values  # targets for the value function

    return observations, actions, log_probs, values, advantages, returns, action_masks, claim_sequences

def vec_collection(num_envs):
    v_observations = []
    v_actions      = []
    v_log_probs    = []
    v_values       = []
    v_advantages   = []
    v_returns      = []
    v_action_masks = []
    v_claim_sequences = []

    for _ in range(num_envs):
        new_env = Environment(cfg)
        o, a, lp, v, adv, r, am, cs = obtain_trajectories(new_env)
        v_observations.append(o)
        v_actions.append(a)
        v_log_probs.append(lp)
        v_values.append(v)
        v_advantages.append(adv)
        v_returns.append(r)
        v_action_masks.append(am)
        v_claim_sequences.append(cs)
    observations    = torch.stack(v_observations)     # (num_envs, T, obs_dim)
    actions         = torch.stack(v_actions)          # (num_envs, T)
    log_probs       = torch.stack(v_log_probs)        # (num_envs, T)
    values          = torch.stack(v_values)           # (num_envs, T)
    advantages      = torch.stack(v_advantages)       # (num_envs, T)
    returns         = torch.stack(v_returns)          # (num_envs, T)
    action_masks    = torch.stack(v_action_masks)     # (num_envs, T, act_dim)
    claim_sequences = torch.stack(v_claim_sequences)  # (num_envs, T, 52, 2)
    print("Trajectory collection complete.")
    print("Observations shape:", observations.shape)
    print("Actions shape:", actions.shape)
    print("Log probs shape:", log_probs.shape)
    print("Values shape:", values.shape)
    print("Advantages shape:", advantages.shape)
    print("Returns shape:", returns.shape)
    print("Action masks shape:", action_masks.shape)
    print("Claim sequences shape:", claim_sequences.shape)
    return observations, actions, log_probs, values, advantages, returns, action_masks, claim_sequences

training_metadata = {
    "losses": [],
    "policy_losses": [],
    "value_losses": [],
    "entropies": [],
    "epoch_mean_rewards": [],
    "epoch_mean_policy_loss": [],
    "epoch_mean_value_loss": [],
}

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    ### Trajectory Farming
    # Collect trajectories
    observations, actions, log_probs, values, advantages, returns, action_masks, claim_sequences = vec_collection(NUM_ENVS)

    epoch_rewards = returns.mean().item()
    training_metadata["epoch_mean_rewards"].append(epoch_rewards)

    observations = observations.reshape(-1, observations.size(-1))
    actions = actions.reshape(-1)
    log_probs = log_probs.reshape(-1)
    values = values.reshape(-1)
    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)
    action_masks = action_masks.reshape(-1, action_masks.size(-1))
    claim_sequences = claim_sequences.reshape(-1, claim_sequences.size(-2), claim_sequences.size(-1))

    epoch_policy_losses = []
    epoch_value_losses = []

    for b in range(0, observations.size(0), BATCH_SIZE):
        batch_slice = slice(b, b + BATCH_SIZE)

        batch_obs = observations[batch_slice]
        batch_actions = actions[batch_slice]
        batch_log_probs = log_probs[batch_slice]
        batch_values = values[batch_slice]
        batch_advantages = advantages[batch_slice]
        batch_returns = returns[batch_slice]
        batch_action_masks = action_masks[batch_slice]
        batch_claim_sequences = claim_sequences[batch_slice]

        logits, value = ppomodel(batch_obs, batch_claim_sequences)
        logits = logits.masked_fill(batch_action_masks == 0, float("-inf"))
        dist = Categorical(logits=logits)
        n_log_prob = dist.log_prob(batch_actions)
        entropy = dist.entropy().mean()

        ratio = (n_log_prob - batch_log_probs).exp()
        normalized_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
        left = ratio * normalized_advantages
        right = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * normalized_advantages
        policy_loss = -torch.min(left, right).mean()

        value_loss = criterion(value.squeeze(), batch_returns)

        true_loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

        optimizer.zero_grad()
        true_loss.backward()

        nn.utils.clip_grad_norm_(ppomodel.parameters(), 0.5)
        optimizer.step()

        training_metadata["losses"].append(true_loss.item())
        training_metadata["policy_losses"].append(policy_loss.item())
        training_metadata["value_losses"].append(value_loss.item())
        training_metadata["entropies"].append(entropy.item())
        epoch_policy_losses.append(policy_loss.item())
        epoch_value_losses.append(value_loss.item())

    mean_pl = sum(epoch_policy_losses) / len(epoch_policy_losses)
    mean_vl = sum(epoch_value_losses) / len(epoch_value_losses)
    training_metadata["epoch_mean_policy_loss"].append(mean_pl)
    training_metadata["epoch_mean_value_loss"].append(mean_vl)
    print(f"Epoch {epoch + 1} - Policy Loss: {mean_pl:.4f}  Value Loss: {mean_vl:.4f}  Mean Return: {epoch_rewards:.4f}")

torch.save(training_metadata, "ppo_training_metadata.pth")
torch.save(ppomodel.state_dict(), "ppo_model_100_32.pth")