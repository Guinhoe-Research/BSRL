import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch.nn as nn
import torch

from torch.distributions import Categorical
import torch.nn.functional as F

from core.environment import Environment
from core.configs import (
    EnvironmentConfig,
    RewardConfig,
    TrainingConfig,
    save_configs,
    load_configs,
    _config_to_dict,
)
from ppo.actor_critic_model import ActorCritic


# ---------------------------------------------------------------------------
# Configs — edit these or load from a JSON file
# ---------------------------------------------------------------------------

env_cfg = EnvironmentConfig(num_agents=3, SEE_CARD_COUNTS=True)

reward_cfg = RewardConfig(
    claim_shaping=0.1,
    challenge_correct=1.0,
    challenge_incorrect=-1.0,
    claimer_caught=-1.0,
    claimer_survived=1.0,
    truthful_card_bonus=0.0,
    successful_bluff=0.0,
    challenge_cost=0.0,
    win_bonus=10.0,
)

train_cfg = TrainingConfig(
    num_envs=32,
    num_epochs=50,
    trajectory_window=256,
    batch_size=32,
    clip=0.2,
    vf_coef=0.75,
    ent_coef=0.01,
    gamma=0.99,
    lam=0.95,
    lr=3e-4,
    max_grad_norm=0.5,
    hidden_dim=64,
    seed=42,
)

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

ACTION_DIM = 19
OBSERVATION_DIM = 20 + env_cfg.num_agents if env_cfg.SEE_CARD_COUNTS else 20

# ---------------------------------------------------------------------------
# Model & optimiser
# ---------------------------------------------------------------------------

torch.manual_seed(train_cfg.seed)

ppomodel = ActorCritic(
    obs_dim=OBSERVATION_DIM,
    act_dim=ACTION_DIM,
    has_encoder=True,
    encode_claimants=env_cfg.ENCODE_CLAIMANTS,
    positional_embeddings=env_cfg.POSITIONAL_EMBEDDINGS,
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ppomodel.parameters(), lr=train_cfg.lr)


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

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
    for _ in range(train_cfg.trajectory_window):
        state       = obs["observation"]
        action_mask = obs["action_mask"]
        claim_seqs = obs["claim_seq"]

        with torch.no_grad():
            action, log_prob, _ = ppomodel.get_action(state.unsqueeze(0), action_mask.unsqueeze(0), claim_seqs.unsqueeze(0))
            _, value         = ppomodel(state.unsqueeze(0), claim_seqs.unsqueeze(0))
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

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
    advantages = torch.zeros(train_cfg.trajectory_window, dtype=torch.float32)
    last_gae = 0.0

    with torch.no_grad():
        _, last_value = ppomodel(obs["observation"].unsqueeze(0), obs["claim_seq"].unsqueeze(0))
        last_value = last_value.squeeze()

    for t in reversed(range(train_cfg.trajectory_window)):
        next_value          = last_value if t == train_cfg.trajectory_window - 1 else values[t + 1]
        next_non_terminal   = 1.0 - dones[t]
        delta               = rewards_buf[t] + train_cfg.gamma * next_value * next_non_terminal - values[t]
        last_gae            = delta + train_cfg.gamma * train_cfg.lam * next_non_terminal * last_gae
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
        new_env = Environment(env_cfg, reward_cfg)
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


# ---------------------------------------------------------------------------
# Training metadata — includes configs for reproducibility
# ---------------------------------------------------------------------------

training_metadata = {
    "configs": {
        "environment": _config_to_dict(env_cfg),
        "reward": _config_to_dict(reward_cfg),
        "training": _config_to_dict(train_cfg),
    },
    "losses": [],
    "policy_losses": [],
    "value_losses": [],
    "entropies": [],
    "epoch_mean_rewards": [],
    "epoch_mean_policy_loss": [],
    "epoch_mean_value_loss": [],
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

for epoch in range(train_cfg.num_epochs):
    print(f"Epoch {epoch + 1}/{train_cfg.num_epochs}")
    ### Trajectory Farming
    # Collect trajectories
    observations, actions, log_probs, values, advantages, returns, action_masks, claim_sequences = vec_collection(train_cfg.num_envs)

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

    for b in range(0, observations.size(0), train_cfg.batch_size):
        batch_slice = slice(b, b + train_cfg.batch_size)

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
        right = torch.clamp(ratio, 1.0 - train_cfg.clip, 1.0 + train_cfg.clip) * normalized_advantages
        policy_loss = -torch.min(left, right).mean()

        value_loss = criterion(value.squeeze(), batch_returns)

        true_loss = policy_loss + train_cfg.vf_coef * value_loss - train_cfg.ent_coef * entropy

        optimizer.zero_grad()
        true_loss.backward()

        nn.utils.clip_grad_norm_(ppomodel.parameters(), train_cfg.max_grad_norm)
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

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

torch.save(training_metadata, "ppo_training_metadata.pth")
torch.save(ppomodel.state_dict(), "ppo_model_100_50.pth")

# Also save configs as human-readable JSON for easy editing
save_configs(
    "training_configs.json",
    env=env_cfg,
    reward=reward_cfg,
    training=train_cfg,
)
print("Saved training_configs.json")
