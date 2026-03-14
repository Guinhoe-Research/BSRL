import torch.nn as nn
import torch

from torch.distributions import Categorical
import torch.nn.functional as F

# VARIABLES
OBSERVATION_DIM = 22
ACTION_DIM = 19

NUM_ENVS = 32
NUM_EPOCHS = 10
TRAJECTORY_WINDOW = 128

BATCH_SIZE = 64

CLIP = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
EPOCHS = 10
MB_SIZE = 256

GAMMA = 0.99
LAMBDA = 0.95

NUM_EPISODES = 25

class EmbeddingModel(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.pile_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, pile_obs):
        # pile_obs: (B, 52, 2)

        card_feats = self.pile_mlp(pile_obs)      # (B, 52, H)
        scores = self.attn_mlp(card_feats)        # (B, 52, 1)
        weights = torch.softmax(scores, dim=1)    # (B, 52, 1)

        pile_summary = (card_feats * weights).sum(dim=1)  # (B, H)
        return pile_summary, weights
    
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, has_encoder=False):
        super().__init__()
        self.has_encoder = has_encoder
        if has_encoder:
            obs_dim += hidden_dim  # Concatenate encoder output to obs

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.encoder = EmbeddingModel(output_dim=hidden_dim)

    def forward(self, obs, claims=None):
        if self.has_encoder and claims is not None:
            pile_summary, _ = self.encoder(claims)
            obs = torch.cat([obs, pile_summary], dim=-1)
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value

    def get_action(self, obs, action_mask, claims=None):
        if self.has_encoder and claims is not None:
            pile_summary, _ = self.encoder(claims)
            obs = torch.cat([obs, pile_summary], dim=-1)
        logits = self.actor(obs)
        logits = logits.masked_fill(action_mask == 0, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, logits
    
from core.environment import Environment
from core.configs import EnvironmentConfig

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
    action_masks = torch.stack(action_masks)  # (T, act_dim)

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
    print("Trajectory collection complete.")
    print("Observations shape:", torch.stack(v_observations).shape)
    print("Actions shape:", torch.stack(v_actions).shape)
    print("Log probs shape:", torch.stack(v_log_probs).shape)
    print("Values shape:", torch.stack(v_values).shape)
    print("Advantages shape:", torch.stack(v_advantages).shape)
    print("Returns shape:", torch.stack(v_returns).shape)
    print("Action masks shape:", torch.stack(v_action_masks).shape)
    print("Claim sequences shape:", torch.stack(v_claim_sequences).shape)
    return v_observations, v_actions, v_log_probs, v_values, v_advantages, v_returns, v_action_masks, v_claim_sequences

for epoch in range(NUM_EPOCHS):
    ### Trajectory Farming
    # Collect trajectories
    observations, actions, log_probs, values, advantages, returns, action_masks, claim_sequences = vec_collection(NUM_ENVS)
    
    observations = observations.reshape(-1, observations.size(-1))
    actions = actions.reshape(-1)
    log_probs = log_probs.reshape(-1)
    values = values.reshape(-1)
    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)
    action_masks = action_masks.reshape(-1, action_masks.size(-1))
    claim_sequences = claim_sequences.reshape(-1, claim_sequences.size(-1), claim_sequences.size(-1))
    
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

        n_action, n_log_prob, n_logits = ppomodel.get_action(batch_obs, batch_action_masks)
        _, value             = ppomodel(batch_obs)
        print("Log prob:", n_log_prob.size())
        print("Log probs:", batch_log_probs.size())

        ratio = n_log_prob.exp() / batch_log_probs.exp()
        normalized_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
        left = ratio * normalized_advantages
        right = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * normalized_advantages
        policy_loss = -torch.min(left, right).mean()

        value_loss = criterion(batch_values, batch_returns)

        masked_logits = n_logits.masked_fill(batch_action_masks == 0, float("-inf"))
        true_loss = policy_loss + value_loss * VF_COEF - ENT_COEF * Categorical(logits=masked_logits).entropy().mean()

        optimizer.zero_grad()
        true_loss.backward()

        nn.utils.clip_grad_norm_(ppomodel.parameters(), 0.5)
        optimizer.step()
