import torch
from torch import nn

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