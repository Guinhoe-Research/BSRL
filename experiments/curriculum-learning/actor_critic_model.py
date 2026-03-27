import torch
from torch import nn

MAX_CLAIMS = 52

class ClaimCrossAttention(nn.Module):
    """Cross-attention: observation queries against claim history keys/values."""

    def __init__(self, obs_dim, claim_input_dim=2, embed_dim=64, num_heads=4, positional=False):
        super().__init__()
        # Project each (rank, count[, claimant]) into embed space
        self.kv_proj = nn.Linear(claim_input_dim, embed_dim)
        # Project flat observation into query
        self.q_proj = nn.Linear(obs_dim, embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True,
        )

        self.positional = positional
        if positional:
            self.pos_embed = nn.Embedding(MAX_CLAIMS, embed_dim)

    def forward(self, obs, pile_obs):
        # obs: (B, obs_dim)  pile_obs: (B, 52, claim_input_dim)

        # Build key/value from claim sequence
        kv = self.kv_proj(pile_obs)                         # (B, 52, E)
        if self.positional:
            kv = kv + self.pos_embed.weight[:kv.size(1)]

        # Build query from observation (single token)
        query = self.q_proj(obs).unsqueeze(1)               # (B, 1, E)

        # Padding mask: True = ignore (PyTorch MHA convention)
        pad_mask = (pile_obs.abs().sum(dim=-1) == 0)        # (B, 52)
        all_padded = pad_mask.all(dim=-1)                   # (B,)

        # MHA produces NaN when all keys are masked for a row.
        # Unmask the first slot for fully-padded rows so MHA stays valid,
        # then zero out those rows afterward.
        safe_mask = pad_mask.clone()
        safe_mask[all_padded, 0] = False

        attn_out, weights = self.cross_attn(
            query, kv, kv, key_padding_mask=safe_mask,
        )                                                   # attn_out: (B, 1, E)

        result = attn_out.squeeze(1)                        # (B, E)
        if all_padded.any():
            result = result.masked_fill(all_padded.unsqueeze(-1), 0.0)

        return result, weights                              # (B, E), (B, 1, 52)
    
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, has_encoder=False,
                 encode_claimants=False, positional_embeddings=False):
        super().__init__()
        self.has_encoder = has_encoder
        combined_dim = obs_dim + hidden_dim if has_encoder else obs_dim

        self.actor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        claim_input_dim = 3 if encode_claimants else 2
        self.encoder = ClaimCrossAttention(
            obs_dim=obs_dim,
            claim_input_dim=claim_input_dim,
            embed_dim=hidden_dim,
            positional=positional_embeddings,
        )

    def _encode(self, obs, claims):
        """Run cross-attention and concatenate summary to obs."""
        pile_summary, _ = self.encoder(obs, claims)
        return torch.cat([obs, pile_summary], dim=-1)

    def forward(self, obs, claims=None):
        x = self._encode(obs, claims) if self.has_encoder and claims is not None else obs
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action(self, obs, action_mask, claims=None):
        x = self._encode(obs, claims) if self.has_encoder and claims is not None else obs
        logits = self.actor(x)
        logits = logits.masked_fill(action_mask == 0, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, logits