from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional


@dataclass
class EnvironmentConfig:
    num_agents: int
    SEE_CARD_COUNTS: bool
    RETURN_AS_TENSOR: bool = True

    ENCODE_CLAIMANTS: bool = True
    POSITIONAL_EMBEDDINGS: bool = True

    max_agents: Optional[int] = None  # pad card_counts to this size; defaults to num_agents


@dataclass
class RewardConfig:
    """All tunable reward values used by the environment.

    Adjust these to reshape agent incentives without touching environment code.
    """

    # --- claim shaping ---
    claim_shaping: float = 0.1          # per card played when finalising a claim

    # --- challenge resolution ---
    challenge_correct: float = 1.0      # challenger reward when claim was BS
    challenge_incorrect: float = -1.0   # challenger penalty when claim was truthful
    claimer_caught: float = -1.0        # claimer penalty when caught lying
    claimer_survived: float = 1.0       # claimer reward when truthful & challenged

    # --- new signals to break degenerate equilibria ---
    truthful_card_bonus: float = 0.0    # per card that matches the required rank
    successful_bluff: float = 0.0       # claimer bonus when BS claim goes unchallenged
    challenge_cost: float = 0.0         # flat cost subtracted every time an agent challenges

    # --- win ---
    win_bonus: float = 10.0


@dataclass
class TrainingConfig:
    """All tunable PPO / training-loop hyper-parameters."""

    # parallelism & collection
    num_envs: int = 32
    num_epochs: int = 50
    trajectory_window: int = 256
    batch_size: int = 32

    # PPO
    clip: float = 0.2
    vf_coef: float = 0.75
    ent_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95               # GAE lambda

    # optimiser
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    # model
    hidden_dim: int = 64

    # misc
    seed: int = 42


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _config_to_dict(cfg) -> dict:
    """Convert any config dataclass to a plain dict."""
    return asdict(cfg)


def _config_from_dict(cls, d: dict):
    """Instantiate a config dataclass from a dict, ignoring unknown keys."""
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})


def save_configs(
    path: str | Path,
    *,
    env: Optional[EnvironmentConfig] = None,
    reward: Optional[RewardConfig] = None,
    training: Optional[TrainingConfig] = None,
) -> None:
    """Save one or more config objects to a JSON file."""
    data: dict = {}
    if env is not None:
        data["environment"] = _config_to_dict(env)
    if reward is not None:
        data["reward"] = _config_to_dict(reward)
    if training is not None:
        data["training"] = _config_to_dict(training)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_configs(path: str | Path) -> dict:
    """Load configs from a JSON file. Returns a dict with keys
    ``environment``, ``reward``, ``training`` mapped to their dataclass
    instances (only those present in the file).
    """
    data = json.loads(Path(path).read_text())
    out: dict = {}
    if "environment" in data:
        out["environment"] = _config_from_dict(EnvironmentConfig, data["environment"])
    if "reward" in data:
        out["reward"] = _config_from_dict(RewardConfig, data["reward"])
    if "training" in data:
        out["training"] = _config_from_dict(TrainingConfig, data["training"])
    return out
