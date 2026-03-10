import random

import torch
import pytest

from core.environment import Environment
from core.configs import EnvironmentConfig

ACTION_DIM = 19

# Action index constants
CLAIM_1, CLAIM_2, CLAIM_3, CLAIM_4 = 0, 1, 2, 3
SELECT_RANK_1, SELECT_RANK_2, SELECT_RANK_3 = 4, 5, 6
CHALLENGE, PASS = 17, 18


@pytest.fixture
def env():
    cfg = EnvironmentConfig(num_agents=2, SEE_CARD_COUNTS=True)
    return Environment(cfg)


def seed_and_reset(env):
    random.seed(42)
    return env.reset()


def play_claim_and_select(env, claim_idx, *select_idxs):
    """Play a CLAIM action followed by SELECT actions, return the last obs."""
    env.step(claim_idx)
    obs = None
    for idx in select_idxs:
        obs, *_ = env.step(idx)
    return obs


# ---- Reset ----

def test_reset_shapes_and_values(env):
    obs, _ = seed_and_reset(env)

    assert obs["action_mask"].shape == (ACTION_DIM,)
    assert obs["hand_counts"].shape == (13,)
    assert torch.equal(obs["hand_counts"], torch.tensor([4, 4, 1, 2, 2, 2, 2, 1, 0, 1, 2, 3, 2]))
    assert torch.equal(obs["phase"], torch.tensor([1, 0, 0]))
    assert torch.equal(obs["round"], torch.tensor([0]))


# ---- CLAIM phase ----

def test_claim_phase(env):
    seed_and_reset(env)
    obs, *_ = env.step(CLAIM_2)

    assert obs["action_mask"].shape == (ACTION_DIM,)
    assert torch.equal(obs["action_mask"], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]))
    assert torch.equal(obs["phase"], torch.tensor([0, 1, 0]))


# ---- SELECT phase ----

def test_select_phase(env):
    seed_and_reset(env)
    env.step(CLAIM_2)

    # First card selection — still in SELECT, hand unchanged until all selected
    obs, *_ = env.step(SELECT_RANK_1)
    assert obs["hand_counts"].shape == (13,)
    assert torch.equal(obs["hand_counts"], torch.tensor([4, 4, 1, 2, 2, 2, 2, 1, 0, 1, 2, 3, 2]))
    assert torch.equal(obs["phase"], torch.tensor([0, 1, 0]))

    # Second card selection — advances to CHALLENGE (obs is now agent_1's view)
    obs, *_ = env.step(SELECT_RANK_2)
    assert torch.equal(obs["hand_counts"], torch.tensor([0, 0, 3, 2, 2, 2, 2, 3, 4, 3, 2, 1, 2]))
    assert torch.equal(obs["phase"], torch.tensor([0, 0, 1]))
    assert obs["agent_selection"] == "agent_1"


# ---- PASS action ----

def test_pass_advances_round(env):
    seed_and_reset(env)
    # agent_0: claim 2 of rank 1, play two rank-1 cards (truthful)
    play_claim_and_select(env, CLAIM_2, SELECT_RANK_1, SELECT_RANK_1)

    # agent_1 passes
    obs, _, terminated, *_ = env.step(PASS)

    assert torch.equal(obs["phase"], torch.tensor([1, 0, 0]))
    assert torch.equal(obs["round"], torch.tensor([1]))
    assert obs["agent_selection"] == "agent_1"
    assert torch.equal(obs["current_claim_rank"], torch.tensor([2]))
    assert torch.equal(obs["pile_size"], torch.tensor([2]))
    assert not terminated


# ---- CHALLENGE action (untruthful claim — challenger wins) ----

def test_challenge_untruthful_claim(env):
    seed_and_reset(env)
    # Round 0: agent_0 claims truthfully, agent_1 passes
    play_claim_and_select(env, CLAIM_2, SELECT_RANK_1, SELECT_RANK_1)
    env.step(PASS)

    # Round 1: agent_1 claims 1 of rank 2, but plays rank 3 (lying)
    play_claim_and_select(env, CLAIM_1, SELECT_RANK_3)

    a1_cards_before = env.agent_private_states["agent_1"].num_cards()

    # agent_0 challenges — should win (claim was untruthful)
    obs, reward, terminated, *_ = env.step(CHALLENGE)

    assert reward == 1.0  # challenger rewarded
    assert torch.equal(obs["phase"], torch.tensor([1, 0, 0]))
    assert torch.equal(obs["round"], torch.tensor([2]))
    assert torch.equal(obs["pile_size"], torch.tensor([0]))  # pile cleared
    assert not terminated

    # Liar (agent_1) picks up the pile
    a1_cards_after = env.agent_private_states["agent_1"].num_cards()
    assert a1_cards_after > a1_cards_before


# ---- CHALLENGE action (truthful claim — challenger loses) ----

def test_challenge_truthful_claim(env):
    seed_and_reset(env)
    # Round 0: pass through
    play_claim_and_select(env, CLAIM_2, SELECT_RANK_1, SELECT_RANK_1)
    env.step(PASS)
    # Round 1: untruthful, challenged
    play_claim_and_select(env, CLAIM_1, SELECT_RANK_3)
    env.step(CHALLENGE)

    # Round 2: agent_0 claims 1 of rank 3, plays rank 3 (truthful)
    play_claim_and_select(env, CLAIM_1, SELECT_RANK_3)

    a1_cards_before = env.agent_private_states["agent_1"].num_cards()

    # agent_1 challenges — should lose (claim was truthful)
    obs, reward, terminated, *_ = env.step(CHALLENGE)

    assert reward == -2.0  # cumulative: +1 from round 1, -1 from round 2
    assert torch.equal(obs["phase"], torch.tensor([1, 0, 0]))
    assert torch.equal(obs["round"], torch.tensor([3]))
    assert torch.equal(obs["pile_size"], torch.tensor([0]))
    assert not terminated

    # Challenger (agent_1) picks up the pile
    a1_cards_after = env.agent_private_states["agent_1"].num_cards()
    assert a1_cards_after > a1_cards_before


# ---- Round counter ----

def test_round_increments_each_turn(env):
    seed_and_reset(env)

    for expected_round in range(3):
        assert env.round == expected_round
        # Each agent claims 1 card, opponent passes
        play_claim_and_select(env, CLAIM_1, SELECT_RANK_1 if env.agent_private_states[env.agent_selection].hand_counts[0] > 0 else SELECT_RANK_3)
        env.step(PASS)

    assert env.round == 3


# ---- Invalid actions ----

def test_invalid_action_in_claim_phase(env):
    seed_and_reset(env)
    with pytest.raises(ValueError):
        env.step(CHALLENGE)  # can't challenge during CLAIM


def test_select_card_not_in_hand(env):
    seed_and_reset(env)
    env.step(CLAIM_1)
    # rank 9 (idx 8, action 12) — agent_0 has 0 cards of rank 9
    with pytest.raises(ValueError):
        env.step(12)
