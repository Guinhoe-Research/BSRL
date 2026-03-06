from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Any


@dataclass
class ClaimAction:
    claim: Tuple[int, int]                 # (rank 1-13, count)
    actual_cards: List[Tuple[int, int]]    # list of (rank, 1) per card actually played


@dataclass
class ChallengeAction:
    """Current agent (agent_selection) challenges the last claim."""
    pass


@dataclass
class PassAction:
    """Current agent (agent_selection) passes during the challenge phase."""
    pass


@dataclass
class StartClaimAction:
    """Active player declares what they claim they are playing."""
    claim_rank: int   # 1-13
    claim_count: int  # 1-4


@dataclass
class SelectCardAction:
    """Active player selects a rank to play one card of."""
    rank_idx: int     # 0-12 (rank = rank_idx + 1)


@dataclass
class Event:
    kind: str
    agent_id: str
    payload: Any


class Phase(str, Enum):
    CLAIM = "CLAIM"       # active player chooses claim_rank/claim_count
    SELECT = "SELECT"     # active player selects k cards sequentially
    CHALLENGE = "CHALLENGE"  # non-active players respond with ChallengeAction or PassAction
