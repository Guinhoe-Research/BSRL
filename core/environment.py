from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
from collections import deque
import random
import torch

from core.configs import EnvironmentConfig
from core.models import (
    ClaimAction, ChallengeAction, PassAction,
    StartClaimAction, SelectCardAction, Event, Phase,
)


# -----------------------------
# Agent + private state
# -----------------------------
class Agent:
    def __init__(self, name: str):
        self.name = name

    def act(self, obs):
        raise NotImplementedError


class AgentPrivateState:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.hand_counts = [0] * 13  # counts per rank: index 0 => rank=1

    def add_cards(self, rank: int, count: int = 1) -> None:
        assert 1 <= rank <= 13
        self.hand_counts[rank - 1] += count

    def remove_cards(self, ranks: List[int]) -> None:
        tmp = self.hand_counts[:]
        for r in ranks:
            if tmp[r - 1] <= 0:
                raise ValueError(f"Agent {self.agent_id} does not have rank {r} to remove.")
            tmp[r - 1] -= 1
        self.hand_counts = tmp

    def num_cards(self) -> int:
        return sum(self.hand_counts)

    def get_hand_counts(self) -> List[int]:
        return self.hand_counts[:]


# -----------------------------
# AEC Environment
# -----------------------------
class Environment:
    """
    PettingZoo-style AEC environment for the BS card game.

    Each call to step(action) corresponds to self.agent_selection taking one
    action. After processing, agent_selection advances to whoever acts next.

    Turn structure
    --------------
    1. CLAIM   — active agent submits StartClaimAction
    2. SELECT  — active agent submits SelectCardAction (one per card to play)
    3. CHALLENGE — every other agent submits ChallengeAction or PassAction,
                   in turn order starting from the player after the active agent
    → challenges are resolved → turn advances → repeat
    """

    metadata = {"name": "bs_env_v0"}

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.possible_agents: List[str] = [f"agent_{i}" for i in range(config.num_agents)]
        self.see_card_counts = config.SEE_CARD_COUNTS
        self.return_as_tensor = config.RETURN_AS_TENSOR
        self._init_state()

    # ---- Initialisation ----

    def _init_state(self) -> None:
        self.agents: List[str] = self.possible_agents[:]
        self.agent_private_states: Dict[str, AgentPrivateState] = {
            aid: AgentPrivateState(aid) for aid in self.agents
        }

        # AEC bookkeeping
        self.rewards: Dict[str, float] = {aid: 0.0 for aid in self.agents}
        self._cumulative_rewards: Dict[str, float] = {aid: 0.0 for aid in self.agents}
        self.terminations: Dict[str, bool] = {aid: False for aid in self.agents}
        self.truncations: Dict[str, bool] = {aid: False for aid in self.agents}
        self.infos: Dict[str, dict] = {aid: {} for aid in self.agents}

        # Turn / phase state
        self._turn_idx: int = 0          # index of active player into self.agents
        self.round: int = 0              # complete turn cycles (CLAIM→SELECT→CHALLENGE→resolve)
        self.phase: Phase = Phase.CLAIM
        self._challenge_queue: deque[str] = deque()   # non-active agents yet to respond
        self._challengers: List[str] = []        # agents that chose to challenge

        # Piles / logs
        self.discard_pile: List[Tuple[int, int]] = []  # (rank, 1) per card
        self.claim_log: List[Event] = []
        self.events_log: List[Event] = []
        self._current_claim_rank: int = 1  # cycles 1..13

        # Pending card-selection state (active during SELECT)
        self._pending_claim_rank: Optional[int] = None
        self._pending_claim_count: Optional[int] = None
        self._pending_selected_ranks: List[int] = []
        self._pending_remaining_counts: Optional[List[int]] = None

        self._deal()
        self.agent_selection: str = self.active_agent()

    def _deal(self) -> None:
        deck = [rank for _ in range(4) for rank in range(1, 14)]
        random.shuffle(deck)
        while deck:
            for aid in self.agents:
                if not deck:
                    break
                self.agent_private_states[aid].add_cards(deck.pop(), 1)

    def reset(self) -> Tuple[Dict[str, Any], dict]:
        """Reset the environment and return (obs, info) for the first agent."""
        self._init_state()
        obs = self.observe(self.agent_selection)
        return obs, self.infos[self.agent_selection]


    # ---- AEC helpers ----

    def active_agent(self) -> str:
        return self.agents[self._turn_idx % len(self.agents)]

    def _advance_turn(self) -> None:
        self._turn_idx = (self._turn_idx + 1) % len(self.agents)
        self._current_claim_rank = (self._current_claim_rank % 13) + 1
        self.round += 1
        self.phase = Phase.CLAIM
        self._challenge_queue = deque()
        self._challengers = []
        self._clear_pending()
        self.agent_selection = self.active_agent()

    def _clear_pending(self) -> None:
        self._pending_claim_rank = None
        self._pending_claim_count = None
        self._pending_selected_ranks = []
        self._pending_remaining_counts = None

    # ---- PettingZoo AEC interface ----

    def last(self) -> Tuple[Any, float, bool, bool, dict]:
        """Returns (obs, cumulative_reward, terminated, truncated, info) for agent_selection."""
        a = self.agent_selection
        return (
            self.observe(a),
            self._cumulative_rewards[a],
            self.terminations[a],
            self.truncations[a],
            self.infos[a],
        )

    def _int_to_action(self, idx: int):
        """Convert a discrete action index to the appropriate action model.

        Layout: [claim_count×4 | select_rank×13 | challenge/pass×2]
        """
        if 0 <= idx <= 3:
            return StartClaimAction(claim_count=idx + 1)
        elif 4 <= idx <= 16:
            return SelectCardAction(rank_idx=idx - 4)
        elif idx == 17:
            return ChallengeAction()
        elif idx == 18:
            return PassAction()
        else:
            raise ValueError(f"Invalid action index: {idx}")

    def tensor_to_action(self, action: torch.Tensor):
        """Convert a one-hot or scalar index tensor (size 19) to a discrete action index."""
        if action.dim() == 0:
            return int(action.item())
        return int(action.argmax().item())

    def step(self, action: int | torch.Tensor) -> Tuple[Any, float, bool, bool, dict]:
        """
        Process one action from self.agent_selection, then advance agent_selection.
        Accepts a discrete integer action index (Gymnasium-style) or a tensor
        (one-hot or scalar index), which is converted via tensor_to_action.
        Returns (obs, reward, terminated, truncated, info) for the next agent.
        """
        if isinstance(action, torch.Tensor):
            action = self.tensor_to_action(action)
        action = self._int_to_action(action)

        # Reset per-step rewards; cumulative rewards accumulate across the episode.
        self.rewards = {aid: 0.0 for aid in self.agents}

        agent = self.agent_selection

        # ------------------------------------------------------------------ #
        # CLAIM — only the active player acts                                 #
        # ------------------------------------------------------------------ #
        if self.phase == Phase.CLAIM:
            assert agent == self.active_agent(), (
                f"CLAIM phase expects {self.active_agent()}, got {agent}"
            )
            if not isinstance(action, StartClaimAction):
                raise ValueError(f"Expected StartClaimAction in CLAIM phase, got {type(action)}")
            if not (1 <= action.claim_count <= 4):
                raise ValueError("claim_count must be 1..4")
            hand_size = self.agent_private_states[agent].num_cards()
            if action.claim_count > hand_size:
                raise ValueError(
                    f"claim_count {action.claim_count} exceeds hand size {hand_size}"
                )

            self._pending_claim_rank = self._current_claim_rank
            self._pending_claim_count = action.claim_count
            self._pending_selected_ranks = []
            self._pending_remaining_counts = self.agent_private_states[agent].get_hand_counts()[:]
            self.phase = Phase.SELECT

            self.events_log.append(Event(kind="start_claim", agent_id=agent, payload=action))

        # ------------------------------------------------------------------ #
        # SELECT — active player picks cards one at a time                    #
        # ------------------------------------------------------------------ #
        elif self.phase == Phase.SELECT:
            assert agent == self.active_agent(), (
                f"SELECT phase expects {self.active_agent()}, got {agent}"
            )
            if not isinstance(action, SelectCardAction):
                raise ValueError(f"Expected SelectCardAction in SELECT phase, got {type(action)}")
            if self._pending_remaining_counts is None:
                raise RuntimeError("Pending claim state missing.")
            if not (0 <= action.rank_idx < 13):
                raise ValueError("rank_idx must be 0..12")

            rank = action.rank_idx + 1
            if self._pending_remaining_counts[action.rank_idx] <= 0:
                raise ValueError(f"No cards of rank {rank} remaining.")

            self._pending_remaining_counts[action.rank_idx] -= 1
            self._pending_selected_ranks.append(rank)
            self.events_log.append(Event(kind="select_card", agent_id=agent, payload=action))

            if len(self._pending_selected_ranks) == self._pending_claim_count:
                # Finalise the claim: remove cards from hand, add to discard pile
                self.agent_private_states[agent].remove_cards(self._pending_selected_ranks)
                actual_cards = [(r, 1) for r in self._pending_selected_ranks]
                self.discard_pile.extend(actual_cards)

                claim_event = Event(
                    kind="claim",
                    agent_id=agent,
                    payload=ClaimAction(
                        claim=(self._pending_claim_rank, self._pending_claim_count),
                        actual_cards=actual_cards,
                    ),
                )
                self.claim_log.append(claim_event)
                self.events_log.append(claim_event)

                # Shaping reward: small bonus for playing cards (reducing hand size)
                self.rewards[agent] += 0.1 * self._pending_claim_count

                # Transition to CHALLENGE — build queue of all other agents
                self.phase = Phase.CHALLENGE
                n = len(self.agents)
                self._challenge_queue = deque(
                    self.agents[(self._turn_idx + i) % n] for i in range(1, n)
                )
                self._challengers = []
                self.agent_selection = self._challenge_queue[0]
                self._accumulate_rewards()
                return self.last()  # agent_selection already updated

        # ------------------------------------------------------------------ #
        # CHALLENGE — each non-active agent either challenges or passes       #
        # ------------------------------------------------------------------ #
        elif self.phase == Phase.CHALLENGE:
            if agent not in self._challenge_queue:
                raise ValueError(f"{agent} is not in the challenge queue.")
            if not isinstance(action, (ChallengeAction, PassAction)):
                raise ValueError(
                    f"Expected ChallengeAction or PassAction in CHALLENGE phase, got {type(action)}"
                )

            if isinstance(action, ChallengeAction):
                self._challengers.append(agent)
                self.events_log.append(Event(kind="challenge", agent_id=agent, payload=action))
            else:
                self.events_log.append(Event(kind="pass", agent_id=agent, payload=action))

            self._challenge_queue.popleft()

            if self._challenge_queue:
                # More agents still need to respond
                self.agent_selection = self._challenge_queue[0]
            else:
                # All agents have responded — resolve then advance turn
                self._resolve_challenges()

                # Win check: did the claimer empty their hand?
                former_active = self.active_agent()
                if self.agent_private_states[former_active].num_cards() == 0:
                    self._handle_win(former_active)
                    self._accumulate_rewards()
                    return self.last()

                self._advance_turn()

        else:
            raise ValueError(f"Unknown phase: {self.phase}")

        self._accumulate_rewards()
        return self.last()


    # ---- Resolution ----

    def _resolve_challenges(self) -> None:
        """
        Resolve after all agents have responded.

        If no one challenged: the pile stays (grows into next turn).
        If anyone challenged:
          - Claim was truthful  → challengers were wrong; first challenger takes the
                                  pile; challengers get -1 reward, claimer gets +1.
          - Claim was untruthful → challengers were right; claimer takes the pile;
                                   challengers get +1 reward, claimer gets -1.
        """
        if not self._challengers:
            return  # pile stays, no rewards

        last_claim: ClaimAction = self.claim_log[-1].payload
        claimed_rank, claimed_count = last_claim.claim
        actual = last_claim.actual_cards
        truthful = (len(actual) == claimed_count) and all(r == claimed_rank for r, _ in actual)
        claimer_id = self.claim_log[-1].agent_id

        self.events_log.append(Event(
            kind="resolve_challenge",
            agent_id=claimer_id,
            payload={"truthful": truthful, "challengers": list(self._challengers)},
        ))

        if truthful:
            # Challengers were wrong — first challenger picks up the pile
            loser = self._challengers[0]
            for r, c in self.discard_pile:
                self.agent_private_states[loser].add_cards(r, c)
            for challenger in self._challengers:
                self.rewards[challenger] -= 1.0
            self.rewards[claimer_id] += 1.0
        else:
            # Challengers were right — claimer picks up the pile
            for r, c in self.discard_pile:
                self.agent_private_states[claimer_id].add_cards(r, c)
            for challenger in self._challengers:
                self.rewards[challenger] += 1.0
            self.rewards[claimer_id] -= 1.0

        self.discard_pile.clear()

    def _handle_win(self, winner_id: str) -> None:
        self.rewards[winner_id] += 10.0
        self.terminations = {aid: True for aid in self.agents}
        self.agent_selection = winner_id

    def _accumulate_rewards(self) -> None:
        for aid in self.agents:
            self._cumulative_rewards[aid] += self.rewards[aid]

    # ---- Observation ----

    def selection_action_mask(self) -> List[int]:
        """1 for each rank (0-indexed) the agent still has available this SELECT."""
        if self._pending_remaining_counts is None:
            return [0] * 13
        return [1 if c > 0 else 0 for c in self._pending_remaining_counts]

    def action_mask(self) -> torch.Tensor:
        """Mask of valid actions. Layout: [claim_count×4 | select_rank×13 | challenge×2]"""
        mask = torch.zeros(19, dtype=torch.int64)
        if self.phase == Phase.CLAIM:
            hand_size = self.agent_private_states[self.agent_selection].num_cards()
            max_claim = min(4, hand_size)
            mask[:max_claim] = 1
        elif self.phase == Phase.SELECT:
            mask[4:17] = torch.tensor(self.selection_action_mask(), dtype=torch.int64)
        elif self.phase == Phase.CHALLENGE:
            mask[17:] = 1
        return mask


    def observe(self, agent_id: str) -> Dict[str, Any]:
        """Build an observation for the given agent.

        When return_as_tensor is True, returns a Gymnasium-style dict:
            "observation" — flat float tensor the policy network consumes
            "action_mask" — int tensor of valid actions (size 19)
            + metadata keys for logging / debugging

        When return_as_tensor is False, returns a plain dict of Python objects.
        """
        priv = self.agent_private_states[agent_id]
        hand_counts = priv.get_hand_counts()
        last_claim: Optional[ClaimAction] = (
            self.claim_log[-1].payload if self.claim_log else None
        )
        pile_size = len(self.discard_pile)

        card_counts: Optional[Dict[str, int]] = None
        if self.see_card_counts:
            card_counts = {
                aid: self.agent_private_states[aid].num_cards() for aid in self.agents
            }

        if not self.return_as_tensor:
            return {
                "hand_counts": hand_counts,
                "phase": self.phase.value,
                "action_mask": self.action_mask(),
                "pile_size": pile_size,
                "last_claim": last_claim,
                "current_claim_rank": self._current_claim_rank,
                "agent_selection": self.agent_selection,
                "active_agent": self.active_agent(),
                "turn_index": self._turn_idx,
                "round": self.round,
                "card_counts": card_counts,
            }

        # ---------- tensor observation ----------

        # Phase one-hot [CLAIM, SELECT, CHALLENGE] — (3,)
        phase_vec = torch.zeros(3, dtype=torch.float32)
        phase_vec[{Phase.CLAIM: 0, Phase.SELECT: 1, Phase.CHALLENGE: 2}[self.phase]] = 1.0

        # Hand counts per rank (1-13), normalized by max per rank (4) — (13,)
        hand_vec = torch.tensor(hand_counts, dtype=torch.float32) / 4.0

        # Discard pile size normalized by deck size (52) — (1,)
        pile_vec = torch.tensor([pile_size / 52.0], dtype=torch.float32)

        # Claim history as (claimed_rank, claimed_count) pairs — (MAX_CLAIMS, 2) long tensor
        # Ranks 1-13, counts 1-4; row of [0, 0] = padding. Ordered chronologically.
        # Max claims = 52 (worst case: every claim is 1 card). Use len(claim_log) for valid length.
        MAX_CLAIMS = 52
        claim_seq = torch.zeros(MAX_CLAIMS, 2, dtype=torch.float32)
        for i, event in enumerate(self.claim_log):
            claim: ClaimAction = event.payload
            claim_seq[i, 0] = claim.claim[0] / 13.0  # claimed rank normalized
            claim_seq[i, 1] = claim.claim[1] / 4.0   # claimed count normalized

        # Current claim rank normalized to [0, 1] — (1,)
        claim_rank_vec = torch.tensor([self._current_claim_rank / 13.0], dtype=torch.float32)

        # Last claim rank normalized to [0, 1] — (1,); 0 if no prior claim
        # Last claim count normalized to [0, 1] — (1,); 0 if no prior claim
        last_rank_val = last_claim.claim[0] / 13.0 if last_claim is not None else 0.0
        last_count_val = last_claim.claim[1] / 4.0 if last_claim is not None else 0.0
        last_claim_vec = torch.tensor([last_rank_val, last_count_val], dtype=torch.float32)

        parts = [
            phase_vec,           # 3   — current game phase
            hand_vec,            # 13  — cards in this agent's hand
            pile_vec,            # 1   — number of cards in the discard pile
            claim_rank_vec,      # 1   — rank that must be claimed this turn
            last_claim_vec,      # 2   — rank and count of the most recent claim
        ]

        # Optional: other agents' card counts — (num_agents,)
        if card_counts is not None:
            parts.append(
                torch.tensor([card_counts[aid] / 52.0 for aid in self.agents], dtype=torch.float32)
            )

        observation = torch.cat(parts)

        return {
            # --- policy input ---
            "observation": observation,
            "action_mask": self.action_mask(),
            # Claim history for embedding/LSTM — (52, 2) float tensor
            # Each row: [claimed_rank/13, claimed_count/4]. [0,0] = padding.
            "claim_seq": claim_seq,
            # --- game metadata ---
            "agent_selection": self.agent_selection,
            "active_agent": self.active_agent(),
            "turn_index": self._turn_idx,
            "round": self.round,
            "discard_pile_size": pile_size,
        }
