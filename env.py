from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
import random
import torch

from configs import EnvironmentConfig
from models import (
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
        self.phase: Phase = Phase.CLAIM
        self._challenge_queue: List[str] = []   # non-active agents yet to respond
        self._challengers: List[str] = []        # agents that chose to challenge

        # Piles / logs
        self.discard_pile: List[Tuple[int, int]] = []  # (rank, 1) per card
        self.claim_log: List[Event] = []
        self.events_log: List[Event] = []

        # Pending card-selection state (active during SELECT)
        self._pending_claim_rank: Optional[int] = None
        self._pending_claim_count: Optional[int] = None
        self._pending_selected_ranks: List[int] = []
        self._pending_slots: Optional[List[int]] = None

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

    def reset(self) -> None:
        self._init_state()

    # ---- AEC helpers ----

    def active_agent(self) -> str:
        return self.agents[self._turn_idx % len(self.agents)]

    def _advance_turn(self) -> None:
        self._turn_idx = (self._turn_idx + 1) % len(self.agents)
        self.phase = Phase.CLAIM
        self._challenge_queue = []
        self._challengers = []
        self._clear_pending()
        self.agent_selection = self.active_agent()

    def _clear_pending(self) -> None:
        self._pending_claim_rank = None
        self._pending_claim_count = None
        self._pending_selected_ranks = []
        self._pending_slots = None

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

    def step(self, action) -> None:
        """
        Process one action from self.agent_selection, then advance agent_selection.
        """
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
            if not (1 <= action.claim_rank <= 13):
                raise ValueError("claim_rank must be 1..13")
            if not (1 <= action.claim_count <= 4):
                raise ValueError("claim_count must be 1..4")

            self._pending_claim_rank = action.claim_rank
            self._pending_claim_count = action.claim_count
            self._pending_selected_ranks = []
            self._pending_slots = self.expanded_hand_slots(agent)
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
            if self._pending_slots is None:
                raise RuntimeError("Pending claim state missing.")
            if not (0 <= action.slot_idx < 13):
                raise ValueError("slot_idx must be 0..12")

            rank = self._pending_slots[action.slot_idx]
            if rank == -1:
                raise ValueError("Selected an empty slot.")

            self._pending_slots[action.slot_idx] = -1
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

                # Transition to CHALLENGE — build queue of all other agents
                self.phase = Phase.CHALLENGE
                n = len(self.agents)
                self._challenge_queue = [
                    self.agents[(self._turn_idx + i) % n] for i in range(1, n)
                ]
                self._challengers = []
                self.agent_selection = self._challenge_queue[0]
                self._accumulate_rewards()
                return  # agent_selection already updated

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

            self._challenge_queue.pop(0)

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
                    return

                self._advance_turn()

        else:
            raise ValueError(f"Unknown phase: {self.phase}")

        self._accumulate_rewards()

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

    def expanded_hand_slots(self, agent_id: str) -> List[int]:
        """Length-13 list of ranks (1..13) for the agent's hand, padded with -1."""
        counts = self.agent_private_states[agent_id].get_hand_counts()
        slots: List[int] = []
        for rank_idx, c in enumerate(counts):
            slots.extend([rank_idx + 1] * c)
        if len(slots) > 13:
            slots = slots[:13]
        slots.extend([-1] * (13 - len(slots)))
        return slots

    def selection_action_mask(self) -> List[int]:
        """1 for selectable slots (only valid during SELECT phase for the active player)."""
        if self.phase != Phase.SELECT or self._pending_slots is None:
            return [0] * 13
        return [1 if r != -1 else 0 for r in self._pending_slots]

    def observe(self, agent_id: str) -> Dict[str, Any]:
        hand_counts = self.agent_private_states[agent_id].get_hand_counts()
        expanded_slots = self.expanded_hand_slots(agent_id)
        last_claim = self.claim_log[-1].payload if self.claim_log else None
        pile_size = len(self.discard_pile)

        card_counts = None
        if self.see_card_counts:
            card_counts = {aid: self.agent_private_states[aid].num_cards() for aid in self.agents}

        obs = {
            "phase": self.phase.value,
            "agent_selection": self.agent_selection,
            "active_agent": self.active_agent(),
            "hand_counts": hand_counts,
            "hand_slots": expanded_slots,
            "selection_mask": self.selection_action_mask(),
            "pile_size": pile_size,
            "last_claim": last_claim,
            "turn_index": self._turn_idx,
            "card_counts": card_counts,
        }

        if self.return_as_tensor:
            obs_t: Dict[str, Any] = {
                "phase": obs["phase"],
                "agent_selection": obs["agent_selection"],
                "active_agent": obs["active_agent"],
                "hand_counts": torch.tensor(hand_counts, dtype=torch.int64),
                "hand_slots": torch.tensor(expanded_slots, dtype=torch.int64),
                "selection_mask": torch.tensor(obs["selection_mask"], dtype=torch.int64),
                "pile_size": torch.tensor([pile_size], dtype=torch.int64),
                "turn_index": torch.tensor([self._turn_idx], dtype=torch.int64),
            }
            if card_counts is not None:
                obs_t["card_counts"] = torch.tensor(
                    [card_counts[aid] for aid in self.agents], dtype=torch.int64
                )
            return obs_t

        return obs
