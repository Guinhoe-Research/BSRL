from configs import EnvironmentConfig

class Agent:
    def __init__(self, name):
        self.name = name

    def act(self, state):
        print(f"Agent '{self.name}' is acting based on state '{state}'.")
        return "action"

class AgentPrivateState:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.hand = [0] * 13  # Cards in hand

    def update_hand(self, card):
        self.hand[card - 1] += 1
        print(f"Agent '{self.agent_id}' updated hand with card '{card}'. Current hand: {self.hand}")

    def get_num_cards(self):
        return sum(self.hand)

    def get_hand(self):
        return self.hand
    
class Environment:
    def __init__(self, config: EnvironmentConfig):
        self.deck = []
        for _ in range(4):  # 4 suits
            for rank in range(1, 14):
                self.deck.append(rank)   
        
        self.agent_ids = [f"agent_{i}" for i in range(config.num_agents)]
        self.agent_private_states = {agent_id: AgentPrivateState(agent_id) for agent_id in self.agent_ids}
        
        # Distribute cards to agents
        while self.deck:
            for agent_id in self.agent_ids:
                if self.deck:
                    card = self.deck.pop()
                    self.agent_private_states[agent_id].update_hand(card)

        self.current_required_rank = 1
        self.turn_index = 0

        # pairing -> (rank, count)
        # discard_pile holds TRUE card information, not just the rank
        self.discard_pile: list[tuple[int, int]] = []

        # self.log holds the history of claims made by agents, including the rank and count claimed
        self.log: list[tuple[int, int]] = []

        # just knobs 
        # Open Card Count: Whether agents can see the number of cards in other agents' hands
        self.open_card_count = config.OPEN_CARD_COUNT


    def reset(self):
        self.deck = []
        for _ in range(4):  # 4 suits
            for rank in range(1, 14):
                self.deck.append(rank)   
        
        self.agent_private_states = {agent_id: AgentPrivateState(agent_id) for agent_id in self.agent_ids}
        
        # Distribute cards to agents
        while self.deck:
            for agent_id in self.agent_ids:
                if self.deck:
                    card = self.deck.pop()
                    self.agent_private_states[agent_id].update_hand(card)

        self.current_required_rank = 1
        self.turn_index = 0
        self.discard_pile = []
        self.log = []

        print(f"Environment '{self.name}' has been reset.")

    def step(self, action):
        print(f"Action '{action}' taken in environment '{self.name}'.")
        return "next_state", 0, False  # next_state, reward, done
    
    def observe(self, agent):
        hand = self.agent_private_states[agent.name].get_hand()
        pile_size = len(self.discard_pile)
        last_claim = self.log[-1] if self.log else None
        turn_index = self.turn_index

        if self.open_card_count:
            # If open card count is enabled, include the number of cards in each agent's hand
            card_counts = {agent_id: self.agent_private_states[agent_id].get_num_cards() for agent_id in self.agent_ids}
        else:
            card_counts = None

        print(f"Observing environment '{self.name}' from agent '{agent.name}'. Hand: {hand}, Pile Size: {pile_size}, Last Claim: {last_claim}, Turn Index: {turn_index}, Card Counts: {card_counts}")
        
        return {
            "hand": hand,
            "pile_size": pile_size,
            "last_claim": last_claim,
            "turn_index": turn_index,
            "card_counts": card_counts
        }
    
    def render(self):
        print(f"Rendering environment '{self.name}'.")

    def close(self):
        print(f"Environment '{self.name}' has been closed.")