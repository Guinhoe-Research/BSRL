@dataclass
class EnvironmentConfig:
    num_agents: int
    SEE_CARD_COUNTS: bool
    RETURN_AS_TENSOR: bool = True