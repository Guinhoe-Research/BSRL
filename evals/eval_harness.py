import numpy as np
import torch.nn as nn
import torch

import sys
from pathlib import Path

# Ensure parent project folder is importable in this notebook kernel
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
	sys.path.append(str(project_root))
     
from core.environment import Environment
from core.configs import EnvironmentConfig
from ppo.actor_critic_model import ActorCritic

torch.manual_seed(505)

def summarizer(action: int, agent_id: str, current_claim_rank: int) -> str:
    rank_names = {1:"A", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"10", 11:"J", 12:"Q", 13:"K"}
    if 0 <= action <= 3:
        count = action + 1
        return f"[CLAIM] {agent_id} CLAIMED {count} cards of rank {rank_names[current_claim_rank]}"
    elif 4 <= action <= 16:
        rank = action - 4 + 1
        return f"[SELECT] {agent_id} selected card of rank {rank_names[rank]}"
    elif action == 17:
        return f"[CHALLENGE] {agent_id} CHALLENGED"
    elif action == 18:
        return f"[PASS] {agent_id} PASSED"
    else:
        return f"[UNKNOWN] {agent_id} took unknown action {action}"

def run_one_game(config: EnvironmentConfig = EnvironmentConfig(num_agents=2, SEE_CARD_COUNTS=True), model_path="../outputs/ppo_model_100_32.pth"):
    cfg = config
    env = Environment(cfg)

    observation, info = env.reset()
    # print(observation)

    model = ActorCritic(obs_dim=22, act_dim=19)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()


    global_round_index = 1
    index = 0

    agent_ids = set([f'agent_{i}' for i in range(cfg.num_agents)])
    agent_ids.add("GAME_STATE")
    game_eval_data = []
    current_round_observations = {agent_id: [] for agent_id in agent_ids}

    while True or global_round_index < 35:
        if global_round_index != env._get_round():
            # New round, reset current round observations
            current_round_observations["GAME_STATE"].append({
                "round_index": global_round_index,
                "discard_pile": env._get_discard_pile(),
            })
            game_eval_data.append(current_round_observations)
            current_round_observations = {agent_id: [] for agent_id in agent_ids}
            global_round_index = env._get_round()

        state = observation["observation"]
        action_mask = observation["action_mask"]
        claim_sequence = observation["claim_seq"]
        active_agent_id = observation["active_agent"]
        discard_pile_size = observation["discard_pile_size"]

        with torch.no_grad():
            action, log_prob, logits = model.get_action(state, action_mask, claim_sequence)

        game_state_data = {
            "state": state.tolist(),
            "action_mask": action_mask.tolist(),
            "claim_sequence": claim_sequence.tolist(),
            "discard_pile_size": discard_pile_size,
            "action": action.item(),
            "log_prob": log_prob.item(),
            "logits": logits.tolist(),
            "summary": summarizer(action.item(), active_agent_id, env._current_claim_rank)
        }
        current_round_observations[active_agent_id].append(game_state_data)

        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation, reward, terminated, truncated, info)
        if terminated or truncated:
            break
        index += 1
    return game_eval_data

def main():
    print("Evaluation script")
    config = EnvironmentConfig(num_agents=2, SEE_CARD_COUNTS=True)

    eval_data = run_one_game(config)
    print("Evaluation complete. Sample data:")
    for round_data in eval_data[:5]:  # Print first 5 rounds of data
        for agent_id, observations in round_data.items():
            print(f"Agent: {agent_id}")
            for obs in observations:
                if "summary" in obs:
                    print(obs["summary"])
                else:
                    print(obs)
            print("-" * 20)

    with open("eval_output.json", "w") as f:
        import json
        json.dump(eval_data, f, indent=2)
    print("Evaluation data saved to eval_output.json")


if __name__ == "__main__":
    main()