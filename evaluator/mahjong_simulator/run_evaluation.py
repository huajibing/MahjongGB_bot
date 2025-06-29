# mahjong_simulator/run_evaluation.py

import argparse
import json # For potentially parsing complex agent configs if needed, or simple string split
from .evaluation_manager import EvaluationManager, AgentConfig # Assuming it's in the same directory
from .main import run_game # The actual game runner function, assuming it's in the same directory
from typing import List, Dict, Any
from collections import Counter # For checking duplicate names

def parse_agent_config(agent_str: str) -> AgentConfig:
    """
    Parses an agent configuration string like "name=BotName,path=path/to/bot".
    Returns an AgentConfig dictionary.
    Expects comma-separated key=value pairs.
    """
    config: AgentConfig = {}
    try:
        parts = agent_str.split(',')
        for part in parts:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            if key not in ["name", "path"]: # Add other valid keys if any in future
                raise ValueError(f"Invalid key '{key}' in agent string.")
            config[key] = value
        if "path" not in config: # Path is mandatory
            raise ValueError("Agent string must contain 'path'.")
        if "name" not in config: # If name is not provided, use the last part of the path
            config["name"] = config["path"].split('/')[-1].split('\\')[-1] or "UnnamedAgent"
            if not config["name"]: # Handle cases like path="." or path="/"
                 config["name"] = "UnnamedAgent"
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid agent string format '{agent_str}'. Error: {e}. Expected 'name=BotName,path=path/to/bot'.")
    return config

def main():
    parser = argparse.ArgumentParser(description="Mahjong Agent Evaluation Framework")

    parser.add_argument("mode", choices=["pairwise", "round_robin", "duplicate"],
                        help="Evaluation mode: 'pairwise', 'round_robin', or 'duplicate' for duplicate-style tournament.")

    parser.add_argument("--agents", type=parse_agent_config, nargs='+', required=True,
                        help="List of agent configurations. "
                             "Each agent as a string: 'name=BotName,path=path/to/bot'. "
                             "Example: --agents name=RLBot,path=agent_trainer name=RuleBot,path=base_bot. "
                             "Required for all modes.")

    parser.add_argument("--bot_list_file", type=str,
                        help="Path to a file containing a list of bot configurations "
                             "(one 'name=BotName,path=path/to/bot' per line). "
                             "Can be used as an alternative to --agents. If both are provided, they are combined.")

    # Pairwise specific arguments
    parser.add_argument("--agent_a_name", type=str,
                        help="Name of Agent A for pairwise mode (must match a name in --agents).")
    parser.add_argument("--agent_b_name", type=str,
                        help="Name of Agent B for pairwise mode (must match a name in --agents).")

    # Common game number arguments
    parser.add_argument("--num_games", type=int, default=4,
                        help="Number of games. For pairwise/round_robin, it's games per setup/matchup. "
                             "For duplicate, it's the number of seating permutations per wall (e.g., 4 or 24). Default: 4")

    args = parser.parse_args()

    eval_manager = EvaluationManager(run_game_func=run_game)
    all_agent_configs: List[AgentConfig] = args.agents if args.agents else []

    # Handle bot list file
    if args.bot_list_file:
        try:
            with open(args.bot_list_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'): # Skip empty lines and comments
                        continue
                    try:
                        all_agent_configs.append(parse_agent_config(line))
                    except argparse.ArgumentTypeError as e: # Catch errors from parse_agent_config
                        parser.error(f"Error parsing line {line_num} in '{args.bot_list_file}': {e}")
        except FileNotFoundError:
            parser.error(f"Bot list file not found: {args.bot_list_file}")
        except Exception as e: # Catch other potential file reading errors
            parser.error(f"Error reading bot list file '{args.bot_list_file}': {e}")

    if not all_agent_configs:
        parser.error("No agent configurations provided via --agents or --bot_list_file.")

    agent_names_list = [agent.get("name", "") for agent in all_agent_configs]
    if "" in agent_names_list:
        parser.error("One or more agents ended up with an empty name. This should not happen.")

    name_counts = Counter(agent_names_list)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        parser.error(f"Duplicate agent names found: {', '.join(duplicates)}. Please provide unique names for each agent configuration.")

    results: Dict[str, Any] = {}

    if args.mode == "pairwise":
        if not args.agent_a_name or not args.agent_b_name:
            parser.error("For pairwise mode, --agent_a_name and --agent_b_name are required.")

        agent_a_conf = next((conf for conf in all_agent_configs if conf.get("name") == args.agent_a_name), None)
        agent_b_conf = next((conf for conf in all_agent_configs if conf.get("name") == args.agent_b_name), None)

        if not agent_a_conf:
            parser.error(f"Agent with name '{args.agent_a_name}' not found in --agents list.")
        if not agent_b_conf:
            parser.error(f"Agent with name '{args.agent_b_name}' not found in --agents list.")
        
        if args.agent_a_name == args.agent_b_name:
             print(f"Warning: Agent A and B are the same ('{args.agent_a_name}'). Comparing an agent against itself.")

        print(f"Running Pairwise Evaluation for {args.agent_a_name} vs {args.agent_b_name}")
        results = eval_manager.run_pairwise_battle(
            agent_A_config=agent_a_conf,
            agent_B_config=agent_b_conf,
            num_games_per_setup=args.num_games
        )

    elif args.mode == "round_robin":
        if len(all_agent_configs) < 4:
            parser.error("Round-robin mode requires at least 4 agents.")

        print(f"Running Round-Robin Evaluation for {len(all_agent_configs)} agents.")
        results = eval_manager.run_round_robin_tournament(
            list_of_agent_configs=all_agent_configs,
            num_games_per_matchup=args.num_games
        )

    elif args.mode == "duplicate":
        if len(all_agent_configs) % 4 != 0:
            parser.error("Duplicate mode requires the number of agents to be a multiple of 4.")
        
        print(f"Running Duplicate (Revival) Tournament for {len(all_agent_configs)} agents.")
        results = eval_manager.run_duplicate_tournament(
            list_of_agent_configs=all_agent_configs,
            num_seatings_per_wall=args.num_games
        )

    print("\n--- Evaluation Complete ---")
    if results.get("error"):
        print(f"Evaluation finished with an error: {results['error']}")

if __name__ == "__main__":
    # Example command line for duplicate mode:
    # python mahjong_simulator/run_evaluation.py duplicate --agents name=A,path=bots/base_bot name=B,path=bots/base_bot name=C,path=bots/base_bot name=D,path=bots/base_bot --num_games 4
    main()