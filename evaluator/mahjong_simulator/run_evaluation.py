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

    parser.add_argument("mode", choices=["pairwise", "round_robin"],
                        help="Evaluation mode: 'pairwise' or 'round_robin'.")

    parser.add_argument("--agents", type=parse_agent_config, nargs='+',
                        help="List of agent configurations. "
                             "Each agent as a string: 'name=BotName,path=path/to/bot'. "
                             "Example: --agents name=RLBot,path=agent_trainer name=RuleBot,path=base_bot. "
                             "Required if --bot_list_file is not used in round_robin mode. "
                             "Disallowed if --bot_list_file is used in round_robin mode.")

    parser.add_argument("--bot_list_file", type=str,
                        help="Path to a file containing a list of bot configurations for round-robin mode "
                             "(one 'name=BotName,path=path/to/bot' per line). "
                             "If provided for round_robin, --agents should not be used. Not applicable for pairwise mode.")

    # Pairwise specific arguments
    parser.add_argument("--agent_a_name", type=str,
                        help="Name of Agent A for pairwise mode (must match a name in --agents).")
    parser.add_argument("--agent_b_name", type=str,
                        help="Name of Agent B for pairwise mode (must match a name in --agents).")

    # Common game number arguments
    parser.add_argument("--num_games", type=int, default=3,
                        help="Number of games per setup (for pairwise) or per matchup (for round_robin). Default: 3")

    args = parser.parse_args()

    eval_manager = EvaluationManager(run_game_func=run_game)
    all_agent_configs: List[AgentConfig] = []

    if args.mode == "round_robin":
        if args.bot_list_file:
            if args.agents:
                parser.error("--agents cannot be used with --bot_list_file in round_robin mode.")
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
                 parser.error(f"No valid agent configurations found in '{args.bot_list_file}'.")
        elif args.agents:
            all_agent_configs = args.agents
        else:
            parser.error("For round_robin mode, either --agents or --bot_list_file must be provided.")
    elif args.mode == "pairwise":
        if args.bot_list_file:
            parser.error("--bot_list_file is not applicable for pairwise mode.")
        if not args.agents:
            parser.error("For pairwise mode, --agents argument is required.")
        all_agent_configs = args.agents # In pairwise, agents are always from --agents

    if not all_agent_configs: # Should be caught by mode-specific checks above, but as a safeguard
        parser.error("No agent configurations provided.")

    agent_names_list = [agent.get("name", "") for agent in all_agent_configs]
    if "" in agent_names_list:
        print("Error: One or more agents ended up with an empty name. This should not happen.")
        return

    name_counts = Counter(agent_names_list)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        print(f"Error: Duplicate agent names found: {', '.join(duplicates)}. Please provide unique names for each agent configuration.")
        return

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

        # This check is slightly redundant if names must be unique as per above,
        # but good if that check was relaxed. If names are unique, agent_a_name == agent_b_name implies it's the same agent.
        if args.agent_a_name == args.agent_b_name:
             print(f"Warning: Agent A and B are the same ('{args.agent_a_name}'). Comparing an agent against itself.")
             # Allow this, as it might be a way to test an agent's stability or performance in homogeneous environments.

        print(f"Running Pairwise Evaluation for {args.agent_a_name} vs {args.agent_b_name}")
        results = eval_manager.run_pairwise_battle(
            agent_A_config=agent_a_conf,
            agent_B_config=agent_b_conf,
            num_games_per_setup=args.num_games
        )

    elif args.mode == "round_robin":
        # The EvaluationManager's round_robin method has its own check for min 4 agents.
        # A check for len < 2 might be useful here if we don't want to even call it.
        if len(all_agent_configs) < 2:
            parser.error("Round-robin mode requires at least 2 agents to be defined in --agents for meaningful comparison, though 4 are needed for standard tables.")

        print(f"Running Round-Robin Evaluation for {len(all_agent_configs)} agents.")
        results = eval_manager.run_round_robin_tournament(
            list_of_agent_configs=all_agent_configs,
            num_games_per_matchup=args.num_games
        )

    print("\n--- Evaluation Complete ---")
    if results.get("error"): # Check if the results dictionary itself contains an error key
        print(f"Evaluation finished with an error: {results['error']}")

    # Optional: Pretty print the full results dictionary
    # print("\nFull Results:")
    # print(json.dumps(results, indent=2))


if __name__ == "__main__":
    # Example command lines (ensure paths are correct relative to execution directory, typically project root):
    # To run from project root (e.g., .../MahjongRim):
    # python mahjong_simulator/run_evaluation.py round_robin --agents name=Bot1,path=bots/base_bot name=Bot2,path=bots/base_bot name=Bot3,path=bots/base_bot name=Bot4,path=bots/base_bot --num_games 1
    # Example with the moved agent_trainer:
    # python mahjong_simulator/run_evaluation.py round_robin --agents name=TBot1,path=bots/agent_trainer name=TBot2,path=bots/agent_trainer name=Base1,path=bots/base_bot name=Base2,path=bots/base_bot --num_games 1
    # Example using the agent_trainer at the project root:
    # python mahjong_simulator/run_evaluation.py pairwise --agents name=A1,path=bots/base_bot name=B1,path=agent_trainer --agent_a_name A1 --agent_b_name B1 --num_games 1
    #
    # Paths for agents are relative to the CWD of the run_evaluation.py script (usually project root).
    # The Agent class in game_utils.py uses this path as the CWD for the bot's subprocess.
    # So, if run_evaluation.py is run from project root:
    #   - For a bot in bots/base_bot, use path="bots/base_bot".
    #   - For a bot in bots/agent_trainer, use path="bots/agent_trainer".
    #   - For the agent_trainer at the project root, use path="agent_trainer".
    main()
