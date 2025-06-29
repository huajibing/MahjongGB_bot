# mahjong_simulator/evaluation_manager.py

import random
import itertools # For combinations and permutations
from typing import List, Dict, Tuple, Optional, Any
from .logging_utils import logger
from .game_utils import ALL_TILES

AgentConfig = Dict[str, str]

class EvaluationManager:
    def __init__(self, run_game_func: callable):
        """
        Initializes the EvaluationManager.
        :param run_game_func: A callable that matches the signature of the refactored
                              `run_game`.
        """
        self.run_game = run_game_func
        self.evaluation_results: Dict[str, Any] = {} # To store results

    def _get_agent_path(self, agent_config: AgentConfig) -> str:
        """Extracts the path from agent configuration."""
        path = agent_config.get("path")
        if not path:
            raise ValueError(f"Agent configuration missing 'path': {agent_config}")
        return path

    def _get_agent_name(self, agent_config: AgentConfig) -> str:
        """Extracts the name from agent configuration, defaults to path if name is missing."""
        return agent_config.get("name", agent_config.get("path", "UnknownAgent"))

    def run_duplicate_tournament(self,
                                 list_of_agent_configs: List[AgentConfig],
                                 num_seatings_per_wall: int = 4) -> Dict[str, Any]:
        
        num_agents = len(list_of_agent_configs)
        if num_agents % 4 != 0:
            msg = f"Duplicate tournament requires a number of agents that is a multiple of 4, but got {num_agents}."
            logger.error(msg)
            return {"error": msg}

        logger.info(f"Starting Duplicate Tournament for {num_agents} agents.")
        logger.info(f"Each sub-match will use {num_seatings_per_wall} seating permutations per wall.")

        # Group agents into tables of 4
        random.shuffle(list_of_agent_configs) # Shuffle to create random tables
        tables = [list_of_agent_configs[i:i+4] for i in range(0, num_agents, 4)]
        
        agent_names = [self._get_agent_name(c) for c in list_of_agent_configs]
        tournament_summary = {
            name: {"ranking_points": 0, "total_game_score": 0, "config": config}
            for name, config in zip(agent_names, list_of_agent_configs)
        }

        all_matches_details = []

        for table_idx, table_agents in enumerate(tables):
            table_agent_names = [self._get_agent_name(c) for c in table_agents]
            logger.info(f"\n--- Processing Table {table_idx + 1}/{len(tables)}: {', '.join(table_agent_names)} ---")

            # Generate 4 fixed tile walls for this table's match
            full_wall = list(ALL_TILES)
            random.shuffle(full_wall)
            master_walls = {
                seat_idx: full_wall[i:i+34] for seat_idx, i in enumerate(range(0, 136, 34))
            }

            match_details = {
                "table_id": table_idx + 1,
                "agents": table_agent_names,
                "sub_matches": []
            }
            
            # A match consists of 4 "sub-matches", one for each prevalent wind / master wall
            prevalent_winds = [0, 1, 2, 3] # E, S, W, N
            for sub_match_idx, prevalent_wind in enumerate(prevalent_winds):
                wall_id = sub_match_idx # Using the index as the wall identifier
                logger.info(f"  Starting Sub-Match {sub_match_idx + 1}/4 (Prevalent Wind: {['E','S','W','N'][prevalent_wind]})")

                # Scores for this specific wall/wind combination
                wall_aggregate_scores = {name: 0 for name in table_agent_names}
                
                # Get all 24 permutations of seatings
                all_permutations = list(itertools.permutations(table_agents))
                seatings_to_run = all_permutations[:num_seatings_per_wall]

                game_results_for_sub_match = []
                for game_idx, seating_tuple in enumerate(seatings_to_run):
                    seating_configs = list(seating_tuple)
                    seating_names = [self._get_agent_name(c) for c in seating_configs]
                    agent_paths = [self._get_agent_path(c) for c in seating_configs]
                    
                    logger.info(f"    Game {game_idx + 1}/{num_seatings_per_wall}. Seating (E->N): {', '.join(seating_names)}")

                    final_scores, _, error_message = self.run_game(
                        agent_paths=agent_paths,
                        prevalent_wind=prevalent_wind,
                        fixed_walls=master_walls
                    )
                    
                    game_results_for_sub_match.append({
                        "seating": seating_names,
                        "scores": final_scores,
                        "error": error_message
                    })

                    if final_scores:
                        for p_idx, score in final_scores.items():
                            agent_name = seating_names[p_idx]
                            wall_aggregate_scores[agent_name] += score

                # After all games for this wall, calculate ranking points
                sorted_by_agg_score = sorted(wall_aggregate_scores.items(), key=lambda item: item[1], reverse=True)
                
                ranking_points_map = {name: 0 for name in table_agent_names}
                rank_points_dist = [4, 3, 2, 1]
                for rank, (name, agg_score) in enumerate(sorted_by_agg_score):
                    points = rank_points_dist[rank]
                    ranking_points_map[name] = points
                    tournament_summary[name]["ranking_points"] += points
                    # Also accumulate the total game score for tie-breaking
                    tournament_summary[name]["total_game_score"] += agg_score
                
                logger.info(f"  Sub-Match {sub_match_idx + 1} Results:")
                for name, score in sorted_by_agg_score:
                    logger.info(f"    - {name:<20}: Aggregate Score = {score:>+5}, Ranking Points = {ranking_points_map[name]}")

                match_details["sub_matches"].append({
                    "prevalent_wind": prevalent_wind,
                    "wall_id": wall_id,
                    "aggregate_scores": wall_aggregate_scores,
                    "ranking_points": ranking_points_map,
                    "games": game_results_for_sub_match,
                })

            all_matches_details.append(match_details)

        # Final ranking across all agents
        final_ranking = sorted(
            tournament_summary.items(),
            key=lambda item: (item[1]["ranking_points"], item[1]["total_game_score"]),
            reverse=True
        )

        logger.info("\n--- Duplicate Tournament Final Ranking ---")
        for i, (name, data) in enumerate(final_ranking):
            logger.info(f"  {i+1:>2}. {name:<20} | Ranking Points: {data['ranking_points']:>3} | Total Game Score: {data['total_game_score']:>+7}")

        final_results = {
            "mode": "duplicate",
            "ranking": [
                {"rank": i + 1, "name": name, "ranking_points": data["ranking_points"], "total_game_score": data["total_game_score"]}
                for i, (name, data) in enumerate(final_ranking)
            ],
            "details": all_matches_details,
            "summary_by_agent": tournament_summary
        }
        self.evaluation_results["duplicate_tournament"] = final_results
        return final_results

    # ... (keep the existing run_pairwise_battle and run_round_robin_tournament methods)
    def run_pairwise_battle(self, agent_A_config: AgentConfig, agent_B_config: AgentConfig, num_games_per_setup: int = 3) -> Dict[str, Any]:
        # ... (implementation from the original file)
        agent_A_name = self._get_agent_name(agent_A_config)
        agent_B_name = self._get_agent_name(agent_B_config)
        agent_A_path = self._get_agent_path(agent_A_config)
        agent_B_path = self._get_agent_path(agent_B_config)
        logger.info(f"Starting pairwise battle: {agent_A_name} vs {agent_B_name}")
        setups_config = {
            "A_B_A_A": [agent_A_path, agent_B_path, agent_A_path, agent_A_path],
            "A_B_B_B": [agent_A_path, agent_B_path, agent_B_path, agent_B_path],
            "A_B_A_B": [agent_A_path, agent_B_path, agent_A_path, agent_B_path]
        }
        player_to_agent_type_map = {
            "A_B_A_A": {0: "A", 1: "B", 2: "A", 3: "A"},
            "A_B_B_B": {0: "A", 1: "B", 2: "B", 3: "B"},
            "A_B_A_B": {0: "A", 1: "B", 2: "A", 3: "B"}
        }
        battle_results: Dict[str, Any] = {
            "agent_A_config": agent_A_config,
            "agent_B_config": agent_B_config,
            "num_games_per_setup": num_games_per_setup,
            "setups_results": [],
            "summary": {
                agent_A_name: {"wins": 0, "total_score": 0, "games_played": 0, "win_rate": 0.0, "avg_score": 0.0},
                "errors": 0,
                "total_games_run": 0
            }
        }
        battle_results["summary"][agent_A_name] = {"wins": 0, "total_score": 0, "games_played": 0, "win_rate": 0.0, "avg_score": 0.0}
        if agent_A_name != agent_B_name:
            battle_results["summary"][agent_B_name] = {"wins": 0, "total_score": 0, "games_played": 0, "win_rate": 0.0, "avg_score": 0.0}
        for setup_name, agent_path_list in setups_config.items():
            setup_summary_details = {
                "setup_name": setup_name,
                "agent_paths": agent_path_list,
                "games": []
            }
            current_player_map = player_to_agent_type_map[setup_name]
            logger.info(f"  Running setup: {setup_name} for {num_games_per_setup} games...")
            for game_num in range(num_games_per_setup):
                battle_results["summary"]["total_games_run"] += 1
                logger.info(f"    Starting game {game_num + 1}/{num_games_per_setup} for setup {setup_name}...")
                final_scores, winner_index, error_message = self.run_game(agent_path_list)
                game_result_details = {
                    "game_number": game_num + 1,
                    "scores": final_scores,
                    "winner_index": winner_index,
                    "error": error_message,
                    "player_identities": current_player_map
                }
                setup_summary_details["games"].append(game_result_details)
                if error_message:
                    logger.error(f"      Game ended with error: {error_message}")
                    battle_results["summary"]["errors"] += 1
                    continue
                if final_scores is None:
                    logger.warning("      Game ended with no scores and no error. Skipping score processing.")
                    battle_results["summary"]["errors"] += 1
                    continue
                for player_idx, score in final_scores.items():
                    agent_type_at_player_idx = current_player_map.get(player_idx)
                    current_agent_name_for_stats = None
                    if agent_type_at_player_idx == "A": current_agent_name_for_stats = agent_A_name
                    elif agent_type_at_player_idx == "B": current_agent_name_for_stats = agent_B_name
                    if current_agent_name_for_stats:
                        battle_results["summary"][current_agent_name_for_stats]["total_score"] += score
                        battle_results["summary"][current_agent_name_for_stats]["games_played"] += 1
                        if winner_index == player_idx:
                            battle_results["summary"][current_agent_name_for_stats]["wins"] += 1
                winner_name_str = "Draw/No Winner"
                if winner_index is not None:
                    winning_agent_type = current_player_map.get(winner_index)
                    if winning_agent_type == "A": winner_name_str = agent_A_name
                    elif winning_agent_type == "B": winner_name_str = agent_B_name
                    else: winner_name_str = f"Unknown (Player {winner_index})"
                logger.info(f"      Game {game_num + 1} ended. Winner: {winner_name_str}. Scores: {final_scores}")
            battle_results["setups_results"].append(setup_summary_details)
            logger.info(f"  Finished setup: {setup_name}")
        for agent_name_key in battle_results["summary"]:
            if isinstance(battle_results["summary"][agent_name_key], dict):
                agent_stats = battle_results["summary"][agent_name_key]
                if agent_stats["games_played"] > 0:
                    agent_stats["win_rate"] = agent_stats["wins"] / agent_stats["games_played"]
                    agent_stats["avg_score"] = agent_stats["total_score"] / agent_stats["games_played"]
        logger.info("\nPairwise Battle Summary:")
        logger.info(f"  Total games run: {battle_results['summary']['total_games_run']}")
        logger.info(f"  Errors: {battle_results['summary']['errors']}")
        summary_A_data = battle_results['summary'][agent_A_name]
        logger.info(f"  {agent_A_name}: Wins={summary_A_data['wins']}, Games Played={summary_A_data['games_played']}, Win Rate={summary_A_data['win_rate']:.2f}, Avg Score={summary_A_data['avg_score']:.2f}")
        if agent_A_name != agent_B_name:
            summary_B_data = battle_results['summary'][agent_B_name]
            logger.info(f"  {agent_B_name}: Wins={summary_B_data['wins']}, Games Played={summary_B_data['games_played']}, Win Rate={summary_B_data['win_rate']:.2f}, Avg Score={summary_B_data['avg_score']:.2f}")
        self.evaluation_results[f"pairwise_{agent_A_name}_vs_{agent_B_name}"] = battle_results
        return battle_results

    def run_round_robin_tournament(self, list_of_agent_configs: List[AgentConfig], num_games_per_matchup: int = 1) -> Dict[str, Any]:
        # ... (implementation from the original file)
        num_total_agents = len(list_of_agent_configs)
        logger.info(f"Starting round-robin tournament for {num_total_agents} agents.")
        base_return_structure = { "error": "", "summary_table": {}, "ranking": [], "total_games_run": 0, "errors": 0, "num_agents": num_total_agents, "agent_configs": list_of_agent_configs, "num_games_per_matchup": num_games_per_matchup, "matchups_details": [] }
        if num_total_agents < 4:
            msg = "Round-robin tournament with '4 distinct agents per table' setup requires at least 4 unique agents."
            logger.error(msg)
            base_return_structure["error"] = msg
            return base_return_structure
        agent_names = [self._get_agent_name(conf) for conf in list_of_agent_configs]
        if len(agent_names) != len(set(agent_names)):
            msg = "Duplicate agent names found. Agent names must be unique for round-robin summary."
            logger.error(msg)
            base_return_structure["error"] = msg
            return base_return_structure
        tournament_results: Dict[str, Any] = dict(base_return_structure)
        tournament_results["summary_table"] = { name: {"wins": 0, "total_score": 0, "games_played": 0, "win_rate": 0.0, "avg_score": 0.0, "config": config} for name, config in zip(agent_names, list_of_agent_configs) }
        agent_indices = list(range(num_total_agents))
        matchup_agent_indices_combinations = list(itertools.combinations(agent_indices, 4))
        if not matchup_agent_indices_combinations:
             logger.error("No matchups could be generated despite having >= 4 agents. This is unexpected.")
             tournament_results["error"] = "Internal error: No matchups generated for >= 4 agents."
             return tournament_results
        logger.info(f"Generated {len(matchup_agent_indices_combinations)} unique matchups of 4 agents.")
        for matchup_idx_tuple in matchup_agent_indices_combinations:
            current_matchup_configs = [list_of_agent_configs[i] for i in matchup_idx_tuple]
            current_matchup_agent_names = [self._get_agent_name(conf) for conf in current_matchup_configs]
            matchup_detail = { "matchup_participants_names": current_matchup_agent_names, "matchup_participants_configs": current_matchup_configs, "games": [] }
            logger.info(f"  Processing matchup: {', '.join(current_matchup_agent_names)}")
            for game_num in range(num_games_per_matchup):
                tournament_results["total_games_run"] += 1
                game_specific_agent_order_configs = list(current_matchup_configs)
                random.shuffle(game_specific_agent_order_configs)
                game_agent_paths = [self._get_agent_path(conf) for conf in game_specific_agent_order_configs]
                table_agent_names_ordered = [self._get_agent_name(c) for c in game_specific_agent_order_configs]
                logger.info(f"    Starting game {game_num + 1}/{num_games_per_matchup} for matchup. Agents at table (Seat 0 to 3): {table_agent_names_ordered}")
                final_scores, winner_index, error_message = self.run_game(game_agent_paths)
                game_run_details = { "game_number": game_num + 1, "agents_at_table_ordered_configs": game_specific_agent_order_configs, "agents_at_table_ordered_names": table_agent_names_ordered, "scores": final_scores, "winner_player_index_at_table": winner_index, "error": error_message }
                matchup_detail["games"].append(game_run_details)
                if error_message:
                    logger.error(f"      Game ended with error: {error_message}")
                    tournament_results["errors"] += 1
                    continue
                if final_scores is None:
                    logger.warning("      Game ended with no scores and no error message. Counting as error for stat purposes.")
                    tournament_results["errors"] += 1
                    continue
                for player_idx, score in final_scores.items():
                    agent_config_for_this_player = game_specific_agent_order_configs[player_idx]
                    original_agent_name = self._get_agent_name(agent_config_for_this_player)
                    tournament_results["summary_table"][original_agent_name]["total_score"] += score
                    tournament_results["summary_table"][original_agent_name]["games_played"] += 1
                    if winner_index == player_idx:
                        tournament_results["summary_table"][original_agent_name]["wins"] += 1
                winner_display_name = "Draw/No Winner"
                if winner_index is not None:
                    winning_agent_config_at_table = game_specific_agent_order_configs[winner_index]
                    winner_display_name = self._get_agent_name(winning_agent_config_at_table)
                logger.info(f"      Game {game_num + 1} ended. Winner: {winner_display_name}. Scores: {final_scores}")
            tournament_results["matchups_details"].append(matchup_detail)
            logger.info(f"  Finished matchup: {', '.join(current_matchup_agent_names)}")
        for agent_name_key, stats_data in tournament_results["summary_table"].items():
            if stats_data["games_played"] > 0:
                stats_data["win_rate"] = stats_data["wins"] / stats_data["games_played"]
                stats_data["avg_score"] = stats_data["total_score"] / stats_data["games_played"]
        sorted_agents_for_ranking = sorted( tournament_results["summary_table"].items(), key=lambda item: (item[1]["win_rate"], item[1]["avg_score"]), reverse=True )
        tournament_results["ranking"] = [ {"rank": i + 1, "name": name, "stats": stats_dict_item} for i, (name, stats_dict_item) in enumerate(sorted_agents_for_ranking) ]
        logger.info("\nRound-Robin Tournament Summary:")
        logger.info(f"  Total agents participated: {num_total_agents}")
        logger.info(f"  Total unique matchups of 4 agents: {len(matchup_agent_indices_combinations)}")
        logger.info(f"  Games per matchup: {num_games_per_matchup}")
        logger.info(f"  Total games run: {tournament_results['total_games_run']}")
        logger.info(f"  Errors during games: {tournament_results['errors']}")
        logger.info("\n  Ranking:")
        if not tournament_results["ranking"]:
            logger.info("    No ranking available (e.g., no games played or error before ranking).")
        for rank_info in tournament_results["ranking"]:
           s_item = rank_info['stats']
           logger.info(f"  {rank_info['rank']:>2}. {rank_info['name']:<20}: Wins={s_item['wins']:>3}, GP={s_item['games_played']:>3}, WR={s_item['win_rate']:.2f}, AvgScore={s_item['avg_score']:>+7.2f}")
        self.evaluation_results[f"round_robin_{num_total_agents}_agents"] = tournament_results
        return tournament_results