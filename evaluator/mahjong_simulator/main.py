from .game_utils import Agent, GameState
import sys
from typing import List, Tuple, Optional, Dict, Any
from .logging_utils import logger # Import logger

def run_game(agent_paths: List[str]) -> Tuple[Optional[Dict[int, int]], Optional[int], Optional[str]]:
    # Initial print can be kept for immediate console feedback, but also logged
    print(f"Starting game with agent paths: {agent_paths}", flush=True)
    logger.info(f"Starting game with agent paths: {agent_paths}")
    agents: List[Agent] = []
    gs: Optional[GameState] = None
    try:
        logger.info("Initializing agents...")
        if len(agent_paths) != 4:
            error_msg = f"Error: Expected 4 agent paths, got {len(agent_paths)}"
            logger.error(error_msg)
            # print(error_msg, flush=True) # Keep for console
            return None, None, error_msg

        for i in range(4):
            agents.append(Agent(agent_id=i, agent_path=agent_paths[i]))
        logger.info(f"Agents initialized with paths: {agent_paths}")

        logger.info("Sending initial newline to agents...")
        for i, agent_proc in enumerate(agents):
            agent_proc.send_request("")
            logger.debug(f"Sent initial newline to agent {i}.")
        logger.info("Initial newlines sent.")

        logger.info("Creating GameState...")
        gs = GameState(agents)
        logger.info("GameState created.")
        logger.info(f"Prevalent wind (Quan): {gs.prevalent_wind}")
        for i, player in enumerate(gs.players):
            logger.info(f"Player {i} initial hand: {player.hand}")


        # 4. Initial Agent Communication Loop
        for p_idx in range(4):
            player = gs.players[p_idx]
            agent = player.agent

            logger.info(f"Communicating with Player {p_idx} (Agent ID: {agent.agent_id}, Seat Wind: {player.seat_wind})...")

            req0 = f"0 {player.player_id} {gs.prevalent_wind}"
            logger.debug(f"P{p_idx} Sending Req0: '{req0}'")
            agent.send_request(req0)

            resp0 = agent.receive_response()
            logger.debug(f"P{p_idx} Received Resp0: '{resp0}'")
            if resp0 != "PASS":
                logger.error(f"Error: P{p_idx} Req0 expected PASS, got {resp0}")
                # Potentially raise an error or handle more gracefully

            hand_str = " ".join(player.hand)
            req1 = f"1 0 0 0 0 {hand_str}"
            logger.debug(f"P{p_idx} Sending Req1: '{req1}'")
            agent.send_request(req1)

            resp1 = agent.receive_response()
            logger.debug(f"P{p_idx} Received Resp1: '{resp1}'")
            if resp1 != "PASS":
                logger.error(f"Error: P{p_idx} Req1 expected PASS, got {resp1}")
                # Potentially raise an error or handle more gracefully

        logger.info("Initial communication sequence finished for all players.")

        # --- Main Game Loop ---
        logger.info("--- Starting Main Game Loop ---")
        while not gs.game_over:
            current_player_index = gs.current_player_index
            current_player = gs.players[current_player_index]
            current_agent = current_player.agent
            current_player.hand.sort()

            action_context_tile: Optional[str] = None
            response_from_agent: Optional[str] = None

            if gs.pending_qiangganghu_check:
                logger.info(f"--- Evaluating QiangGangHu for P{gs.last_discarding_player_index}'s BUGANG of {gs.about_to_BUGANG_tile} ---")
                gs.pending_qiangganghu_check = False

                bugang_tile_being_robbed = gs.about_to_BUGANG_tile
                bugang_player_idx = gs.last_discarding_player_index

                potential_robbers = []
                for p_idx, response_str in gs.current_action_responses.items():
                    if response_str.upper() == "HU":
                        if gs.can_player_hu_discard(p_idx, bugang_tile_being_robbed, len(gs.tile_wall), is_potential_robbing_kong=True):
                            potential_robbers.append(p_idx)
                            logger.info(f"  Player {p_idx} can QiangGangHu on {bugang_tile_being_robbed}.")
                        else:
                            logger.warning(f"  Player {p_idx} claimed HU on BUGANG {bugang_tile_being_robbed} but validation failed.")

                if potential_robbers:
                    robbing_player_idx = min(potential_robbers)
                    # Keep this print for console visibility
                    print(f"Player {robbing_player_idx} WINS by ROBBING THE KONG (QiangGangHu) of {bugang_tile_being_robbed} from P{bugang_player_idx}!", flush=True)
                    logger.info(f"Player {robbing_player_idx} WINS by ROBBING THE KONG (QiangGangHu) of {bugang_tile_being_robbed} from P{bugang_player_idx}!")


                    robbed_player = gs.players[bugang_player_idx]
                    for i, meld_tuple in enumerate(robbed_player.melds):
                        if meld_tuple[0] == 'BUGANG' and meld_tuple[1] == bugang_tile_being_robbed:
                            robbed_player.melds[i] = ('PENG', meld_tuple[1], meld_tuple[2], None)
                            robbed_player.hand.append(bugang_tile_being_robbed)
                            robbed_player.hand.sort()
                            logger.info(f"  P{bugang_player_idx}'s BUGANG of {bugang_tile_being_robbed} reverted. Hand: {robbed_player.hand}, Melds: {robbed_player.melds}")
                            break

                    gs.end_game(winner_index=robbing_player_idx, winning_tile=bugang_tile_being_robbed,
                                is_robbing_kong=True, last_discarding_player_idx_for_payment=bugang_player_idx)

                    for i in range(4):
                        gs.players[i].agent.send_request(f"3 {robbing_player_idx} HU")
                else:
                    success_bugang_tile = gs.about_to_BUGANG_tile
                    logger.info(f"No QiangGangHu. P{bugang_player_idx}'s BUGANG of {success_bugang_tile} is successful.")

                    if success_bugang_tile:
                        bugang_self_req_str = f"3 {bugang_player_idx} BUGANG {success_bugang_tile}"
                        bugang_agent = gs.players[bugang_player_idx].agent
                        logger.debug(f"Sim: Notifying P{bugang_player_idx} (Self) of successful BUGANG: '{bugang_self_req_str}'")
                        bugang_agent.send_request(bugang_self_req_str)
                        resp_self_bugang = bugang_agent.receive_response()
                        logger.debug(f"  P{bugang_player_idx} (Self) response to successful BUGANG notification of {success_bugang_tile}: '{resp_self_bugang}'")
                        if resp_self_bugang.upper() != "PASS":
                            logger.warning(f"WARNING: P{bugang_player_idx} (Self) did not PASS after successful BUGANG self-notification. Got: {resp_self_bugang}")

                    gs.current_player_index = bugang_player_idx
                    gs.just_declared_kong = True
                    gs.about_to_BUGANG_tile = None
                    gs.last_discarding_player_index = None
                    gs.current_action_responses.clear()
                    continue

            elif gs.just_declared_kong:
                logger.info(f"--- Turn {gs.turn_number} (Player {current_player_index} KONG REPLACEMENT) ---")
                logger.debug(f"Player {current_player_index} hand before KONG replacement: {current_player.hand}")
                gs.drew_kong_replacement_this_action = True

                action_context_tile = gs.draw_tile(current_player_index)
                if action_context_tile is None:
                    logger.info("Wall is empty during KONG replacement draw. Game is a draw.")
                    gs.end_game(is_draw=True, error_message="Wall empty on kong replacement", was_kong_replacement_draw=True)
                    break

                logger.info(f"Player {current_player_index} draws KONG replacement tile: {action_context_tile}")
                current_player.hand.append(action_context_tile)
                current_player.hand.sort()
                logger.debug(f"Player {current_player_index} hand after KONG replacement (before decision): {current_player.hand}")

                logger.debug(f"SIM_DEBUG: P{current_player_index} hand before agent decision (after KONG replacement draw {action_context_tile}): {current_player.hand}")
                request_str = f"2 {action_context_tile}"
                logger.debug(f"SIM_DEBUG: P{current_player_index} sending KONG replacement request: '{request_str}'")
                current_agent.send_request(request_str)
                response_from_agent = current_agent.receive_response()
                logger.debug(f"SIM_DEBUG: P{current_player_index} agent KONG replacement response: '{response_from_agent}'")
                logger.info(f"Player {current_player_index} KONG replacement draw req: '{request_str}', Agent response: '{response_from_agent}'")

            elif gs.just_discarded:
                logger.info(f"--- Evaluating Discard from P{gs.last_discarding_player_index} (Tile: {gs.last_discarded_tile}) ---")
                logger.debug(f"  Responses from other players: {gs.current_action_responses}")

                potential_actions = []
                num_wall_tiles = len(gs.tile_wall)
                processed_player_action_on_discard = False

                for p_idx, response_str in gs.current_action_responses.items():
                    if gs.last_discarding_player_index is None or p_idx == gs.last_discarding_player_index:
                        continue

                    response_parts = response_str.split()
                    if not response_parts: continue
                    action = response_parts[0].upper()
                    player_to_check = gs.players[p_idx]

                    if action == "HU":
                        is_robbing_kong_context = (gs.about_to_BUGANG_tile == gs.last_discarded_tile)
                        if gs.can_player_hu_discard(p_idx, gs.last_discarded_tile, num_wall_tiles, is_potential_robbing_kong=is_robbing_kong_context):
                            potential_actions.append({'type': 'HU', 'player_idx': p_idx, 'tile': gs.last_discarded_tile, 'is_robbing': is_robbing_kong_context})
                            logger.info(f"  Player {p_idx} validated HU on {gs.last_discarded_tile} (Robbing context: {is_robbing_kong_context}).")
                        else:
                            logger.warning(f"  Player {p_idx} claimed HU on {gs.last_discarded_tile} but validation FAILED. Hand: {player_to_check.hand}, Melds: {player_to_check.melds}")
                    elif action == "PENG":
                        if gs.can_player_peng(p_idx, gs.last_discarded_tile):
                            if len(response_parts) < 2:
                                logger.warning(f"  Player {p_idx} PENG response invalid (missing tile to play): '{response_str}'")
                                continue
                            tile_to_play_after_peng = response_parts[1]
                            potential_actions.append({
                                'type': 'PENG', 'player_idx': p_idx,
                                'tile': gs.last_discarded_tile,
                                'play_after': tile_to_play_after_peng
                            })
                            logger.info(f"  Player {p_idx} validated PENG on {gs.last_discarded_tile}.")
                        else:
                            logger.warning(f"  Player {p_idx} claimed PENG on {gs.last_discarded_tile} but validation FAILED (Hand: {player_to_check.hand}).")
                    elif action == "GANG":
                        if gs.can_player_ming_kong_from_discard(p_idx, gs.last_discarded_tile):
                            potential_actions.append({'type': 'KONG', 'player_idx': p_idx, 'tile': gs.last_discarded_tile})
                            logger.info(f"  Player {p_idx} validated KONG on {gs.last_discarded_tile}.")
                        else:
                            logger.warning(f"  Player {p_idx} claimed KONG on {gs.last_discarded_tile} but validation FAILED (Hand: {player_to_check.hand}).")
                    elif action == "CHI":
                        if p_idx == (gs.last_discarding_player_index + 1) % 4:
                            if len(response_parts) < 3:
                                logger.warning(f"  Player {p_idx} CHI response invalid (missing middle tile or tile to play): '{response_str}'")
                                continue
                            chi_middle_tile = response_parts[1]
                            tile_to_play_after_chi = response_parts[2]
                            required_hand_tiles_for_chi = gs.get_chi_hand_tiles_to_remove(p_idx, gs.last_discarded_tile, chi_middle_tile)
                            if required_hand_tiles_for_chi:
                                potential_actions.append({
                                    'type': 'CHI', 'player_idx': p_idx,
                                    'discarded_tile': gs.last_discarded_tile,
                                    'middle_tile': chi_middle_tile,
                                    'play_after': tile_to_play_after_chi,
                                    'hand_tiles_for_chi': required_hand_tiles_for_chi
                                })
                                logger.info(f"  Player {p_idx} validated CHI on {gs.last_discarded_tile} with middle {chi_middle_tile}.")
                            else:
                                logger.warning(f"  Player {p_idx} claimed CHI on {gs.last_discarded_tile} with middle {chi_middle_tile} but validation FAILED (Hand: {player_to_check.hand}).")
                        else:
                             logger.debug(f"  Player {p_idx} attempted CHI out of turn for {gs.last_discarded_tile}.")

                hu_action = None
                all_hu_actions = [act for act in potential_actions if act['type'] == 'HU']
                if all_hu_actions:
                    hu_action = min(all_hu_actions, key=lambda x: x['player_idx'])
                    if len(all_hu_actions) > 1:
                         logger.info(f"  Multiple valid HU claims. Player {hu_action['player_idx']} selected.")

                if hu_action:
                    acting_player_idx = hu_action['player_idx']
                    winning_tile = hu_action['tile']
                    is_robbing = hu_action.get('is_robbing', False)
                    # Keep this print for console visibility
                    print(f"Player {acting_player_idx} WINS by HU on discard {winning_tile} from P{gs.last_discarding_player_index}! (Robbing: {is_robbing})", flush=True)
                    logger.info(f"Player {acting_player_idx} WINS by HU on discard {winning_tile} from P{gs.last_discarding_player_index}! (Robbing: {is_robbing})")
                    gs.end_game(winner_index=acting_player_idx, winning_tile=winning_tile, is_self_drawn=False, is_robbing_kong=is_robbing)
                    for i in range(4): gs.players[i].agent.send_request(f"3 {acting_player_idx} HU")
                    processed_player_action_on_discard = True

                else:
                    kong_action = next((act for act in potential_actions if act['type'] == 'KONG'), None)
                    if kong_action:
                        acting_player_idx = kong_action['player_idx']
                        acted_tile = kong_action['tile']
                        acting_player = gs.players[acting_player_idx]
                        logger.info(f"Player {acting_player_idx} KONGs (Ming) {acted_tile} from P{gs.last_discarding_player_index}.")

                        for _ in range(3):
                            if acted_tile in acting_player.hand: acting_player.hand.remove(acted_tile)
                            else:
                                gs.end_game(error_message=f"P{acting_player_idx} KONG validation mismatch for {acted_tile}"); break
                        if gs.game_over: continue

                        acting_player.melds.append(('GANG', acted_tile, gs.last_discarding_player_index))
                        acting_player.hand.sort()
                        logger.debug(f"  P{acting_player_idx} hand after KONG: {acting_player.hand}, Melds: {acting_player.melds}")

                        kong_broadcast_msg = f"3 {acting_player_idx} GANG"
                        for i in range(4):
                            if i == acting_player_idx: continue
                            gs.players[i].agent.send_request(kong_broadcast_msg)
                            gs.players[i].agent.receive_response()

                        gs.current_player_index = acting_player_idx
                        gs.just_declared_kong = True
                        gs.just_discarded = False
                        gs.last_discarded_tile = None
                        gs.last_discarding_player_index = None
                        gs.current_action_responses.clear()
                        gs.about_to_BUGANG_tile = None
                        processed_player_action_on_discard = True

                    else:
                        peng_action = next((act for act in potential_actions if act['type'] == 'PENG'), None)
                        if peng_action:
                            acting_player_idx = peng_action['player_idx']
                            acted_tile = peng_action['tile']
                            tile_to_play_after_peng = peng_action['play_after']
                            acting_player = gs.players[acting_player_idx]
                            logger.info(f"Player {acting_player_idx} PENGs {acted_tile} from P{gs.last_discarding_player_index}.")

                            for _ in range(2):
                                if acted_tile in acting_player.hand: acting_player.hand.remove(acted_tile)
                                else: gs.end_game(error_message=f"P{acting_player_idx} PENG validation mismatch for {acted_tile}"); break
                            if gs.game_over: continue

                            acting_player.melds.append(('PENG', acted_tile, gs.last_discarding_player_index))
                            acting_player.hand.sort()
                            logger.debug(f"  P{acting_player_idx} hand after PENG: {acting_player.hand}, Melds: {acting_player.melds}")

                            logger.debug(f"SIM_DEBUG: P{acting_player_idx} hand before PENG discard decision ('{tile_to_play_after_peng}'): {acting_player.hand}")
                            if tile_to_play_after_peng not in acting_player.hand:
                                gs.end_game(error_message=f"P{acting_player_idx} PENG invalid discard {tile_to_play_after_peng}. Hand: {acting_player.hand}")
                                processed_player_action_on_discard = True
                            else:
                                logger.debug(f"SIM_DEBUG: P{acting_player_idx} PENG discard is valid.")
                                acting_player.hand.remove(tile_to_play_after_peng)
                                acting_player.hand.sort()
                                acting_player.discarded_tiles.append(tile_to_play_after_peng)
                                logger.info(f"  P{acting_player_idx} then discards {tile_to_play_after_peng}. Hand: {acting_player.hand}")

                                peng_broadcast_msg = f"3 {acting_player_idx} PENG {tile_to_play_after_peng}"
                                new_responses = {}
                                for i in range(4):
                                    if i == acting_player_idx: continue
                                    gs.players[i].agent.send_request(peng_broadcast_msg)
                                    new_responses[i] = gs.players[i].agent.receive_response()
                                    logger.debug(f"    P{i} (Agent {gs.players[i].agent.agent_id}) response to PENG broadcast (P{acting_player_idx} played {tile_to_play_after_peng}): '{new_responses[i]}'")

                                gs.last_discarded_tile = tile_to_play_after_peng
                                gs.last_discarding_player_index = acting_player_idx
                                gs.current_player_index = acting_player_idx
                                gs.just_discarded = True
                                gs.current_action_responses = new_responses
                                gs.about_to_BUGANG_tile = None
                                processed_player_action_on_discard = True

                                acting_agent_for_peng = gs.players[acting_player_idx].agent
                                logger.debug(f"Sim: Notifying P{acting_player_idx} (self) of PENG and PLAY: '{peng_broadcast_msg}'")
                                acting_agent_for_peng.send_request(peng_broadcast_msg)
                                self_response_peng = acting_agent_for_peng.receive_response()
                                logger.debug(f"Sim: P{acting_player_idx} (self) response to PENG-PLAY notification: '{self_response_peng}'")
                                if self_response_peng.upper() != "PASS":
                                    logger.warning(f"WARNING: P{acting_player_idx} (self) did not PASS after PENG-PLAY notification. Got: {self_response_peng}")

                        else:
                            chi_action = next((act for act in potential_actions if act['type'] == 'CHI'), None)
                            if chi_action:
                                acting_player_idx = chi_action['player_idx']
                                discarded_chi_tile = chi_action['discarded_tile']
                                middle_tile = chi_action['middle_tile']
                                tile_to_play_after_chi = chi_action['play_after']
                                hand_tiles_for_chi = chi_action['hand_tiles_for_chi']
                                acting_player = gs.players[acting_player_idx]

                                logger.info(f"Player {acting_player_idx} CHIs {discarded_chi_tile} (using middle {middle_tile}, needs {hand_tiles_for_chi}) from P{gs.last_discarding_player_index}.")

                                for tile_to_remove in hand_tiles_for_chi:
                                    if tile_to_remove in acting_player.hand: acting_player.hand.remove(tile_to_remove)
                                    else: gs.end_game(error_message=f"P{acting_player_idx} CHI validation mismatch for {tile_to_remove}"); break
                                if gs.game_over: continue

                                suit = middle_tile[0]
                                mid_num = int(middle_tile[1:])
                                full_sequence_str = f"{suit}{mid_num-1}{suit}{mid_num}{suit}{mid_num+1}"
                                acting_player.melds.append(('CHI', middle_tile, full_sequence_str, gs.last_discarding_player_index))
                                acting_player.hand.sort()
                                logger.debug(f"  P{acting_player_idx} hand after CHI: {acting_player.hand}, Melds: {acting_player.melds}")

                                logger.debug(f"SIM_DEBUG: P{acting_player_idx} hand before CHI discard decision ('{tile_to_play_after_chi}'): {acting_player.hand}")
                                if tile_to_play_after_chi not in acting_player.hand:
                                    gs.end_game(error_message=f"P{acting_player_idx} CHI invalid discard {tile_to_play_after_chi}. Hand: {acting_player.hand}")
                                    processed_player_action_on_discard = True
                                else:
                                    logger.debug(f"SIM_DEBUG: P{acting_player_idx} CHI discard is valid.")
                                    acting_player.hand.remove(tile_to_play_after_chi)
                                    acting_player.hand.sort()
                                    acting_player.discarded_tiles.append(tile_to_play_after_chi)
                                    logger.info(f"  P{acting_player_idx} then discards {tile_to_play_after_chi}. Hand: {acting_player.hand}")

                                    chi_broadcast_msg = f"3 {acting_player_idx} CHI {middle_tile} {tile_to_play_after_chi}"
                                    new_responses = {}
                                    for i in range(4):
                                        if i == acting_player_idx: continue
                                        gs.players[i].agent.send_request(chi_broadcast_msg)
                                        new_responses[i] = gs.players[i].agent.receive_response()
                                        logger.debug(f"    P{i} (Agent {gs.players[i].agent.agent_id}) response to CHI broadcast (P{acting_player_idx} played {tile_to_play_after_chi}): '{new_responses[i]}'")

                                    gs.last_discarded_tile = tile_to_play_after_chi
                                    gs.last_discarding_player_index = acting_player_idx
                                    gs.current_player_index = acting_player_idx
                                    gs.just_discarded = True
                                    gs.current_action_responses = new_responses
                                    gs.about_to_BUGANG_tile = None
                                    processed_player_action_on_discard = True

                                acting_agent_for_chi = gs.players[acting_player_idx].agent
                                logger.debug(f"Sim: Notifying P{acting_player_idx} (self) of CHI and PLAY: '{chi_broadcast_msg}'")
                                acting_agent_for_chi.send_request(chi_broadcast_msg)
                                self_response_chi = acting_agent_for_chi.receive_response()
                                logger.debug(f"Sim: P{acting_player_idx} (self) response to CHI-PLAY notification: '{self_response_chi}'")
                                if self_response_chi.upper() != "PASS":
                                    logger.warning(f"WARNING: P{acting_player_idx} (self) did not PASS after CHI-PLAY notification. Got: {self_response_chi}")

                if not processed_player_action_on_discard and not gs.game_over:
                    logger.info(f"  All other players PASS on discarded tile {gs.last_discarded_tile} from P{gs.last_discarding_player_index}.")
                    gs.current_player_index = (gs.last_discarding_player_index + 1) % 4
                    gs.just_discarded = False
                    gs.last_discarded_tile = None
                    gs.last_discarding_player_index = None
                    gs.current_action_responses.clear()
                    gs.about_to_BUGANG_tile = None
                    continue

                if gs.game_over:
                    break
                if processed_player_action_on_discard:
                    continue
                if not processed_player_action_on_discard:
                    logger.info(f"  All other players PASS on discarded tile {gs.last_discarded_tile} from P{gs.last_discarding_player_index} (final check).")
                    gs.current_player_index = (gs.last_discarding_player_index + 1) % 4
                    gs.just_discarded = False
                    gs.last_discarded_tile = None
                    gs.last_discarding_player_index = None
                    gs.current_action_responses.clear()
                    gs.about_to_BUGANG_tile = None
                    continue

            else: # Normal player turn: Draw a tile
                gs.turn_number += 1
                logger.info(f"--- Turn {gs.turn_number}: Player {current_player_index} (Seat: {current_player.seat_wind}, AgentID: {current_agent.agent_id}) ---")
                logger.debug(f"Player {current_player_index} hand before normal draw: {current_player.hand}")

                action_context_tile = gs.draw_tile(current_player_index)
                if action_context_tile is None:
                    logger.info("Wall is empty during normal draw. Game is a draw.")
                    gs.end_game(is_draw=True, error_message="Wall empty on normal draw")
                    break

                logger.info(f"Player {current_player_index} draws normal tile: {action_context_tile}")
                current_player.hand.append(action_context_tile)
                current_player.hand.sort()
                logger.debug(f"Player {current_player_index} hand after normal draw (before decision): {current_player.hand}")

                logger.debug(f"SIM_DEBUG: P{current_player_index} hand before agent decision (after draw {action_context_tile}): {current_player.hand}")
                request_str = f"2 {action_context_tile}"
                logger.debug(f"SIM_DEBUG: P{current_player_index} sending draw request: '{request_str}'")
                current_agent.send_request(request_str)
                response_from_agent = current_agent.receive_response()
                logger.debug(f"SIM_DEBUG: P{current_player_index} agent draw response: '{response_from_agent}'")
                logger.info(f"Player {current_player_index} normal draw req: '{request_str}', Agent response: '{response_from_agent}'")

            if gs.game_over:
                break

            logger.debug(f"DEBUG: P{current_player_index} - In COMMON RESPONSE PROCESSING BLOCK.")
            logger.debug(f"DEBUG: P{current_player_index} - response_from_agent = '{response_from_agent}' (type: {type(response_from_agent)})")
            if response_from_agent is None:
                err_msg = f"P{current_player_index} missing agent response unexpectedly (not a discard evaluation cycle)."
                logger.critical(f"CRITICAL LOGIC ERROR: {err_msg}. response_from_agent was None.")
                gs.end_game(error_message=err_msg)
                break

            current_player.hand.sort()
            logger.info(f"Player {current_player_index} processing response: '{response_from_agent}' for action tile '{action_context_tile}'. Hand: {current_player.hand}")

            action_parts = response_from_agent.split()
            action_type = action_parts[0] if action_parts else "NO_ACTION"

            if action_type == "HU":
                # Keep this print for console visibility
                print(f"Player {current_player_index} declares SELF-DRAWN HU with {action_context_tile}!", flush=True)
                logger.info(f"Player {current_player_index} declares SELF-DRAWN HU with {action_context_tile}!")
                gs.end_game(
                    winner_index=current_player_index,
                    winning_tile=action_context_tile,
                    is_self_drawn=True,
                    was_kong_replacement_draw=gs.drew_kong_replacement_this_action
                )

            elif action_type == "PLAY":
                if len(action_parts) < 2:
                    err_msg = f"P{current_player_index} PLAY action missing tile. Response: '{response_from_agent}'"
                    logger.critical(err_msg)
                    gs.end_game(error_message=err_msg)
                    break
                tile_played = action_parts[1]

                if tile_played not in current_player.hand:
                    err_msg = f"P{current_player_index} tried to play {tile_played} which is NOT in hand {current_player.hand} (action tile was: {action_context_tile})."
                    logger.critical(err_msg)
                    gs.end_game(error_message=err_msg)
                    break

                current_player.hand.remove(tile_played)
                current_player.hand.sort()
                current_player.discarded_tiles.append(tile_played)
                gs.last_discarded_tile = tile_played
                gs.last_discarding_player_index = current_player_index
                gs.just_discarded = True

                logger.info(f"Player {current_player_index} plays tile: {tile_played}. Hand: {current_player.hand}")

                gs.current_action_responses = {}
                for p_idx in range(4):
                    if p_idx == current_player_index:
                        continue
                    other_player = gs.players[p_idx]
                    other_agent = other_player.agent
                    request_play_broadcast = f"3 {current_player_index} PLAY {tile_played}"
                    other_agent.send_request(request_play_broadcast)
                    response_broadcast = other_agent.receive_response()
                    gs.current_action_responses[p_idx] = response_broadcast
                    logger.debug(f"  P{p_idx} (Agent {other_agent.agent_id}) saw P{current_player_index} play {tile_played}. Agent response: '{response_broadcast}'")

                self_play_notification_req = f"3 {current_player_index} PLAY {tile_played}"
                current_agent.send_request(self_play_notification_req)
                self_play_notification_resp = current_agent.receive_response()
                if self_play_notification_resp.upper() != "PASS":
                    logger.warning(f"WARNING: Player {current_player_index} (acting agent) did not respond with PASS to self-play notification. Got: {self_play_notification_resp}")

            elif action_type == "GANG":
                if len(action_parts) < 2:
                    err_msg = f"P{current_player_index} GANG action missing tile. Response: '{response_from_agent}'"
                    logger.critical(err_msg)
                    gs.end_game(error_message=err_msg)
                    break
                tile_kong = action_parts[1]

                if current_player.hand.count(tile_kong) != 4:
                    err_msg = f"P{current_player_index} declared GANG {tile_kong} but hand count is {current_player.hand.count(tile_kong)} (requires 4). Hand: {current_player.hand} (action tile was: {action_context_tile})"
                    logger.critical(err_msg)
                    gs.end_game(error_message=err_msg)
                    break

                for _ in range(4):
                    current_player.hand.remove(tile_kong)
                current_player.hand.sort()
                current_player.melds.append(('ANGANG', tile_kong, current_player_index))
                logger.info(f"Player {current_player_index} declares ANGANG with {tile_kong}. Hand: {current_player.hand}. Melds: {current_player.melds}")

                an_gang_req_str = f"3 {current_player_index} GANG {tile_kong}"
                logger.debug(f"Sim: Broadcasting AnGang notification: '{an_gang_req_str}'")
                for notify_p_idx in range(4):
                    notified_agent = gs.players[notify_p_idx].agent
                    notified_agent.send_request(an_gang_req_str)
                    resp = notified_agent.receive_response()
                    logger.debug(f"  P{notify_p_idx} (Agent {notified_agent.agent_id}) response to AnGang broadcast from P{current_player_index} of {tile_kong}: '{resp}'")
                    if notify_p_idx == current_player_index and resp.upper() != "PASS":
                        logger.warning(f"WARNING: P{current_player_index} (self) did not PASS after AnGang self-notification. Got: {resp}")
                    elif notify_p_idx != current_player_index and resp.upper() != "PASS":
                         logger.warning(f"WARNING: P{notify_p_idx} did not PASS after AnGang notification for P{current_player_index}. Got: {resp}")
                gs.just_declared_kong = True

            elif action_type == "BUGANG":
                if len(action_parts) < 2:
                    err_msg = f"P{current_player_index} BUGANG action missing tile. Response: '{response_from_agent}'"
                    logger.critical(err_msg)
                    gs.end_game(error_message=err_msg); break
                tile_kong = action_parts[1]

                if tile_kong != action_context_tile:
                    err_msg = f"P{current_player_index} BUGANG tile {tile_kong} must be the action context tile {action_context_tile}."
                    logger.critical(err_msg)
                    gs.end_game(error_message=err_msg); break

                peng_meld_to_upgrade = None
                peng_meld_idx = -1
                original_peng_from_idx = -1
                for i, meld in enumerate(current_player.melds):
                    if meld[0] == 'PENG' and meld[1] == tile_kong:
                        peng_meld_to_upgrade = meld
                        peng_meld_idx = i
                        original_peng_from_idx = meld[2]
                        break

                if not peng_meld_to_upgrade:
                    err_msg = f"P{current_player_index} declared BUGANG {tile_kong} but no corresponding PENG found. Melds: {current_player.melds}"
                    logger.critical(err_msg)
                    gs.end_game(error_message=err_msg); break

                current_player.melds[peng_meld_idx] = ('BUGANG', tile_kong, original_peng_from_idx, None)
                current_player.hand.remove(tile_kong)
                current_player.hand.sort()
                logger.info(f"Player {current_player_index} declares BUGANG with {tile_kong}. Hand: {current_player.hand}, Melds: {current_player.melds}")

                gs.about_to_BUGANG_tile = tile_kong
                gs.last_discarding_player_index = current_player_index
                gs.current_action_responses.clear()

                bugang_broadcast_msg = f"3 {current_player_index} BUGANG {tile_kong}"
                for p_idx in range(4):
                    if p_idx == current_player_index: continue
                    gs.players[p_idx].agent.send_request(bugang_broadcast_msg)
                    response_qgh = gs.players[p_idx].agent.receive_response()
                    gs.current_action_responses[p_idx] = response_qgh
                    logger.debug(f"  P{p_idx} response to P{current_player_index}'s BUGANG of {tile_kong}: '{response_qgh}'")

                gs.pending_qiangganghu_check = True
                continue

            else:
                err_msg = f"P{current_player_index} responded with unexpected action '{response_from_agent}' after action on tile '{action_context_tile}'. Expected PLAY, GANG, BUGANG, or HU."
                logger.critical(err_msg)
                gs.end_game(error_message=err_msg)

            if gs.game_over:
                break

            if gs.turn_number >= 80 and not gs.game_over:
                logger.info("Turn limit (80) reached.")
                # Keep this print for console visibility
                print("Turn limit (80) reached.", flush=True)
                gs.end_game(is_draw=True, error_message="Turn limit reached")

        logger.info("--- Main Game Loop Ended ---")
        print("\n--- Main Game Loop Ended ---", flush=True) # Keep for console

        # --- Print Game Over Information ---
        # Keep these prints for console visibility
        print("\n--- Game Over ---", flush=True)
        logger.info("--- Game Over ---")
        if gs.error_message:
            print(f"Game ended due to an error: {gs.error_message}", flush=True)
            logger.error(f"Game ended due to an error: {gs.error_message}")
        elif gs.winner_index is not None:
            print(f"Player {gs.winner_index} is the winner!", flush=True)
            logger.info(f"Player {gs.winner_index} is the winner!")
            if gs.winning_tile:
                print(f"Winning Tile: {gs.winning_tile}", flush=True)
                logger.info(f"Winning Tile: {gs.winning_tile}")

            win_type_str = "UNKNOWN"
            if gs.is_self_drawn_win:
                win_type_str = "SELF-DRAWN (Zimo)"
                if gs.drew_kong_replacement_this_action:
                    win_type_str = "SELF-DRAWN After Kong (Ling Shang Kai Hua)"
            elif gs.is_robbing_kong_win:
                win_type_str = "Robbing the Kong (Qiang Gang Hu)"
            elif gs.last_discarding_player_index is not None:
                win_type_str = f"Win by Discard from Player {gs.last_discarding_player_index}"
            print(f"Type: {win_type_str}", flush=True)
            logger.info(f"Type: {win_type_str}")

            if gs.win_details:
                print("Fan Breakdown:", flush=True)
                logger.info("Fan Breakdown:")
                total_fan_calc = 0
                for points, count, name_zh, name_en in gs.win_details:
                    print(f"  - {name_zh} ({name_en}): {points} Fan x {count}", flush=True)
                    logger.info(f"  - {name_zh} ({name_en}): {points} Fan x {count}")
                    total_fan_calc += points * count
                print(f"Total Fan Points (from calculator): {total_fan_calc}", flush=True)
                logger.info(f"Total Fan Points (from calculator): {total_fan_calc}")
        else:
            print("Game ended in a draw (e.g., wall empty or turn limit).", flush=True)
            logger.info("Game ended in a draw (e.g., wall empty or turn limit).")

        print("\nFinal Scores:", flush=True)
        logger.info("Final Scores:")
        for i in range(4):
            player_score = gs.final_scores.get(i, gs.players[i].score) if gs else "N/A"
            print(f"  Player {i}: {player_score} points", flush=True)
            logger.info(f"  Player {i}: {player_score} points")

        if gs:
            logger.debug("Returning scores, winner index and error message from GameState.")
            return gs.final_scores, gs.winner_index, gs.error_message
        else:
            logger.error("Error: GameState was not available to return results.")
            # print("Error: GameState was not available to return results.", flush=True) # Keep for console
            return None, None, "Critical Error: GameState not initialized before game end."

    except Exception as e:
        # Keep this print for console visibility
        print(f"An unhandled error occurred during the game: {e}", flush=True)
        logger.exception(f"An unhandled error occurred during the game: {e}") # Logs with stack trace
        # import traceback # Not needed with logger.exception
        # traceback.print_exc(file=sys.stdout) # Already handled by logger.exception
        # sys.stdout.flush() # Logger should handle flushing if configured, or it goes to stderr
        error_detail = f"An unhandled error occurred: {str(e)}"
        final_scores_on_error = None
        winner_index_on_error = None
        if gs:
            final_scores_on_error = gs.final_scores
            winner_index_on_error = gs.winner_index
            if gs.error_message:
                 error_detail += f". Previous game error: {gs.error_message}"
        return final_scores_on_error, winner_index_on_error, error_detail
    finally:
        logger.info("Closing agent processes...")
        # print("\nClosing agent processes...", flush=True) # Keep for console
        sys.stdout.flush() # Ensure console prints are flushed before logger messages if interleaved
        for i, agent_proc in enumerate(agents):
            if agent_proc:
                logger.info(f"Closing agent {i} (Path: {agent_paths[i] if i < len(agent_paths) else 'N/A'})...")
                # print(f"Closing agent {i} (Path: {agent_paths[i] if i < len(agent_paths) else 'N/A'})...", flush=True) # Keep for console
                agent_proc.close()
                logger.info(f"Agent {i} closed.")
                # print(f"Agent {i} closed.", flush=True) # Keep for console
        logger.info("All agent processes closed.")
        # print("All agent processes closed.", flush=True) # Keep for console

if __name__ == "__main__":
    # Default paths assume main.py is in mahjong_simulator and bots are in project_root/bots/
    # Thus, paths are relative from where 'python mahjong_simulator/main.py' would be run (i.e., project root)
    # or resolved correctly if main.py is run as a module from project root.
    default_test_paths = ["bots/base_bot", "bots/base_bot", "bots/base_bot", "bots/base_bot"]
    # Example for testing with agent_trainer (moved one) and root agent_trainer:
    # default_test_paths = ["bots/base_bot", "bots/agent_trainer", "agent_trainer", "bots/base_bot"]

    # Keep this print for console visibility when run directly
    print(f"Running a test game from __main__ with agent paths: {default_test_paths}", flush=True)
    logger.info(f"Running a test game from __main__ with agent paths: {default_test_paths}")


    final_scores, winner_idx, error_msg = run_game(default_test_paths)

    # Keep these prints for console visibility when run directly
    if error_msg:
        print(f"Game finished with an error: {error_msg}", flush=True)
        logger.error(f"Game finished with an error from __main__: {error_msg}")

    if winner_idx is not None:
        print(f"Winner: Player {winner_idx}", flush=True)
        logger.info(f"Winner from __main__: Player {winner_idx}")
    elif not error_msg:
        print("Game ended in a draw or without a clear winner.", flush=True)
        logger.info("Game ended in a draw or without a clear winner from __main__.")

    if final_scores:
        print("Final scores:", flush=True)
        logger.info("Final scores from __main__:")
        for player_id, score_val in final_scores.items():
            print(f"  Player {player_id}: {score_val}", flush=True)
            logger.info(f"  Player {player_id}: {score_val}")
    elif not error_msg:
        print("Game finished, but no scores were returned (and no error message).", flush=True)
        logger.warning("Game finished from __main__, but no scores were returned (and no error message).")
