import subprocess
import os
import random
from typing import List, Tuple, Optional, Dict, Any
from MahjongGB import MahjongFanCalculator
from .logging_utils import logger

# Tile Constants
WAN = "W"; TONG = "B"; TIAO = "T"; FENG = "F"; JIAN = "J"
ALL_TILES: List[str] = []
# Per the rules, no flower tiles are used (136 total tiles)
for suit_val in [WAN, TONG, TIAO]:
    for i in range(1, 10): ALL_TILES.extend([f"{suit_val}{i}"] * 4)
for i in range(1, 5): ALL_TILES.extend([f"{FENG}{i}"] * 4) # Winds E S W N
for i in range(1, 4): ALL_TILES.extend([f"{JIAN}{i}"] * 4) # Dragons R G Wh

class Agent:
    # ... (This class is unchanged)
    def __init__(self, agent_id: int, agent_path: str):
        self.agent_id = agent_id
        agent_command = ["python", "__main__.py"]
        self.process = subprocess.Popen(
            agent_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1, cwd=agent_path
        )
    def send_request(self, request_str: str):
        logger.debug(f"Sim: Agent {self.agent_id} sending request: {request_str}")
        if self.process.stdin: self.process.stdin.write(request_str + "\n"); self.process.stdin.flush()
    def receive_response(self) -> str:
        logger.debug(f"Sim: Agent {self.agent_id} ENTERING receive_response")
        if not self.process.stdout:
            logger.error(f"Sim: Agent {self.agent_id} stdout is None, cannot receive response.")
            return ""
        logger.debug(f"Sim: Agent {self.agent_id} waiting for response...")
        actual_response = ""
        log_buffer = []
        while True:
            line_read = self.process.stdout.readline().strip()
            logger.debug(f"Sim: Agent {self.agent_id} raw read: '{line_read}'")
            log_buffer.append(line_read)
            if not line_read:
                logger.warning(f"Sim: Agent {self.agent_id} stdout.readline() returned empty. Possible EOF or agent crash.")
                for i in range(len(log_buffer) - 2, -1, -1):
                    if log_buffer[i] and not log_buffer[i].startswith("AGENT"):
                        actual_response = log_buffer[i]
                        logger.info(f"Sim: Agent {self.agent_id} using last non-empty, non-debug line as response: '{actual_response}' due to EOF.")
                        break
                break
            if line_read.startswith("AGENT"):
                logger.info(f"AGENT_OUTPUT A{self.agent_id}: {line_read}")
                continue
            actual_response = line_read
            if "Traceback" in actual_response or "Error" in actual_response or "Exception" in actual_response:
                logger.error(f"Sim: Agent {self.agent_id} detected error markers in response: '{actual_response}'. Collecting details.")
                error_output = [actual_response]
                try:
                    for _ in range(10):
                        if self.process.stdout.closed:
                            logger.warning(f"Sim: Agent {self.agent_id} stdout closed while collecting error context.")
                            break
                        next_line = self.process.stdout.readline().strip()
                        logger.debug(f"Sim: Agent {self.agent_id} raw read (error context): '{next_line}'")
                        if next_line:
                            if next_line.startswith("AGENT"):
                                logger.info(f"AGENT_OUTPUT A{self.agent_id} (error context): {next_line}")
                                error_output.append(next_line)
                            else:
                                error_output.append(next_line)
                        elif not actual_response and not next_line:
                            break
                        elif actual_response and not next_line:
                            break
                    if any(line for line in error_output if line):
                        logger.error(f"--- Agent {self.agent_id} Diagnostic Output (from error markers) ---")
                        for err_line in error_output:
                            logger.error(f"A{self.agent_id} CAPTURED: {err_line}")
                        logger.error(f"--- End Agent {self.agent_id} Diagnostic ---")
                    break
                except Exception as e:
                    logger.exception(f"Sim: Error while trying to read extended output from agent {self.agent_id} (error context): {e}")
                    break
            break
        if not actual_response:
            logger.warning(f"Sim: Agent {self.agent_id} receive_response is returning empty. Full log buffer for this attempt: {log_buffer}")
        else:
            logger.debug(f"Sim: Agent {self.agent_id} receive_response determined actual response: '{actual_response}'")
        logger.debug(f"Sim: Agent {self.agent_id} EXITING receive_response with '{actual_response}' (type: {type(actual_response)})")
        return actual_response
    def close(self):
        if self.process:
            if self.process.stdin: self.process.stdin.close()
            if self.process.stdout: self.process.stdout.close()
            if self.process.stderr: self.process.stderr.close()
            self.process.terminate(); self.process.wait()

class Player:
    def __init__(self, player_id: int, agent_process: Agent, seat_wind: int):
        self.player_id: int = player_id
        self.agent: Agent = agent_process
        self.seat_wind: int = seat_wind # 0:E, 1:S, 2:W, 3:N
        self.hand: List[str] = []
        self.melds: List[Tuple[str, str, Any, Optional[int]]] = []
        self.discarded_tiles: List[str] = []
        self.score: int = 0
        self.personal_tile_wall: List[str] = [] # For duplicate mode

class GameState:
    def __init__(self, agents: List[Agent], 
                 prevalent_wind: Optional[int] = None, 
                 fixed_walls: Optional[Dict[int, List[str]]] = None):
        
        self.is_duplicate_mode = fixed_walls is not None
        self.fixed_walls = fixed_walls

        self.tile_wall: List[str] = [] if self.is_duplicate_mode else list(ALL_TILES)
        self.players: List[Player] = [Player(i, agents[i], i) for i in range(4)]
        self.dealer_player_index: int = 0
        self.current_player_index: int = self.dealer_player_index
        self.prevalent_wind: int = prevalent_wind if prevalent_wind is not None else 0
        
        # ... (rest of the attributes are the same)
        self.last_discarded_tile: Optional[str] = None
        self.last_discarding_player_index: Optional[int] = None
        self.game_over: bool = False
        self.winner_index: Optional[int] = None
        self.winning_tile: Optional[str] = None
        self.is_self_drawn_win: bool = False
        self.is_robbing_kong_win: bool = False
        self.turn_number: int = 0
        self.kong_declarer_index: Optional[int] = None
        self.kong_tile: Optional[str] = None
        self.just_declared_kong: bool = False
        self.just_discarded: bool = False
        self.about_to_BUGANG_tile: Optional[str] = None
        self.error_message: Optional[str] = None
        self.current_action_responses: Dict[int, str] = {}
        self.win_details: Optional[List[Tuple[int, int, str, str]]] = None
        self.final_scores: Dict[int, int] = {i: 0 for i in range(4)}
        self.drew_kong_replacement_this_action: bool = False
        self.pending_qiangganghu_check: bool = False

        self.shuffle_and_deal()

    def shuffle_and_deal(self):
        if self.is_duplicate_mode and self.fixed_walls:
            # Duplicate mode: deal from fixed walls for each seat
            for seat_idx, player in enumerate(self.players):
                # The player at index `seat_idx` is at seat `seat_idx` (E,S,W,N)
                # They get the wall pre-assigned to that seat.
                player.personal_tile_wall = list(self.fixed_walls[seat_idx]) # Make a copy
                # Deal first 13 tiles, sequentially
                player.hand = [player.personal_tile_wall.pop(0) for _ in range(13)]
        else:
            # Normal mode: shuffle and deal from a single wall
            random.shuffle(self.tile_wall)
            for player in self.players:
                player.hand = [self.tile_wall.pop() for _ in range(13)]

    def draw_tile(self, player_index: int) -> Optional[str]:
        self.just_discarded = False; self.just_declared_kong = False
        self.about_to_BUGANG_tile = None; self.drew_kong_replacement_this_action = False
        self.pending_qiangganghu_check = False

        if self.is_duplicate_mode:
            player = self.players[player_index]
            return player.personal_tile_wall.pop(0) if player.personal_tile_wall else None
        else:
            return self.tile_wall.pop() if self.tile_wall else None

    def is_wall_last(self) -> bool:
        """Checks if the game is on its last potential tile according to mode."""
        if self.is_duplicate_mode:
            # In duplicate, if any player's wall is empty, the game is ending.
            # This is because the game ends after the player upstream to the one with the empty wall discards.
            # So, the last tile is drawn from a non-empty wall, but the game state is "wall last".
            return any(not p.personal_tile_wall for p in self.players)
        else:
            return len(self.tile_wall) == 0

    # ... (the rest of GameState is mostly the same, but we need to update MahjongFanCalculator calls)
    
    def _get_relative_offer(self, meld_from_player_abs_idx: Optional[int], winner_abs_idx: int) -> int:
        if meld_from_player_abs_idx is None: return 0
        if meld_from_player_abs_idx == winner_abs_idx: return 0
        diff = (meld_from_player_abs_idx - winner_abs_idx + 4) % 4
        if diff == 1: return 3
        elif diff == 2: return 2
        elif diff == 3: return 1
        return 0

    def end_game(self, winner_index: Optional[int] = None, winning_tile: Optional[str] = None,
                 is_self_drawn: bool = False, is_robbing_kong: bool = False,
                 is_draw: bool = False, error_message: Optional[str] = None,
                 was_kong_replacement_draw: bool = False,
                 last_discarding_player_idx_for_payment: Optional[int] = None):
        self.game_over = True; self.winner_index = winner_index; self.winning_tile = winning_tile
        self.is_self_drawn_win = is_self_drawn; self.is_robbing_kong_win = is_robbing_kong
        self.error_message = error_message

        if is_draw or error_message:
            msg = f"ERROR: {error_message}" if error_message else "DRAW."
            logger.info(f"GAME ENDED ({msg})")
            self.final_scores = {p.player_id: p.score for p in self.players}
            return

        if winner_index is not None and winning_tile is not None:
            winner = self.players[winner_index]
            calculator_packs = []
            for meld_tuple in winner.melds:
                meld_type = meld_tuple[0].upper(); tile1 = meld_tuple[1]; data = meld_tuple[2]
                offer_relative = 0; actual_calc_meld_type = meld_type
                if meld_type == "CHI": offer_relative = self._get_relative_offer(meld_tuple[3], winner_index)
                elif meld_type in ["PENG", "GANG", "ANGANG", "BUGANG"]:
                    meld_from_abs_idx = data
                    offer_relative = self._get_relative_offer(meld_from_abs_idx, winner_index)
                    if meld_type == "BUGANG": actual_calc_meld_type = "GANG"
                    if meld_type == "ANGANG": actual_calc_meld_type = "KONG"
                calculator_packs.append((actual_calc_meld_type, tile1, offer_relative))
            visible_winning_tile_count = 0
            if winning_tile:
                for p_scan in self.players:
                    for discarded in p_scan.discarded_tiles:
                        if discarded == winning_tile: visible_winning_tile_count += 1
                    for meld in p_scan.melds:
                        meld_type = meld[0].upper(); meld_main_tile = meld[1]
                        if meld_type == 'PENG':
                            if meld_main_tile == winning_tile: visible_winning_tile_count += 3
                        elif meld_type in ['GANG', 'ANGANG', 'BUGANG']:
                            if meld_main_tile == winning_tile: visible_winning_tile_count += 4
                        elif meld_type == 'CHI':
                            suit = meld_main_tile[0]
                            try:
                                mid_num = int(meld_main_tile[1:])
                                if suit in [TIAO, TONG, WAN] and 1 < mid_num < 9:
                                    seq_tiles = [f"{suit}{mid_num-1}", meld_main_tile, f"{suit}{mid_num+1}"]
                                    for seq_t in seq_tiles:
                                        if seq_t == winning_tile: visible_winning_tile_count += 1
                            except ValueError: pass
            is_4th_tile_for_calc = (visible_winning_tile_count == 3)
            is_about_kong_for_calc = is_robbing_kong or (is_self_drawn and was_kong_replacement_draw)
            hand_copy = list(winner.hand)
            if is_self_drawn:
                if winning_tile in hand_copy: hand_copy.remove(winning_tile)
            hand_for_calculator = tuple(sorted(hand_copy))
            try:
                fans_calculator = MahjongFanCalculator(
                    pack=tuple(calculator_packs), hand=hand_for_calculator, winTile=winning_tile,
                    flowerCount=0, isSelfDrawn=is_self_drawn, is4thTile=is_4th_tile_for_calc,
                    isAboutKong=is_about_kong_for_calc, isWallLast=self.is_wall_last(), # MODIFIED
                    seatWind=winner.seat_wind, prevalentWind=self.prevalent_wind, verbose=False
                )
                # ... (rest of end_game is unchanged)
                fan_cnt_total = 0; self.win_details = []
                for fan_item in fans_calculator:
                    if len(fan_item) == 4:
                        fp_orig, cnt_orig, f_zh, f_en = fan_item
                        fp = int(fp_orig)
                        if isinstance(cnt_orig, str) and not cnt_orig.isdigit():
                            cnt = 1
                            if f_en == "Unknown Fan": f_en = cnt_orig
                            if f_zh == "Unknown Fan": f_zh = cnt_orig
                        else: cnt = int(cnt_orig)
                    elif len(fan_item) == 2:
                        fp_orig, cnt_orig = fan_item
                        fp = int(fp_orig)
                        if isinstance(cnt_orig, str) and not cnt_orig.isdigit():
                            cnt = 1; f_zh = cnt_orig; f_en = cnt_orig
                        else: cnt = int(cnt_orig); f_zh = "Unknown Fan"; f_en = "Unknown Fan"
                    else:
                        logger.warning(f"Unexpected fan item format from PyMahjongGB: {fan_item}")
                        continue
                    self.win_details.append((fp, cnt, f_zh, f_en)); fan_cnt_total += fp * cnt
                if fan_cnt_total < 8:
                    logger.info(f"Player {winner_index} HU claim has insufficient fans ({fan_cnt_total} < 8). Treating as Chombo.")
                    self.error_message = f"P{winner_index} Chombo - insufficient fans ({fan_cnt_total})."
                    chombo_penalty_winner = -8 * 3
                    self.players[winner_index].score += chombo_penalty_winner
                    for p_idx in range(4):
                        if p_idx != winner_index: self.players[p_idx].score += 8
                    self.final_scores = {p.player_id: p.score for p in self.players}; return
                base_score = 8
                if is_self_drawn:
                    pts_change = base_score + fan_cnt_total
                    self.players[winner_index].score += pts_change * 3
                    for p_idx in range(4):
                        if p_idx != winner_index: self.players[p_idx].score -= pts_change
                else:
                    payer_idx = last_discarding_player_idx_for_payment if is_robbing_kong else self.last_discarding_player_index
                    if payer_idx is None:
                        self.error_message = "Payer index not determined for non-self-drawn win."
                        logger.error(f"GAME ENDED (ERROR): {self.error_message}")
                        self.final_scores = {p.player_id: p.score for p in self.players}; return
                    if is_robbing_kong:
                        pts_change = base_score + fan_cnt_total
                        self.players[winner_index].score += pts_change
                        self.players[payer_idx].score -= pts_change
                    else:
                        pts_change = base_score + fan_cnt_total
                        self.players[winner_index].score += pts_change
                        self.players[payer_idx].score -= pts_change
                self.final_scores = {p.player_id: p.score for p in self.players}
                win_type_str = "ROBBING KONG" if is_robbing_kong else ("SELF-DRAWN (After Kong)" if was_kong_replacement_draw and is_self_drawn else ("SELF-DRAWN" if is_self_drawn else "DISCARD"))
                logger.info(f"GAME ENDED (WIN): Player {winner_index} wins with {fan_cnt_total} Fan (+{base_score} Base). Type: {win_type_str}.")
                if self.win_details: logger.info(f"  Fan Details: {self.win_details}")
            except Exception as e:
                logger.exception(f"Error during MahjongFanCalculator for P{winner_index}: {e}")
                self.error_message = f"P{winner_index} score calculation error: {e}"
                self.final_scores = {p.player_id: p.score for p in self.players}
        else:
            self.final_scores = {p.player_id: p.score for p in self.players}

    def can_player_hu_discard(self, player_idx: int, discarded_tile: str, is_potential_robbing_kong: bool = False) -> bool:
        player = self.players[player_idx]
        temp_hand = list(player.hand); temp_hand.append(discarded_tile); temp_hand.sort()
        calculator_melds = []
        for meld_tuple in player.melds:
            meld_type = meld_tuple[0].upper(); tile1 = meld_tuple[1]; data = meld_tuple[2]
            offer_relative = 0; actual_calc_meld_type = meld_type
            if meld_type == "CHI": offer_relative = self._get_relative_offer(meld_tuple[3], player_idx)
            elif meld_type in ["PENG", "GANG", "ANGANG", "BUGANG"]:
                offer_relative = self._get_relative_offer(data, player_idx)
                if meld_type == "BUGANG": actual_calc_meld_type = "GANG"
                if meld_type == "ANGANG": actual_calc_meld_type = "KONG"
            calculator_melds.append((actual_calc_meld_type, tile1, offer_relative))
        try:
            fan_total = MahjongFanCalculator(
                pack=tuple(calculator_melds), hand=tuple(temp_hand), winTile=discarded_tile, flowerCount=0,
                isSelfDrawn=False, is4thTile=False,
                isAboutKong=is_potential_robbing_kong,
                isWallLast=self.is_wall_last(), # MODIFIED
                seatWind=player.seat_wind, prevalentWind=self.prevalent_wind
            ).calculate_fan()
            return fan_total >= 8
        except Exception: return False
    
    # ... (rest of the file is unchanged)
    def can_player_peng(self, player_idx: int, discarded_tile: str) -> bool:
        return self.players[player_idx].hand.count(discarded_tile) >= 2
    def can_player_ming_kong_from_discard(self, player_idx: int, discarded_tile: str) -> bool:
        return self.players[player_idx].hand.count(discarded_tile) >= 3
    def get_chi_hand_tiles_to_remove(self, player_idx: int, discarded_tile_str: str, chi_request_middle_tile_str: str) -> Optional[List[str]]:
        player = self.players[player_idx]
        if not (len(discarded_tile_str) == 2 and len(chi_request_middle_tile_str) == 2): return None
        discarded_suit, middle_suit = discarded_tile_str[0], chi_request_middle_tile_str[0]
        if discarded_suit != middle_suit or discarded_suit not in [WAN, TONG, TIAO]: return None
        try: discarded_num, middle_num = int(discarded_tile_str[1:]), int(chi_request_middle_tile_str[1:])
        except ValueError: return None
        if not (1 < middle_num < 9): return None
        sequence_nums = [middle_num - 1, middle_num, middle_num + 1]
        if discarded_num not in sequence_nums: return None
        expected_sequence_tiles = [f"{discarded_suit}{n}" for n in sequence_nums]
        tiles_player_needs_in_hand = [t for t in expected_sequence_tiles if t != discarded_tile_str]
        if len(tiles_player_needs_in_hand) != 2: return None
        temp_hand = list(player.hand)
        for needed_tile in tiles_player_needs_in_hand:
            if needed_tile in temp_hand: temp_hand.remove(needed_tile)
            else: return None
        return sorted(tiles_player_needs_in_hand)

if __name__ == '__main__': pass