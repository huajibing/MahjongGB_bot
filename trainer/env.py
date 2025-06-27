from agent import MahjongGBAgent

import random
from collections import defaultdict

try:
    from MahjongGB import MahjongFanCalculator, MahjongShanten, RegularShanten, SevenPairsShanten, ThirteenOrphansShanten, HonorsAndKnittedTilesShanten
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

class Error(Exception):
    pass

class MahjongGBEnv():
    
    agent_names = ['player_%d' % i for i in range(1, 5)]

    # 1. 非线性向听数奖励
    SHANTEN_DECREASE_REWARD_BASE = 0.5  # 降低向听数的基础奖励
    SHANTEN_INCREASE_PENALTY_BASE = -0.5 # 增加向听数的基础惩罚
    SHANTEN_NON_LINEAR_FACTOR = 1.5      # 非线性因子, >1.0, 越大则越接近听牌奖励越高

    # 2. 牌型潜力奖励
    FAN_POTENTIAL_REWARD = 0.25          # 每降低一个大牌牌型的向听数，获得的额外奖励

    # 3. 精炼副露惩罚
    MELD_PENALTY_BASE = -0.2             # 吃/碰的基础惩罚
    MELD_SHANTEN_BONUS = 0.4             # 副露后每降低1向听数，获得的奖励（用于抵消惩罚）

    # 4. 防守奖励
    SAFE_TILE_REWARD = 0.05              # 打出安全牌（现物）的奖励
    DANGEROUS_TILE_PENALTY = -0.02       # 打出非安全牌的惩罚
    IMMEDIATE_RON_PENALTY_FACTOR = 0.1   # 即时放铳惩罚 = -番数 * 此因子
    
    # 原有奖励参数 (保留或调整)
    EFFECTIVE_TILES_REWARD = 0.01        # 每增加一张有效牌的奖励

    def __init__(self, config):
        assert 'agent_clz' in config, "must specify agent_clz to process features!"
        self.agentclz = config['agent_clz']
        assert issubclass(self.agentclz, MahjongGBAgent), "ageng_clz must be a subclass of MahjongGBAgent!"
        self.duplicate = config.get('duplicate', True)
        self.variety = config.get('variety', -1)
        self.r = random.Random()
        self.normalizeReward = config.get('reward_norm', False)
        self.observation_space = self.agentclz.observation_space
        self.action_space = self.agentclz.action_space

    def _calculate_flush_shanten(self, hand, packs, suit, is_full_flush):
        non_flush_tiles = 0
        all_tiles = list(hand)
        for _, tile, _ in packs:
            if tile[0] == suit:
                all_tiles.extend([tile] * 3)
            else:
                all_tiles.extend([tile] * 3)
        
        for tile in all_tiles:
            if tile[0] != suit:
                if is_full_flush:
                    non_flush_tiles += 1
                elif tile[0] not in 'FJ':
                    non_flush_tiles += 1
        return non_flush_tiles

    def _get_hand_details(self, player_id):
        agent = self.agents[player_id]
        hand = tuple(agent.hand)
        
        valid_packs = [p for p in agent.packs[player_id] if p[1] != 'CONCEALED']
        packs = tuple(valid_packs)
        
        details = {
            'main_shanten': 9,
            'effective_tiles': 0,
            'fan_shantens': {}
        }

        try:
            main_shanten = MahjongShanten(packs, hand)
            details['main_shanten'] = main_shanten
            
            _, u = RegularShanten(hand)
            details['effective_tiles'] = len(set(u))

            details['fan_shantens']['AllPungs'] = RegularShanten(hand, chinhitsu_mode=False)[0]
            if not packs and len(hand) == 13:
                 details['fan_shantens']['SevenPairs'] = SevenPairsShanten(hand)[0]
            else:
                 details['fan_shantens']['SevenPairs'] = 9

            hand_suits = {t[0] for t in hand if t[0] in 'WTB'}
            pack_suits = {p[1][0] for p in packs if p[1][0] in 'WTB'}
            all_suits = hand_suits.union(pack_suits)
            if len(all_suits) == 1:
                suit = all_suits.pop()
                details['fan_shantens']['FullFlush'] = self._calculate_flush_shanten(hand, packs, suit, is_full_flush=True)
                details['fan_shantens']['HalfFlush'] = self._calculate_flush_shanten(hand, packs, suit, is_full_flush=False)
            else:
                details['fan_shantens']['FullFlush'] = 9
                details['fan_shantens']['HalfFlush'] = 9

        except (TypeError, IndexError):
            return details
            
        return details

    def _is_tile_safe(self, tile, player_id):
        for i in range(4):
            if i == player_id:
                continue
            if tile in self.discards[i]:
                return True
        return False
    
    def reset(self, prevalentWind = -1, tileWall = ''):
        self.agents = [self.agentclz(i) for i in range(4)]
        self.reward = None
        self.done = False
        self.step_rewards = [0, 0, 0, 0]
        
        self.shantens = [9] * 4
        self.hand_details = [{} for _ in range(4)] 
        self.discards = [[] for _ in range(4)]

        if self.variety > 0:
            random.seed(self.r.randint(0, self.variety - 1))
        self.prevalentWind = random.randint(0, 3) if prevalentWind < 0 else prevalentWind
        for agent in self.agents:
            agent.request2obs('Wind %d' % self.prevalentWind)
        if tileWall:
            self.tileWall = tileWall.split()
        else:
            self.tileWall = []
            for j in range(4):
                for i in range(1, 10):
                    self.tileWall.append('W' + str(i))
                    self.tileWall.append('B' + str(i))
                    self.tileWall.append('T' + str(i))
                for i in range(1, 5):
                    self.tileWall.append('F' + str(i))
                for i in range(1, 4):
                    self.tileWall.append('J' + str(i))
            random.shuffle(self.tileWall)
        self.originalTileWall = ' '.join(self.tileWall)
        if self.duplicate:
            self.tileWall = [self.tileWall[i * 34 : (i + 1) * 34] for i in range(4)]
        self.shownTiles = defaultdict(int)
        
        self._deal()

        for i in range(4):
            self.hand_details[i] = self._get_hand_details(i)
            self.shantens[i] = self.hand_details[i]['main_shanten']

        return self._obs()
    
    def step(self, action_dict):
        self.step_rewards = [0, 0, 0, 0]
        player_to_update = -1
        before_details = None

        try:
            if self.state == 0:
                player_to_update = self.curPlayer
                before_details = self.hand_details[player_to_update]
                
                response = self.agents[self.curPlayer].action2response(action_dict[self.agent_names[self.curPlayer]]).split()
                if response[0] == 'Play':
                    self._discard(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
                self.isAboutKong = False
            elif self.state == 1:
                player_to_update = self.curPlayer
                before_details = self.hand_details[player_to_update]

                response = self.agents[self.curPlayer].action2response(action_dict[self.agent_names[self.curPlayer]]).split()
                if response[0] == 'Hu':
                    self.shownTiles[self.curTile] += 1
                    self._checkMahjong(self.curPlayer, isSelfDrawn = True, isAboutKong = self.isAboutKong)
                elif response[0] == 'Play':
                    self.hands[self.curPlayer].append(self.curTile)
                    self._discard(self.curPlayer, response[1])
                elif response[0] == 'Gang' and not self.myWallLast and not self.wallLast:
                    self._concealedKong(self.curPlayer, response[1])
                elif response[0] == 'BuGang' and not self.myWallLast and not self.wallLast:
                    self._promoteKong(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
            elif self.state == 2:
                responses = {i : self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if i != self.curPlayer}
                t = {i : responses[i].split() for i in responses}
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if t[i][0] == 'Hu':
                        self._checkMahjong(i)
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if t[i][0] == 'Gang' and self._canDrawTile(i) and not self.wallLast:
                            player_to_update = i
                            before_details = self.hand_details[i]
                            self._kong(i, self.curTile)
                            break
                        elif t[i][0] == 'Peng' and not self.wallLast:
                            player_to_update = i
                            before_details = self.hand_details[i]
                            self._pung(i, self.curTile)
                            break
                    else:
                        i = (self.curPlayer + 1) % 4
                        if t[i][0] == 'Chi' and not self.wallLast:
                            player_to_update = i
                            before_details = self.hand_details[i]
                            self._chow(i, t[i][1])
                        else:
                            for j in range(1, 4):
                                i = (self.curPlayer + j) % 4
                                if t[i][0] != 'Pass': raise Error(i)
                            if self.wallLast:
                                self.obs = {i : self.agents[i].request2obs('Huang') for i in range(4)}
                                self.reward = [0, 0, 0, 0]
                                self.done = True
                            else:
                                self.curPlayer = (self.curPlayer + 1) % 4
                                self._draw(self.curPlayer)
            elif self.state == 3:
                responses = {i : self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if i != self.curPlayer}
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if responses[i] == 'Hu':
                        self._checkMahjong(i, isAboutKong = True)
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if responses[i] != 'Pass': raise Error(i)
                    self._draw(self.curPlayer)

            if player_to_update != -1 and not self.done:
                after_details = self._get_hand_details(player_to_update)

                before_shanten = before_details['main_shanten']
                after_shanten = after_details['main_shanten']
                shanten_diff = before_shanten - after_shanten
                if shanten_diff > 0:
                    reward = shanten_diff * self.SHANTEN_DECREASE_REWARD_BASE * (self.SHANTEN_NON_LINEAR_FACTOR / (after_shanten + 1))
                    self.step_rewards[player_to_update] += reward
                elif shanten_diff < 0:
                    penalty = shanten_diff * abs(self.SHANTEN_INCREASE_PENALTY_BASE)
                    self.step_rewards[player_to_update] += penalty

                eff_tiles_diff = after_details['effective_tiles'] - before_details['effective_tiles']
                self.step_rewards[player_to_update] += eff_tiles_diff * self.EFFECTIVE_TILES_REWARD

                for fan, b_shanten in before_details['fan_shantens'].items():
                    a_shanten = after_details['fan_shantens'].get(fan, 9)
                    if a_shanten < b_shanten:
                        self.step_rewards[player_to_update] += self.FAN_POTENTIAL_REWARD

                self.hand_details[player_to_update] = after_details
                self.shantens[player_to_update] = after_shanten

        except Error as e:
            player = e.args[0]
            self.obs = {i : self.agents[i].request2obs('Player %d Invalid' % player) for i in range(4)}
            self.reward = [10] * 4
            self.reward[player] = -30
            self.done = True
        return self._obs(), self._reward(), self._done()
        
    def _obs(self):
        return {self.agent_names[k] : v for k, v in self.obs.items()}
    
    def _reward(self):
        current_rewards = {self.agent_names[i]: self.step_rewards[i] for i in range(4)}
        if self.reward:
            for i in range(4):
                current_rewards[self.agent_names[i]] += self.reward[i]
        return current_rewards
    
    def _done(self):
        return self.done
    
    def _drawTile(self, player):
        if self.duplicate:
            return self.tileWall[player].pop()
        return self.tileWall.pop()
    
    def _canDrawTile(self, player):
        if self.duplicate:
            return bool(self.tileWall[player])
        return bool(self.tileWall)
    
    def _deal(self):
        self.hands = []
        self.packs = []
        for i in range(4):
            hand = []
            while len(hand) < 13:
                tile = self._drawTile(i)
                hand.append(tile)
            self.hands.append(hand)
            self.packs.append([])
            self.agents[i].request2obs(' '.join(['Deal', *hand]))

        self.curPlayer = 0
        self.drawAboutKong = False
        self._draw(self.curPlayer)
    
    def _draw(self, player):
        tile = self._drawTile(player)
        self.myWallLast = not self._canDrawTile(player)
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.isAboutKong = self.drawAboutKong
        self.drawAboutKong = False
        self.state = 1
        self.curTile = tile
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Draw' % player)
        self.obs = {player : self.agents[player].request2obs('Draw %s' % tile)}
    
    def _discard(self, player, tile):
        if tile not in self.hands[player]: raise Error(player)
        
        if self._is_tile_safe(tile, player):
            self.step_rewards[player] += self.SAFE_TILE_REWARD
        else:
            self.step_rewards[player] += self.DANGEROUS_TILE_PENALTY

        self.hands[player].remove(tile)
        self.discards[player].append(tile)
        self.shownTiles[tile] += 1
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.curTile = tile
        self.state = 2
        self.agents[player].request2obs('Player %d Play %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d Play %s' % (player, tile)) for i in range(4) if i != player}
    
    def _kong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for agent in self.agents:
            agent.request2obs('Player %d Gang' % player)
        self._draw(player)
    
    def _pung(self, player, tile):
        self.step_rewards[player] += self.MELD_PENALTY_BASE
        shanten_diff = self.hand_details[player]['main_shanten'] - self._get_hand_details(player)['main_shanten']
        if shanten_diff > 0:
             self.step_rewards[player] += shanten_diff * self.MELD_SHANTEN_BONUS

        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 3: raise Error(player)
        for i in range(3): self.hands[player].remove(tile)
        self.packs[player].append(('PENG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] += 2
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Peng' % player)
        self.obs = {player : self.agents[player].request2obs('Player %d Peng' % player)}
    
    def _chow(self, player, tile):
        self.step_rewards[player] += self.MELD_PENALTY_BASE
        shanten_diff = self.hand_details[player]['main_shanten'] - self._get_hand_details(player)['main_shanten']
        if shanten_diff > 0:
             self.step_rewards[player] += shanten_diff * self.MELD_SHANTEN_BONUS

        self.hands[player].append(self.curTile)
        self.shownTiles[self.curTile] -= 1
        color = tile[0]
        num = int(tile[1])
        for i in range(-1, 2):
            t = color + str(num + i)
            if t not in self.hands[player]: raise Error(player)
            self.hands[player].remove(t)
            self.shownTiles[t] += 1
        self.packs[player].append(('CHI', tile, int(self.curTile[1]) - num + 2))
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Chi %s' % (player, tile))
        self.obs = {player : self.agents[player].request2obs('Player %d Chi %s' % (player, tile))}
    
    def _concealedKong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d AnGang' % player)
        self.agents[player].request2obs('Player %d AnGang %s' % (player, tile))
        self._draw(player)
    
    def _promoteKong(self, player, tile):
        self.hands[player].append(self.curTile)
        idx = -1
        for i in range(len(self.packs[player])):
            if self.packs[player][i][0] == 'PENG' and self.packs[player][i][1] == tile:
                idx = i
        if idx < 0: raise Error(player)
        self.hands[player].remove(tile)
        offer = self.packs[player][idx][2]
        self.packs[player][idx] = ('GANG', tile, offer)
        self.shownTiles[tile] = 4
        self.state = 3
        self.curPlayer = player
        self.curTile = tile
        self.drawAboutKong = True
        self.isAboutKong = False
        self.agents[player].request2obs('Player %d BuGang %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d BuGang %s' % (player, tile)) for i in range(4) if i != player}
    
    def _checkMahjong(self, player, isSelfDrawn = False, isAboutKong = False):
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[player]),
                hand = tuple(self.hands[player]),
                winTile = self.curTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = (self.shownTiles[self.curTile] + isSelfDrawn) == 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = player,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Exception('Not Enough Fans')
            self.obs = {i : self.agents[i].request2obs('Player %d Hu' % player) for i in range(4)}
            
            # Final (terminal) reward
            if isSelfDrawn:
                self.reward = [-(8 + fanCnt)] * 4
                self.reward[player] = (8 + fanCnt) * 3
            else:
                self.reward = [-8] * 4
                self.reward[player] = 8 * 3 + fanCnt
                self.reward[self.curPlayer] -= fanCnt
                self.step_rewards[self.curPlayer] -= fanCnt * self.IMMEDIATE_RON_PENALTY_FACTOR
            self.done = True
        except Exception as e:
            raise Error(player)