from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

class FeatureAgent(MahjongGBAgent):
    OBS_SIZE = 154
    ACT_SIZE = 235

    OFFSET_OBS = {
        'SEAT_WIND' : 0,
        'PREVALENT_WIND' : 1,
        'HAND' : 2,
        'PACKS' : 6,
        'MY_ANGANG': 34,
        'OPPONENT_ANGANG_COUNT': 35,
        'HISTORY' : 38,
        'REMAINING' : 150
    }

    OFFSET_ACT = {
        'Pass' : 0, 'Hu' : 1, 'Play' : 2, 'Chi' : 36, 'Peng' : 99,
        'Gang' : 133, 'AnGang' : 167, 'BuGang' : 201
    }
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)), *('T%d'%(i+1) for i in range(9)),
        *('B%d'%(i+1) for i in range(9)), *('F%d'%(i+1) for i in range(4)),
        *('J%d'%(i+1) for i in range(3))
    ]
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}

    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.packs = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.obs = np.zeros((self.OBS_SIZE, 36))
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1

    def request2obs(self, request):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            self._hand_embedding_update()
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs()

        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == 'Invalid' or t[2] == 'Hu':
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                self._history_embedding_update()
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2): self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            self._packs_embedding_update()
            if p == 0:
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2): self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return

        if t[2] == 'UnChi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            self._packs_embedding_update()
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return

        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            self._packs_embedding_update()
            if p == 0:
                self.valid = []
                for i in range(2): self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return

        if t[2] == 'UnPeng':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            self._packs_embedding_update()
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return

        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            self._packs_embedding_update()
            if p == 0:
                for i in range(3): self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            self._packs_embedding_update()
            if p == 0:
                self.isAboutKong = True
                for i in range(4): self.hand.remove(tile)
                self._hand_embedding_update()
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            self._packs_embedding_update()
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)

    def action2response(self, action):
        if action < self.OFFSET_ACT['Hu']: return 'Pass'
        if action < self.OFFSET_ACT['Play']: return 'Hu'
        if action < self.OFFSET_ACT['Chi']: return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']: return 'Peng'
        if action < self.OFFSET_ACT['AnGang']: return 'Gang'
        if action < self.OFFSET_ACT['BuGang']: return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]
    
    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']

    def _obs(self):
        self._update_true_remaining_tiles_feature()
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid: mask[a] = 1
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }

    def _update_true_remaining_tiles_feature(self):
        self.obs[self.OFFSET_OBS['REMAINING']:] = 0
        off_wall_counts = defaultdict(int)
        for tile in self.hand:
            off_wall_counts[tile] += 1
        for p_packs in self.packs:
            for packType, tile, offer in p_packs:
                if packType == 'CHI':
                    color = tile[0]
                    num = int(tile[1])
                    for i in range(-1, 2):
                        off_wall_counts[color + str(num + i)] += 1
                elif packType == 'PENG':
                    off_wall_counts[tile] += 3
                elif packType == 'GANG':
                    if tile != 'CONCEALED':
                        off_wall_counts[tile] += 4
        for p_history in self.history:
            for tile in p_history:
                off_wall_counts[tile] += 1
        for tile_idx, tile_code in enumerate(self.TILE_LIST):
            count_in_wall = 4 - off_wall_counts[tile_code]
            if count_in_wall >= 1: self.obs[self.OFFSET_OBS['REMAINING'] + 0, tile_idx] = 1
            if count_in_wall >= 2: self.obs[self.OFFSET_OBS['REMAINING'] + 1, tile_idx] = 1
            if count_in_wall >= 3: self.obs[self.OFFSET_OBS['REMAINING'] + 2, tile_idx] = 1
            if count_in_wall >= 4: self.obs[self.OFFSET_OBS['REMAINING'] + 3, tile_idx] = 1

    def _packs_embedding_update(self):
        self.obs[self.OFFSET_OBS['PACKS'] : self.OFFSET_OBS['HISTORY']] = 0
        opponent_angang_counts = {1:0, 2:0, 3:0}
        for p in range(4):
            pack_start = self.OFFSET_OBS['PACKS'] + p * 7
            for packType, tile, offer in self.packs[p]:
                if tile == 'CONCEALED':
                    if p > 0: opponent_angang_counts[p] += 1
                    continue
                tile_offset = self.OFFSET_TILE[tile]
                if packType == 'CHI':
                    self.obs[pack_start + 0, tile_offset] = 1
                elif packType == 'PENG':
                    self.obs[pack_start + 1, tile_offset] = 1
                elif packType == 'GANG':
                    if offer == 0:
                        if p == 0:
                            self.obs[self.OFFSET_OBS['MY_ANGANG'], tile_offset] = 1
                    else:
                        self.obs[pack_start + 2, tile_offset] = 1
                if packType in ('PENG', 'GANG') and offer > 0:
                    source_p_raw = (p - ((offer-1) + (self.seatWind - self.tileFrom -1) %4 +1) %4 + 4) %4
                    if source_p_raw == 1:
                        self.obs[pack_start + 6, tile_offset] = 1
                    elif source_p_raw == 2:
                        self.obs[pack_start + 5, tile_offset] = 1
                    elif source_p_raw == 3:
                        self.obs[pack_start + 4, tile_offset] = 1
        opp_map = {1:2, 2:1, 3:0}
        for p, count in opponent_angang_counts.items():
            channel_idx = self.OFFSET_OBS['OPPONENT_ANGANG_COUNT'] + opp_map[p]
            self.obs[channel_idx, :] = count / 4.0

    def _history_embedding_update(self):
        self.obs[self.OFFSET_OBS['HISTORY'] : self.OFFSET_OBS['REMAINING']] = 0
        PADDING_VALUE = 0

        for p in range(4):
            hist_start = self.OFFSET_OBS['HISTORY'] + p * 28
            history_len = len(self.history[p])

            for i in range(28):
                history_idx = history_len - 1 - i
                channel_idx = hist_start + (28 - 1 - i)

                if history_idx >= 0:
                    tile = self.history[p][history_idx]
                    tile_id = self.OFFSET_TILE[tile] + 1
                    self.obs[channel_idx, 0] = tile_id
                else:
                    self.obs[channel_idx, 0] = PADDING_VALUE

    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['PACKS']] = 0
        d = defaultdict(int)
        for tile in self.hand: d[tile] += 1
        for tile in d:
            self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1

    def _check_mahjong(self, winTile, isSelfDrawn = False, isAboutKong = False):
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]), hand = tuple(self.hand), winTile = winTile, flowerCount = 0,
                isSelfDrawn = isSelfDrawn, is4thTile = self.shownTiles[winTile] == 4, isAboutKong = isAboutKong,
                isWallLast = self.wallLast, seatWind = self.seatWind, prevalentWind = self.prevalentWind, verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans: fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Exception('Not Enough Fans')
        except:
            return False
        return True