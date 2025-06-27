from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_right
import random


class MahjongGBDataset(Dataset):
    
    def __init__(self, begin = 0, end = 1, augment = False, data_dir_prefix = 'data/'): # Added data_dir_prefix
        import json
        if not data_dir_prefix.endswith('/'):
            data_dir_prefix += '/'

        count_json_path = data_dir_prefix + 'count.json'
        with open(count_json_path) as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        self.cache = {'obs': [], 'mask': [], 'act': [], 'outcome': []}
        for i in range(self.matches):
            if i % 128 == 0: print('loading', i)
            npz_path = data_dir_prefix + '%d.npz' % (i + self.begin) # Use data_dir_prefix
            d = np.load(npz_path)
            for k in d:
                self.cache[k].append(d[k])
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        return (
            self.cache['obs'][match_id][sample_id], 
            self.cache['mask'][match_id][sample_id], 
            self.cache['act'][match_id][sample_id],
            self.cache['outcome'][match_id][sample_id]
        )


class AugmentedMahjongGBDataset(Dataset):
    def __init__(self, original_dataset, augmentation_factor=1):
        self.original_dataset = original_dataset
        self.augmentation_factor = augmentation_factor
        self.total_samples = len(original_dataset) * (1 + augmentation_factor)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        if index < len(self.original_dataset):
            return self.original_dataset[index]
        else:
            original_index = (index - len(self.original_dataset)) % len(self.original_dataset)
            obs, mask, act, outcome = self.original_dataset[original_index]
            aug_obs, aug_mask, aug_act = self._augment_data(obs, mask, act)
            return aug_obs, aug_mask, aug_act, outcome


    def _augment_data(self, obs, mask, act):
        augmentation_funcs = [
            self._swap_suits,
            self._mirror_numbers,
            self._shuffle_hand,
            self._rotate_winds
        ]
        
        freq = [1, 0.5, 0.5, 0]
        for i, func in enumerate(augmentation_funcs):
            if random.random() < freq[i]:
                obs, mask, act = func(obs, mask, act)
        
        return obs, mask, act

    def _swap_suits(self, obs, mask, act):
        order = list(range(3))
        random.shuffle(order)

        obs_indices = [slice(i, i+1) for i in range(3)]
        mask_indices = {
            'play': [slice(2+i*9, 11+i*9) for i in range(3)],
            'chi': [slice(36+i*21, 57+i*21) for i in range(3)],
            'peng': [slice(99+i*9, 108+i*9) for i in range(3)],
            'gang': [slice(133+i*9, 142+i*9) for i in range(3)],
            'angang': [slice(167+i*9, 176+i*9) for i in range(3)],
            'bugang': [slice(201+i*9, 210+i*9) for i in range(3)]
        }

        new_obs = np.concatenate([obs[:, obs_indices[order.index(i)]] for i in range(3)] + [obs[:, 3:]], axis=1)

        new_mask = np.concatenate([
            mask[:2],
            *[mask[mask_indices['play'][order.index(i)]] for i in range(3)],
            mask[29:36],
            *[mask[mask_indices['chi'][order.index(i)]] for i in range(3)],
            *[mask[mask_indices['peng'][order.index(i)]] for i in range(3)],
            mask[126:133],
            *[mask[mask_indices['gang'][order.index(i)]] for i in range(3)],
            mask[160:167],
            *[mask[mask_indices['angang'][order.index(i)]] for i in range(3)],
            mask[194:201],
            *[mask[mask_indices['bugang'][order.index(i)]] for i in range(3)],
            mask[228:]
        ])

        if 2 <= act < 29:
            act = 2 + order[(act - 2) // 9] * 9 + (act - 2) % 9
        elif 36 <= act < 99:
            act = 36 + order[(act - 36) // 21] * 21 + (act - 36) % 21
        elif 99 <= act < 126 or 133 <= act < 160 or 167 <= act < 194 or 201 <= act < 228:
            base = 99 if act < 126 else 133 if act < 160 else 167 if act < 194 else 201
            act = base + order[(act - base) // 9] * 9 + (act - base) % 9
        
        return new_obs, new_mask, act

    def _mirror_numbers(self, obs, mask, act):
        for i in range(3):
            obs[:, i] = obs[:, i][:, ::-1]

        for start in [2, 99, 133, 167, 201]:
            for i in range(3):
                slice_start = start + i * 9
                slice_end = slice_start + 9
                mask[slice_start:slice_end] = mask[slice_start:slice_end][::-1]
        mask_chi_W = [mask[36:39], mask[39:42], mask[42:45], mask[45:48], mask[48:51], mask[51:54], mask[54:57]]
        mask_chi_T = [mask[57:60], mask[60:63], mask[63:66], mask[66:69], mask[69:72], mask[72:75], mask[75:78]]
        mask_chi_B = [mask[78:81], mask[81:84], mask[84:87], mask[87:90], mask[90:93], mask[93:96], mask[96:99]]
        mask[36:99] = np.concatenate(mask_chi_W[::-1]+mask_chi_T[::-1]+mask_chi_B[::-1])

        if 2 <= act < 29:
            act = 2 + (act - 2) // 9 * 9 + (8 - (act - 2) % 9)
        elif 36 <= act < 99:
            index = ((act - 36) % 21) // 3
            offset = [(6 - 2*i) * 3 for i in range(7)]
            act = act + offset[index]
        elif act in range(99, 126) or act in range(133, 160) or act in range(167, 194) or act in range(201, 228):
            base = 99 if act < 126 else 133 if act < 160 else 167 if act < 194 else 201
            act = base + (act - base) // 9 * 9 + (8 - (act - base) % 9)
        
        return obs, mask, act

    def _shuffle_hand(self, obs, mask, act):
        hand = obs[2:6]
        non_zero = np.nonzero(hand)
        values = hand[non_zero]
        np.random.shuffle(values)
        hand[non_zero] = values
        obs[2:6] = hand

        return obs, mask, act

    def _rotate_winds(self, obs, mask, act):
        wind_indices = [0, 1, 2, 3]
        rotation = random.randint(0, 3)
        rotated_winds = np.roll(obs[:, 3:4, wind_indices], rotation, axis=2)
        obs[:, 3:4, wind_indices] = rotated_winds

        wind_mask_indices = [29, 30, 31, 32, 126, 127, 128, 129, 160, 161, 162, 163, 194, 195, 196, 197, 228, 229, 230, 231]
        rotated_wind_mask = np.roll(mask[wind_mask_indices], rotation * 4)
        mask[wind_mask_indices] = rotated_wind_mask

        wind_act_ranges = [(29, 33), (126, 130), (160, 164), (194, 198), (228, 232)]
        for start, end in wind_act_ranges:
            if start <= act < end:
                act = start + (act - start + rotation) % 4
                break
        
        return obs, mask, act