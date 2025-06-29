from multiprocessing import Process
import numpy as np
import torch
from collections import deque

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel

class Actor(Process):

    def __init__(self, config, replay_buffer, metrics_queue):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        self.metrics_queue = metrics_queue

    def run(self):
        torch.set_num_threads(1)

        # connect to model pool
        model_pool = ModelPoolClient(self.config['model_pool_name'])

        # create network model
        model = CNNModel()

        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        model.load_state_dict(state_dict)

        # collect data
        env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
        policies = {player : model for player in env.agent_names}

        recent_episode_rewards = deque(maxlen=100)

        for episode in range(self.config['episodes_per_actor']):
            # update model
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                model.load_state_dict(state_dict)
                version = latest

            # run one episode and collect data
            obs = env.reset()
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'reward' : [],
                'value' : []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                # each player take action
                actions = {}
                values = {}
                for agent_name in obs:
                    agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(state['observation'])
                    agent_data['state']['action_mask'].append(state['action_mask'])
                    state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
                    state['action_mask'] = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0)
                    model.train(False)
                    with torch.no_grad():
                        logits, value = model(state)
                        action_dist = torch.distributions.Categorical(logits = logits)
                        action = action_dist.sample().item()
                        value = value.item()
                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(actions[agent_name])
                    agent_data['value'].append(values[agent_name])
                # interact with env
                next_obs, rewards, done = env.step(actions)
                
                for agent_name in obs:
                    if agent_name in rewards:
                        episode_data[agent_name]['reward'].append(rewards[agent_name])

                obs = next_obs
            
            episode_reward = rewards.get('player_1', 0)
            recent_episode_rewards.append(episode_reward)
            avg_reward = sum(recent_episode_rewards) / len(recent_episode_rewards)
            
            print(f"{self.name} | Episode: {episode}, Model: {latest['id']}, "
                  f"Episode Reward(P1): {episode_reward:.2f}, "
                  f"Avg Reward(100 episodes): {avg_reward:.2f}")

            try:
                self.metrics_queue.put(episode_reward, block=False)
            except:
                pass

            for agent_name, agent_data in episode_data.items():
                if len(agent_data['action']) < len(agent_data['reward']):
                    agent_data['reward'].pop(0)
                
                if not agent_data['action']:
                    continue

                obs_arr = np.stack(agent_data['state']['observation'])
                mask = np.stack(agent_data['state']['action_mask'])
                actions_arr = np.array(agent_data['action'], dtype = np.int64)
                rewards_arr = np.array(agent_data['reward'], dtype = np.float32)
                values_arr = np.array(agent_data['value'], dtype = np.float32)

                assert len(rewards_arr) == len(values_arr), \
                    f"Mismatch for {agent_name}: rewards {len(rewards_arr)}, values {len(values_arr)}"

                next_values = np.array(agent_data['value'][1:] + [0], dtype = np.float32)

                td_target = rewards_arr + next_values * self.config['gamma']
                td_delta = td_target - values_arr
                advs = []
                adv = 0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv) # GAE
                advs.reverse()
                advantages = np.array(advs, dtype = np.float32)

                if rewards_arr[-1] != 0:
                    self.replay_buffer.push({
                        'state': {
                            'observation': obs_arr,
                            'action_mask': mask
                        },
                        'action': actions_arr,
                        'adv': advantages,
                        'target': td_target
                    })