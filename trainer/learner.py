from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F
import os

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
    
    def run(self):
        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        
        # initialize model params
        device = torch.device(self.config['device'])
        model = CNNModel()
        expert_model = CNNModel()

        pretrained_model_path = os.path.join(self.config.get('ckpt_save_path', 'model/'), 'pretrained_rl_cnn.pkl')
        if os.path.exists(pretrained_model_path):
            try:
                print(f"Loading pre-trained model from {pretrained_model_path}")
                state_dict = torch.load(pretrained_model_path, map_location='cpu')
                
                model.load_state_dict(state_dict, strict=False)
                expert_model.load_state_dict(state_dict, strict=False)
                
                print("Successfully loaded pre-trained model weights to both RL and Expert models.")
            except Exception as e:
                print(f"Error loading pre-trained model: {e}. Starting with fresh weights.")
        else:
            print(f"No pre-trained model found at {pretrained_model_path}. Starting with fresh weights.")
        
        # send to model pool
        model_pool.push(model.state_dict())
        model = model.to(device)
        
        expert_model = expert_model.to(device)
        expert_model.eval()
        for param in expert_model.parameters():
            param.requires_grad = False
        print("Expert model is set to eval mode and its parameters are frozen.")

        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'])
        
        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)
        
        cur_time = time.time()
        iterations = 0
        while True:
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            states = {
                'observation': obs,
                'action_mask': mask
            }
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
            advs = torch.tensor(batch['adv']).to(device)
            targets = torch.tensor(batch['target']).to(device)
            
            print('Iteration %d, replay buffer in %d out %d' % (iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']))
            
            # calculate PPO loss
            model.train(True)
            old_logits, _ = model(states)
            old_probs = F.softmax(old_logits, dim = 1).gather(1, actions)
            old_log_probs = torch.log(old_probs + 1e-8).detach()
            for _ in range(self.config['epochs']):
                logits, values = model(states)
                action_dist = torch.distributions.Categorical(logits = logits)
                probs = F.softmax(logits, dim = 1).gather(1, actions)
                log_probs = torch.log(probs + 1e-8)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())

                with torch.no_grad():
                    expert_logits, _ = expert_model(states)

                rl_log_probs = F.log_softmax(logits, dim=-1)
                expert_probs = F.softmax(expert_logits, dim=-1)

                imitation_loss = F.kl_div(rl_log_probs, expert_probs, reduction='batchmean')
                loss = (policy_loss + 
                        self.config['value_coeff'] * value_loss + 
                        self.config['entropy_coeff'] * entropy_loss +
                        self.config['imitation_coeff'] * imitation_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict())
            model = model.to(device)
            
            # save checkpoints
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                path = self.config['ckpt_save_path'] + 'model_%d.pt' % iterations
                torch.save(model.state_dict(), path)
                cur_time = t
            iterations += 1