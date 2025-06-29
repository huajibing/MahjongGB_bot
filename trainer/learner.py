# learner.py

from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F
import os
# --- START OF MODIFICATION ---
from torch.utils.tensorboard import SummaryWriter
# --- END OF MODIFICATION ---

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel

class Learner(Process):

    def __init__(self, config, replay_buffer, metrics_queue):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.metrics_queue = metrics_queue
    
    def run(self):
        writer = SummaryWriter(self.config['log_path'])

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
        total_episodes_processed = 0

        try:
            while True:
                while not self.metrics_queue.empty():
                    reward = self.metrics_queue.get()
                    writer.add_scalar('Reward/episode_reward', reward, total_episodes_processed)
                    total_episodes_processed += 1
                
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
                
                total_value_loss = 0.0
                total_policy_loss = 0.0
                total_entropy = 0.0
                total_imitation_loss = 0.0

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
                    
                    entropy = action_dist.entropy().mean()
                    entropy_loss = -entropy

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

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    total_imitation_loss += imitation_loss.item()
                
                avg_policy_loss = total_policy_loss / self.config['epochs']
                avg_value_loss = total_value_loss / self.config['epochs']
                avg_entropy = total_entropy / self.config['epochs']
                avg_imitation_loss = total_imitation_loss / self.config['epochs']
                
                print(
                    f"Iter: {iterations:6d} | "
                    f"P_Loss: {avg_policy_loss:<8.4f} | "
                    f"V_Loss: {avg_value_loss:<8.4f} | "
                    f"Entropy: {avg_entropy:<8.4f} | "
                    f"I_Loss: {avg_imitation_loss:<8.4f} | "
                    f"Buffer: {self.replay_buffer.size()}"
                )
                
                # Log metrics to TensorBoard
                writer.add_scalar('Loss/policy_loss', avg_policy_loss, iterations)
                writer.add_scalar('Loss/value_loss', avg_value_loss, iterations)
                writer.add_scalar('Loss/imitation_loss', avg_imitation_loss, iterations)
                writer.add_scalar('Stats/entropy', avg_entropy, iterations)
                writer.add_scalar('Stats/replay_buffer_size', self.replay_buffer.size(), iterations)

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
        finally:
            writer.close()