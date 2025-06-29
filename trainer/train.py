from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
from multiprocessing import Queue
import os
from datetime import datetime


if __name__ == '__main__':
    run_name = datetime.now().strftime('%Y%m%d-%H%M%S')

    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',
        'num_actors': 12,
        'episodes_per_actor': 2000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 256,
        'epochs': 5,
        'clip': 0.2,
        'lr': 1e-4,
        'value_coeff': 1,
        'entropy_coeff': 1e-5,
        'imitation_coeff': 5,
        'device': 'cuda',
        'ckpt_save_interval': 50,
        'ckpt_save_path': 'model/',
        'log_path': f'runs/mahjong_rl_{run_name}'
    }
    
    if not os.path.exists(config['ckpt_save_path']):
        os.makedirs(config['ckpt_save_path'])
    if not os.path.exists(config['log_path']):
        os.makedirs(config['log_path'])

    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    metrics_queue = Queue()
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer, metrics_queue)
        actors.append(actor)

    learner = Learner(config, replay_buffer, metrics_queue)
    
    for actor in actors: actor.start()
    learner.start()
    
    for actor in actors: actor.join()
    learner.terminate()