from gym.wrappers import Monitor
from racetrack_env import RaceTrackEnv

import argparse
import numpy as np
import tensorflow.keras as keras

import os
import gym
import matplotlib.pyplot as plt

from agent.DQN import DQNAgent, CDQNAgent
from agent.PPO import PPOAgent
from agent.DDPG import DDPGAgent
from agent.A3C import A3CAgent


GET_AGENT = {
    "DQN" : DQNAgent,
    "CDQN": CDQNAgent,
    "PPO" : PPOAgent,
    "DDPG": DDPGAgent,
    "A3C" : A3CAgent
}

DISCRETE_ACTION_SPACE = {
        0: [1 ,-1], 1: [1 , 0], 2: [1 , 1],
        3: [0 ,-1], 4: [0 , 0], 5: [0 , 1],
        6: [-1,-1], 7: [-1, 0], 8: [-1, 1]
    }

SIMPLE_DISCRETE_ACTION_SPACE = {
        0: [-0.5], 1 : [0], 2 : [0.5]
    }


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # Configuration Settings
        self.parser.add_argument('--mode', default='train', help='Train, Test, Manual')
        self.parser.add_argument('--agent', default='PPO', help='DQN, DDPG, A3C, PPO')
        self.parser.add_argument('--exp_id', default='default', help='Unique Experiment Name for Saving Logs & Models')
        self.parser.add_argument('--resume', action='store_true', help='Whether to Load Last Model for Further Training')
        self.parser.add_argument('--load_model', default=None, help='Model to load for Testing')
        self.parser.add_argument('--save_model', default=True, help='Whether to Save Model during Training')
        self.parser.add_argument('--save_video', action='store_true', help='Saves Env Render as Video')

        # Neural Network Settings
        self.parser.add_argument('--arch', default='Identity', help='Neural Net Backbone')
        self.parser.add_argument('--fc_layers', default=2, type=int, help='Number of Dense Layers')
        self.parser.add_argument('--fc_width', default=256, type=int, help='Number of Channels in Dense Layers')

        # Problem Space Settings
        self.parser.add_argument('--obs_dim', default=(2,18,18), type=int, nargs=3, help='Agent Observation Space')
        self.parser.add_argument('--num_actions', default=1, type=int, help='Agent Action Space')
        self.parser.add_argument('--offroad_thres', default=-1, type=int, help='Number of Steps Agent is Allowed to Ride Offroad')
        self.parser.add_argument('--spawn_vehicles', default=0, type=int, help='Number of Non-Agent Vehicles to Spawn, Set 0 to Disable')
        self.parser.add_argument('--all_random', action='store_true', help='Whether to Train on All Random Vehicles')
        self.parser.add_argument('--random_obstacles', default=0, type=int, help='Number of Static Obstacles to Spawn (Unused)')

        # Experiment Settings
        self.parser.add_argument('--num_episodes', default=10000, type=int, help='Number of Episodes to Train')
        self.parser.add_argument('--log_freq', default=20, type=int, help='Frequency of Logging (Episodes)')
        self.parser.add_argument('--eval_freq', default=100, type=int, help='Frequency to Run Evaluation Runs')
        self.parser.add_argument('--min_reward', default=50, type=int, help='Minimum Reward to Save Model')

        # Hyperparameters
        self.parser.add_argument('--lr', default=5e-4, type=float, help='Policy Learning Rate')
        self.parser.add_argument('--lr_decay', action='store_true', help='Whether to Decay Learning Rate')
        self.parser.add_argument('--batch_size', default=64, type=int, help='Policy Update Batch Size')
        self.parser.add_argument('--num_epochs', default=10, type=int, help='Num Epochs for Policy Gradient')

        # DQN Hyperparameters
        self.parser.add_argument('--epsilon', default=1, type=float, help='Initial Value of Epsilon')
        self.parser.add_argument('--epsilon_decay', default=0.9995, type=float, help='Decay Ratio of Epsilon')
        self.parser.add_argument('--min_epsilon', default=0, type=float, help='Minimum Value of Epsilon')
        self.parser.add_argument('--dqn_gamma', default=0.99, type=float, help='Frequency of Updating Target Model')
        self.parser.add_argument('--update_freq', default=20, type=int, help='Frequency of Updating Target Model')
        self.parser.add_argument('--replay_size', default=10000, type=int, help='Size of the Replay Memory Buffer')
        self.parser.add_argument('--min_replay_size', default=500, type=int, help='Minimum Memory Entries before Training')

        # PPO Hyperparameters
        self.parser.add_argument('--gae_lambda', default=0.95, type=float, help='Generalised Advantage Estimate Lambda')
        self.parser.add_argument('--gae_gamma', default=0.9, type=float, help='Generalised Advantage Estimate Gamma')
        self.parser.add_argument('--ppo_epsilon', default=0.2, type=float, help='Clipping Loss Epsilon')
        self.parser.add_argument('--target_kl', default=None, type=float, help='Max KL Divergence for Training Sequence')
        self.parser.add_argument('--ppo_memory_size', default=2048, type=int, help='Size of Replay Buffer for PPO')
        
        # A3C Hyperparameters
        self.parser.add_argument('--a3c_gamma', default=0.99, type=float, help='Generalised Advantage Estimate Gamma')
        self.parser.add_argument('--rmsprop_epsilon', default=1e-5, type=float, help='RMSProp epsilon')
        self.parser.add_argument('--update_global_freq', default=5, type=int, help='Frequency of Updating Master Agent')
        self.parser.add_argument('--num_workers', default=None, type=int, help='Number of Workers')
        
        # DDPG Hyperparameters
        self.parser.add_argument('--ddpg_actor_lr', default=0.001, type=float, help='DDPG Actor Learning Rate')
        self.parser.add_argument('--ddpg_critic_lr', default=0.002, type=float, help='DDPG Critic Learning Rate')
        self.parser.add_argument('--ddpg_gamma', default=0.99, type=float, help='Generalised Advantage Estimate Gamma')
        self.parser.add_argument('--ddpg_tau', default=0.005, type=float, help='Soft Update Coefficient')
        self.parser.add_argument('--ddpg_best', default=False, type=bool, help='Get Best Scoring Model For DDPG')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
            
        opt.exp_dir = f"./logs/{opt.exp_id}"
        if not os.path.exists(opt.exp_dir):
            os.mkdir(opt.exp_dir)

        return opt


def display_observations(obs):
    """Display Grayscale Observation Plots"""
    obs = np.array([o.T for o in obs])
    _, axes = plt.subplots(ncols=2, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
       ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == "__main__":
    
    # Parse Arguments
    opt = opts().parse()
    print(opt)

    # Set up Environment According to Debug Mode
    env = RaceTrackEnv(opt)

    # For Recording or Visualisation
    if opt.save_video:
        env = Monitor(env, f'./videos/{opt.agent}/', force=True)

    if opt.mode == "train":
        
        agent = GET_AGENT[opt.agent](opt=opt)
        agent.learn(env, opt)

    elif opt.mode == "test":
        
        model = keras.models.load_model(opt.load_model)
        total_reward, obs, seq = 0, env.reset(), []
        
        if opt.agent in ["DQN", "CDQN"]:
            
            done = False
            while not done:
                action_idx = model.predict(np.array([obs])/255)[0]
                action_idx = np.argmax(action_idx)
                obs, reward, done, _ = env.step(DISCRETE_ACTION_SPACE[action_idx] if opt.num_actions == 2 else
                                                SIMPLE_DISCRETE_ACTION_SPACE[action_idx])
                total_reward += reward
                print(reward)
            print("Total Reward: ", total_reward)
            
        elif opt.agent in ["DDPG"]:

            agent = GET_AGENT[opt.agent](opt=opt)
            agent.initialize_networks(obs)
            if opt.ddpg_best == True:
                agent.load_best()
            else:
                agent.load_models()

            done = False
            while not done:
                action = agent.select_action(np.expand_dims(obs/255, axis=0), env, test_model=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                print(reward)
            print("Total Reward:", total_reward)

        else:
            
            done = False            
            while not done:
                action = model(np.array([obs]))[0]
                obs, reward, done, info = env.step(action)
                total_reward += reward
                print(reward)
            print("Total Reward: ", total_reward)
            
    elif opt.mode == "manual":
        
        env.configure({"manual_control": True})
        obs = env.reset()
        # display_observations(obs)
        total_reward, done = 0, False
        
        while not done:
            obs, reward, done, _ = env.step(env.action_space.sample())
            total_reward += reward
        print("Total Reward: ", total_reward)