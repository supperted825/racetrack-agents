from tqdm import tqdm
from gym.wrappers import Monitor
from racetrack_env import RaceTrackEnv, RaceTrackEnv2

import argparse
import numpy as np
import tensorflow.keras as keras

import os
import gym
import matplotlib.pyplot as plt

from agent.DQN import DQNAgent, CDQNAgent
from agent.PPO import PPOAgent


GET_AGENT = {
    "DQN" : DQNAgent,
    "CDQN": CDQNAgent,
    "PPO" : PPOAgent,
}

DISCRETE_ACTION_SPACE = {
        0: [1 ,-1], 1: [1 , 0], 2: [1 , 1],
        3: [0 ,-1], 4: [0 , 0], 5: [0 , 1],
        6: [-1,-1], 7: [-1, 0], 8: [-1, 1]
    }

SIMPLE_DISCRETE_ACTION_SPACE = {
        0: [-1], 1 : [0], 2 : [1]
    }


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # Configuration Settings
        self.parser.add_argument('--mode', default='train', help='Train, Test, Manual')
        self.parser.add_argument('--agent', default='PPO', help='DQN, DDPG, PPO')
        self.parser.add_argument('--exp_id', default='default', help='Unique Experiment Name for Saving Logs & Models')
        self.parser.add_argument('--debug', default=0, type=int, help='Test Algo with (1) HighwayEnv Implementation, (2) OpenAI Gym')
        self.parser.add_argument('--resume', action='store_true', help='Whether to Load Last Model for Further Training')
        self.parser.add_argument('--load_model', default=None, help='Model to load for Testing')
        self.parser.add_argument('--save_model', default=True, help='Whether to Save Model during Training')
        self.parser.add_argument('--save_video', action='store_true', help='Saves Env Render as Video')

        # Neural Network Settings
        self.parser.add_argument('--arch', default='DoubleConv', help='Neural Net Backbone')
        self.parser.add_argument('--fc_layers', default=2, type=int, help='Number of Dense Layers')
        self.parser.add_argument('--fc_width', default=256, type=int, help='Number of Channels in Dense Layers')

        # Problem Space Settings
        self.parser.add_argument('--obs_dim', default=(4,64,64), type=int, nargs=3, help='Agent Observation Space')
        self.parser.add_argument('--num_actions', default=1, type=int, help='Agent Action Space')
        self.parser.add_argument('--offroad_thres', default=-1, type=int, help='Number of Steps Agent is Allowed to Ride Offroad')
        self.parser.add_argument('--spawn_vehicles', default=1, type=int, help='Number of Non-Agent Vehicles to Spawn, Set 0 to Disable')
        self.parser.add_argument('--random_obstacles', default=0, type=int, help='Number of Static Obstacles to Spawn')

        # Experiment Settings
        self.parser.add_argument('--num_episodes', default=10000, type=int, help='Number of Episodes to Train')
        self.parser.add_argument('--log_freq', default=20, type=int, help='Frequency of Logging (Episodes)')
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
    _, axes = plt.subplots(ncols=4, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
       ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
    plt.show()


def trainDQN(env, agent, num_episodes, opt):

    """Training Sequence for DQN"""
    rewards, best = [], 0
    epsilon = opt.epsilon

    for episode in tqdm(range(1, num_episodes + 1)):

        episode_reward = 0
        obs = env.reset() if not opt.debug == 2 else env.reset().T
        done = False

        while not done:

            # E-Soft Action Selection
            if np.random.random() > epsilon:
                action_idx = np.argmax(agent.get_qvalues(obs))
            elif opt.num_actions == 2:
                action_idx = np.random.randint(0, len(DISCRETE_ACTION_SPACE))
            else:
                action_idx = np.random.randint(0, len(SIMPLE_DISCRETE_ACTION_SPACE))

            # Step through Environment with Continuous Actions
            new_obs, reward, done, _ = env.step(DISCRETE_ACTION_SPACE[action_idx] if opt.num_actions == 2 else
                                                SIMPLE_DISCRETE_ACTION_SPACE[action_idx])
            episode_reward += reward

            # Update Replay Memory & Train Agent Model
            if opt.debug == 2:
                new_obs = new_obs.T
            agent.update_replay((obs, action_idx, reward, new_obs, done))
            agent.train(done)

            obs = new_obs

        # Log Episode Rewards
        rewards.append(episode_reward)
        print(episode_reward)
        
        # For Logging Interval, Extract Average, Lowest, Best Reward Attained
        if episode % opt.log_freq == 0 or episode == 1:
            avg_reward = round(np.mean(rewards[-opt.log_freq:]),3)
            min_reward = round(np.min(rewards[-opt.log_freq:]),3)
            max_reward = round(np.max(rewards[-opt.log_freq:]),3)
            agent.write_log(episode, reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save Model if Average Reward is Greater than a Minimum & Better than Before
            if avg_reward >= np.max([opt.min_reward, best]) and opt.save_model:
                best = avg_reward
                agent.model.save(f'{opt.exp_dir}/last_best.model')

        # Decay Epsilon
        if epsilon > opt.min_epsilon:
            epsilon *= opt.epsilon_decay
            epsilon = max(opt.min_epsilon, epsilon)
            
        """
        # Linear Epsilon Decay
        if epsilon > opt.min_epsilon:
            epsilon = opt.epsilon - episode/num_episodes * (opt.epsilon - opt.min_epsilon)
        """

if __name__ == "__main__":
    
    # Parse Arguments
    opt = opts().parse()
    print(opt)

    # Set up Environment According to Debug Mode
    if opt.debug == 1:
        env = RaceTrackEnv2(opt)
        opt.obs_dim = [2, 36, 36]
    elif opt.debug == 2:
        env = gym.make("CarRacing-v0")
        opt.obs_dim = [3, 96, 96]
        opt.num_actions = 3
    else:
        env = RaceTrackEnv(opt)

    # For Recording or Visualisation
    if opt.save_video:
        env = Monitor(env, './videos/', force=True)

    if opt.mode == "train":
        
        agent = GET_AGENT[opt.agent](opt=opt)
        
        if opt.agent in ["DQN", "CDQN"]:
            trainDQN(env, agent, int(opt.num_episodes), opt)
        else:
            agent.learn(env, opt)

    elif opt.mode == "test":
        
        model = keras.models.load_model(opt.load_model)
        total_reward, obs, seq = 0, env.reset(), []
        
        if opt.agent in ["DQN", "CDQN"]:
            for _ in range(200):
                action_idx = model.predict(np.array([obs])/255)[0]
                action_idx = np.argmax(action_idx)
                obs, reward, done, _ = env.step(DISCRETE_ACTION_SPACE[action_idx] if opt.num_actions == 2 else
                                                SIMPLE_DISCRETE_ACTION_SPACE[action_idx])
                total_reward += reward
                print(reward)
            print("Total Reward: ", total_reward)

        else:
            
            if opt.obs_dim[0] in [1,4]:
            
                for _ in range(200):
                    action = model(np.array([obs])/255)[0]
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    print(reward)
                print("Total Reward: ", total_reward)
                
            else:
            
                for _ in range(200):
                    action = model(np.array([obs]))[0]
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    print(reward)
                print("Total Reward: ", total_reward)
            
    elif opt.mode == "manual":
        
        env.configure({"manual_control": True})
        env.reset()
        total_reward, done = 0, False
        
        while not done:
            obs, reward, done, _ = env.step(env.action_space.sample())
            total_reward += reward
        print("Total Reward: ", total_reward)