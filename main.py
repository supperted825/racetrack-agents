from tqdm import tqdm
from gym.wrappers import Monitor
from racetrack_env import RaceTrackEnv, RaceTrackEnv2

import argparse
import datetime
import numpy as np
import tensorflow.keras as keras

import os
import glob
import subprocess
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from agent.DQN import DQNAgent, CDQNAgent
from agent.PPO import PPOAgent


GET_AGENT = {
    "DQN" : DQNAgent,
    "CDQN": CDQNAgent,
    "PPO" : PPOAgent
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
        self.parser.add_argument('--mode', default='train', help='Train or Test')
        self.parser.add_argument('--agent', default='PPO', help='DQN, DDPG, PPO')
        self.parser.add_argument('--debug', action='store_true', help='Use HighwayEnv Implementation for Testing')
        self.parser.add_argument('--load_model', default=None, help='Model to load for Testing')
        self.parser.add_argument('--save_model', default=True, help='Whether to Save Model during Training')
        self.parser.add_argument('--save_video', action='store_true', help='Saves Env Render as Video')

        # Neural Network Settings
        self.parser.add_argument('--arch', default='DoubleConv256', help='Neural Net Backbone')
        self.parser.add_argument('--fc_layers', default=2, type=int, help='Number of Dense Layers')
        self.parser.add_argument('--fc_width', default=256, type=int, help='Number of Channels in Dense Layers')

        # Problem Space Settings
        self.parser.add_argument('--obs_dim', default=(4,128,128), type=int, nargs=3, help='Agent Observation Space')
        self.parser.add_argument('--obs_stack', default=4, type=int, help='Grayscale Observation Stack Size')
        self.parser.add_argument('--num_actions', default=1, type=int, help='Agent Action Space')

        # Experiment Settings
        self.parser.add_argument('--num_episodes', default=100, type=int, help='Number of Episodes to Train')
        self.parser.add_argument('--log_freq', default=20, type=int, help='Frequency of Logging (Episodes)')
        self.parser.add_argument('--min_reward', default=50, type=int, help='Minimum Reward to Save Model')

        # Hyperparameters
        self.parser.add_argument('--lr', default=5e-4, type=float, help='Policy Learning Rate')
        self.parser.add_argument('--batch_size', default=64, type=int, help='Policy Update Batch Size')
        self.parser.add_argument('--num_epochs', default=10, type=int, help='Num Epochs for Policy Gradient')

        # DQN Hyperparameters
        self.parser.add_argument('--epsilon', default=1, type=float, help='Initial Value of Epsilon')
        self.parser.add_argument('--epsilon_decay', default=1, type=float, help='Decay Ratio of Epsilon')
        self.parser.add_argument('--min_epsilon', default=1, type=float, help='Minimum Value of Epsilon')
        self.parser.add_argument('--dqn_gamma', default=0.99, type=float, help='Frequency of Updating Target Model')
        self.parser.add_argument('--update_freq', default=20, type=int, help='Frequency of Updating Target Model')
        self.parser.add_argument('--replay_size', default=10000, type=int, help='Size of the Replay Memory Buffer')
        self.parser.add_argument('--min_replay_size', default=500, type=int, help='Minimum Memory Entries before Training')

        # PPO Hyperparameters
        self.parser.add_argument('--gae_lambda', default=0.99, type=float, help='Generalised Advantage Estimate Lambda')
        self.parser.add_argument('--gae_gamma', default=0.95, type=float, help='Generalised Advantage Estimate Gamma')
        self.parser.add_argument('--ppo_epsilon', default=0.2, type=float, help='Clipping Loss Epsilon')
        self.parser.add_argument('--ppo_entropy', default=0.001, type=float, help='Regulariser Entropy Loss Ratio')
        self.parser.add_argument('--target_alpha', default=0.9, type=float, help='Target Network Update Coefficient')
        self.parser.add_argument('--actor_sigma', default=0.2, type=float, help='Actor Continuous Action Variance')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return opt


def generate_video(seq, name, folder="./video"):
    """Save Videos from Sequence of Images"""
    for i in range(len(seq)):
        plt.imshow(seq[i], cmap=cm.Greys_r)
        plt.savefig(folder + "/tmp/file%02d.png" % i)

    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png',
        '-r', '30', '-pix_fmt', 'yuv420p', '{}/{}.mp4'.format(folder, name)])
    for file_name in glob.glob(folder + "/tmp/*.png"):
        os.remove(file_name)


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
        obs = env.reset()
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
            agent.update_replay((obs, action_idx, reward, new_obs, done))
            agent.train(done)

            obs = new_obs

        # Log Episode Rewards
        rewards.append(episode_reward)
        print(episode_reward)
        
        # For Logging Interval, Extract Average, Lowest, Best Reward Attained
        if episode % opt.log_freq == 0 or episode == 1:
            avg_reward = np.mean(rewards[-opt.log_freq:])
            min_reward = np.min(rewards[-opt.log_freq:])
            max_reward = np.max(rewards[-opt.log_freq:])
            agent.write_log(episode, reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save Model if Average Reward is Greater than a Minimum & Better than Before
            if avg_reward >= np.max([opt.min_reward, best]) and opt.save_model:
                best = avg_reward
                time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
                agent.model.save(f'models/{agent.name}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{time}.model')

        # Decay Epsilon
        if epsilon > opt.min_epsilon:
            epsilon *= opt.epsilon_decay
            epsilon = max(opt.min_epsilon, epsilon)

        """
        # Linear Epsilon Decay
        if epsilon > opt.min_epsilon:
            epsilon = opt.epsilon - episode/num_episodes * (opt.epsilon - opt.min_epsilon)
        """


def trainPPO(env, agent, num_episodes, opt=None):

    """Training Sequence for PPO"""

    rewards, best = [], 0

    for episode in tqdm(range(1, num_episodes + 1)):

        episode_reward = 0
        obs = env.reset()
        done = False

        while not done:

            # Get Action & Step Environment
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

            # Update Replay Memory
            agent.update_replay(obs, action, reward, done)
        
        # Train Agent & Clear Replay Memory
        agent.train()
        agent.replay_memory.clear()
        
        # Log Episode Rewards
        rewards.append(episode_reward)
        print(episode_reward)
        
        # For Logging Interval, Extract Average, Lowest, Best Reward Attained
        if episode % opt.log_freq == 0 or episode == 1:
            avg_reward = np.mean(rewards[-opt.log_freq:])
            min_reward = np.min(rewards[-opt.log_freq:])
            max_reward = np.max(rewards[-opt.log_freq:])
            agent.write_log(episode, reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward)

            # Save Model if Average Reward is Greater than a Minimum & Better than Before
            if avg_reward >= np.max([opt.min_reward, best]) and opt.save_model:
                best = avg_reward
                time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
                agent.actor.save(f'models/{agent.name}_actor__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{time}.model')
                agent.critic.save(f'models/{agent.name}_critic__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{time}.model')


if __name__ == "__main__":
    
    # Parse Arguments
    opt = opts().parse()
    print(opt)
    env = RaceTrackEnv2(opt) if opt.debug else RaceTrackEnv(opt)
    agent = GET_AGENT[opt.agent](opt=opt)

    # For Recording or Visualisation
    if opt.save_video:
        env = Monitor(env, './videos/', force=True)

    if opt.mode == "train":

        if opt.agent in ["DQN", "CDQN"]:
            trainDQN(env, agent, int(opt.num_episodes), opt)
        else:
            trainPPO(env, agent, int(opt.num_episodes), opt)

    else:
        
        model = keras.models.load_model(opt.load_model)
        total_reward, obs = 0, env.reset()
        
        if opt.agent in ["DQN", "CDQN"]:
            for _ in range(200):
                action_idx = np.argmax(model.predict(np.array([obs])/255)[0])
                obs, reward, done, info = env.step(DISCRETE_ACTION_SPACE[action_idx])
                total_reward += reward
            print(total_reward)

        else:
            for _ in range(200):
                action = agent.get_qvalues(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            print(total_reward)