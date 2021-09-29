import gym
from gym.wrappers import Monitor
from gym.wrappers.monitoring import video_recorder
from racetrack_env import RaceTrackEnv

import highway_env
import argparse
import base64
import keras
import datetime
import numpy as np

from tqdm import tqdm
from pathlib import Path
from pyvirtualdisplay import Display

from agent.DQN import DQNAgent, CDQNAgent


EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

GET_AGENT = {
    "DQN" : DQNAgent,
    "CDQN": CDQNAgent
}

DISCRETE_ACTION_SPACE = {
        0: [1 ,-1], 1: [1 , 0], 2: [1 , 1],
        3: [0 ,-1], 4: [0 , 0], 5: [0 , 1],
        6: [-1,-1], 7: [-1, 0], 8: [-1, 1]
    }


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--mode', default='train', help='Train or Test')
        self.parser.add_argument('--agent', default='CDQN', help='DQN, DDPG, PPO')
        self.parser.add_argument('--arch', default='DoubleConv256', help='Neural Net Backbone')
        self.parser.add_argument('--load_model', default=None, help='Model to load for Testing')
        self.parser.add_argument('--save_model', default=True, help='Whether to Save Model during Training')
        self.parser.add_argument('--num_episodes', default=200, help='Number of Episodes to Train')
        self.parser.add_argument('--log_freq', default=20, help='Frequency of Logging (Episodes)')
        self.parser.add_argument('--min_reward', default=100, help='Minimum Reward to Save Model')
        self.parser.add_argument('--epsilon', default=1, help='Initial Value of Epsilon')

    
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return opt


def show_video():
    html = []
    for mp4 in Path("video").glob("*.mp4"):
        print("video")
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                      loop controls style="height: 400px    ;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    Display.display(Display.HTML(data="<br>".join(html)))
    

def trainDQN(env, agent, num_episodes, opt):

    """Training Sequence for DQN"""

    rewards = []
    epsilon = opt.epsilon

    # Run Episodes
    for episode in tqdm(range(1, num_episodes + 1)):
        agent.tensorboard.step = episode

        episode_reward = 0
        step = 1
        obs = env.reset()
        done = False

        while not done:

            # E-Soft Action Selection
            if np.random.random() > epsilon:
                action_idx = np.argmax(agent.get_qvalues(obs))
            else:
                action_idx = np.random.randint(0, len(DISCRETE_ACTION_SPACE))

            # Step through Environment with Continuous Actions
            new_obs, reward, done, _ = env.step(DISCRETE_ACTION_SPACE[action_idx])
            episode_reward += reward

            # Update Replay Memory & Train Agent Model
            agent.update_replay((obs, action_idx, reward, new_obs, done))
            agent.train(done)

            obs = new_obs
            step += 1

        # Logging
        rewards.append(episode_reward)
        best = 0
        
        # For Logging Interval, Extract Average, Lowest, Best Reward Attained
        if episode % opt.log_freq == 0 or episode == 1:
            avg_reward = sum(rewards[-opt.log_freq:])/len(rewards[-opt.log_freq:])
            min_reward = min(rewards[-opt.log_freq:])
            max_reward = max(rewards[-opt.log_freq:])
            agent.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save Model if Average Reward is Greater than a Minimum & Better than Before
            if avg_reward >= np.max([opt.min_reward, best]) and opt.save_model:
                best = avg_reward
                time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
                agent.model.save(f'models/{agent.name}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{time}.model')

        # Decay Epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


def trainContinuous(env, agent, num_episodes, opt=None):
    pass

if __name__ == "__main__":
    
    # Parse Arguments
    opt = opts().parse()
    env = RaceTrackEnv()
    agent = GET_AGENT[opt.agent](opt=opt)

    # For Recording or Visualisation
    # env = Monitor(env, './video', force=True, video_callable=lambda episode: True)
    # vid = video_recorder.VideoRecorder(env,path="/Users/jontan/Desktop/vid.mp4")

    if opt.mode == "train":

        if opt.agent in ["DQN", "CDQN"]:
            trainDQN(env, agent, opt.num_episodes, opt)
        else:
            trainContinuous(env, agent, opt.num_episodes, opt)

    else:
        
        # If not Training, Load Model
        model = keras.models.load_model(opt.load_model)
        
        if opt.agent == "DQN":
            obs = env.reset()
            for _ in range(1000):
                action_idx = np.argmax(agent.get_qvalues(obs))
                obs, reward, done, info = env.step(DISCRETE_ACTION_SPACE[action_idx])
                #env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
                #vid.capture_frame()
            #vid.close()

        else:

            for _ in range(1000):
                action = agent.get_qvalues(obs)
                obs, reward, done, info = env.step(action)
                env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
                #vid.capture_frame()
            #vid.close()
        
        show_video()