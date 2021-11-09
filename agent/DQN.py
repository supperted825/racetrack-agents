import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from collections import deque
import os
import csv
import random
import numpy as np
import datetime

from .models import get_model


DISCRETE_ACTION_SPACE = {
        0: [1 ,-1], 1: [1 , 0], 2: [1 , 1],
        3: [0 ,-1], 4: [0 , 0], 5: [0 , 1],
        6: [-1,-1], 7: [-1, 0], 8: [-1, 1]
    }

SIMPLE_DISCRETE_ACTION_SPACE = {
        0: [-0.5], 1 : [0], 2 : [0.5]
    }


class DQNAgent(object):
    """Double DQN Agent"""

    def __init__(self, opt=None):

        # Configs & Hyperparameters
        self.name = "{}_{}_{}Actions".format(opt.agent, opt.arch, opt.num_actions)
        self.lr = opt.lr
        self.batch_size = opt.batch_size
        self.num_actions = opt.num_actions

        # DQN Hyperparameters
        self.GAMMA = opt.dqn_gamma
        self.UPDATE_FREQ = opt.update_freq
        self.REPLAY_SIZE = opt.replay_size
        self.MIN_REPLAY_SIZE = opt.min_replay_size

        # Main Model to be Trained
        self.model = self.create_model(opt)

        # Target Model is Used for Prediction at Every Step
        self.target_model = self.create_model(opt)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.REPLAY_SIZE)
        self.target_update_counter = 0

        # Logging
        time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        self.logdir = f"{opt.exp_dir}/log_{time}"
        os.mkdir(self.logdir)

        with open(self.logdir + '/log.csv', 'w+', newline ='') as file:
            write = csv.writer(file)
            write.writerow(['Step', 'Avg Reward', 'Min Reward', 'Max Reward', 'Eval Reward', 'Epsilon'])

        with open(self.logdir + '/opt.txt', 'w+', newline ='') as file:
            args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
            for k, v in sorted(args.items()):
                file.write('  %s: %s\n' % (str(k), str(v)))


    def write_log(self, step, **logs):
        """Write Episode Information to CSV File"""
        line = [step] + [round(value,3) for value in logs.values()]
        with open(self.logdir + '/log.csv', 'a', newline ='') as file:
            write = csv.writer(file)
            write.writerow(line)


    def create_model(self, opt):
        """Create Model Neural Network, Common Between Main & Target Models"""

        # Retrieve Model Backbone from Model File
        model = get_model(opt)

        # Add Fully Connected Layers to Model
        for _ in range(opt.fc_layers):
            model.add(Dense(opt.fc_width))

        # Add Final Layer According to Num Actions
        if self.num_actions == 2:
            model.add(Dense(9, activation="linear"))
        else:
            model.add(Dense(3, activation="linear"))

        # Compile & Visualise Model in Console
        lr_schedule = PolynomialDecay(self.lr, opt.num_episodes * 200, end_learning_rate=0)
        model.compile(loss="mse", optimizer=Adam(learning_rate=lr_schedule if opt.lr_decay else self.lr))
        model.summary()

        return model


    def update_replay(self, item):
        """Append Experience Step to the Replay Buffer"""
        self.replay_memory.append(item)


    def get_qvalues(self, state):
        """Get Q Value Outputs from Target Model"""
        return self.target_model.predict_on_batch(np.array(state).reshape(-1, *state.shape)/255)[0]


    def learn(self, env, opt):
        """Training Sequence for DQN"""
        
        num_episodes = opt.num_episodes
        rewards, eval_rewards, best = [], [], 0
        epsilon = opt.epsilon

        for episode in range(1, num_episodes + 1):

            episode_reward = 0
            obs = env.reset()
            done = False

            while not done:

                # E-Soft Action Selection
                if np.random.random() > epsilon:
                    action_idx = np.argmax(self.get_qvalues(obs))
                elif opt.num_actions == 2:
                    action_idx = np.random.randint(0, len(DISCRETE_ACTION_SPACE))
                else:
                    action_idx = np.random.randint(0, len(SIMPLE_DISCRETE_ACTION_SPACE))

                # Step through Environment with Continuous Actions
                new_obs, reward, done, _ = env.step(DISCRETE_ACTION_SPACE[action_idx] if opt.num_actions == 2 else
                                                    SIMPLE_DISCRETE_ACTION_SPACE[action_idx])
                episode_reward += reward

                # Update Replay Memory & Train Agent Model
                self.update_replay((obs, action_idx, reward, new_obs, done))
                self.train()

                obs = new_obs

            # Log Episode Rewards
            rewards.append(episode_reward)
            print(f"Episode {episode} | Training: {episode_reward:.3f}", end=' | ')
            
            # Run One Evaluation Run (Deterministic Actions)
            obs, eval_reward, done = env.reset(), 0, False
            while not done:
                action_idx = np.argmax(self.model.predict_on_batch(np.array([obs])/255)[0])
                obs, reward, done, _ = env.step(DISCRETE_ACTION_SPACE[action_idx] if opt.num_actions == 2 else
                                                SIMPLE_DISCRETE_ACTION_SPACE[action_idx])
                eval_reward += reward
            eval_rewards.append(eval_reward)
            
            print(f"Evaluation: {eval_reward:.3f}")
            
            # For Logging Interval, Extract Average, Lowest, Best Reward / Eval Reward Attained
            if episode % opt.log_freq == 0 or episode == 1:

                # Get Average Stats for Latest Range of Episodes
                avg_reward = round(np.mean(rewards[-opt.log_freq:]),3)
                min_reward = round(np.min(rewards[-opt.log_freq:]),3)
                max_reward = round(np.max(rewards[-opt.log_freq:]),3)
                eval_reward = round(np.mean(eval_rewards[-opt.log_freq:]),3)
                
                self.write_log(episode, reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, eval_reward=eval_reward, epsilon=epsilon)

            # Save Model if Eval Reward is Greater than a Minimum & Better than Before
            if eval_reward >= np.max([opt.min_reward, best]) and opt.save_model:
                best = eval_reward
                print("New best model achieved. Saving!")
                self.model.save(f'{opt.exp_dir}/last_best.model')
            
            # Checkpoint Model every 100 Episodes
            if episode % 100 == 0:
                self.model.save(f'{opt.exp_dir}/checkpoint_{episode}.model')
                
            # Linear Epsilon Decay
            if epsilon > opt.min_epsilon:
                epsilon = (opt.epsilon - opt.min_epsilon) * (num_episodes - episode) / num_episodes + opt.min_epsilon
                
        self.model.save(f'{opt.exp_dir}/model_last.model')


    def train(self):
        """Training Sequence for DQN Agent"""

        # Don't Train Unless Sufficient Data
        if len(self.replay_memory) < self.MIN_REPLAY_SIZE:
            return
        
        # Sample Batch of Data for Updating Model
        minibatch = random.sample(self.replay_memory, self.batch_size)

        current_states = np.array([item[0] for item in minibatch])/255
        new_current_states = np.array([item[3] for item in minibatch])/255
        
        current_qvalues = self.model.predict_on_batch(current_states)
        future_qvalues = self.model.predict_on_batch(new_current_states)

        x, y = [], []

        for index, (current_state, action, reward, _, done) in enumerate(minibatch):

            # Q Value is Reward if Terminal, otherwise we use Return
            if not done:
                new_qvalue = reward + self.GAMMA * np.max(future_qvalues[index])
            else:
                new_qvalue = reward

            # Update Target Q Value for this Action for this Entry
            current_qvalue = current_qvalues[index]
            current_qvalue[action] = new_qvalue

            x.append(current_state)
            y.append(current_qvalue)

        self.model.train_on_batch(np.array(x)/255, np.array(y))
        self.target_update_counter += 1
        
        if self.target_update_counter > self.UPDATE_FREQ:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class CDQNAgent(DQNAgent):
    """Double DQN Agent With Clipping"""

    def train(self):
        """Modified Training Sequence with Clipped Update Rule"""

        if len(self.replay_memory) < self.MIN_REPLAY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, self.batch_size)
        
        current_states = np.array([item[0] for item in minibatch])/255
        new_current_states = np.array([item[3] for item in minibatch])/255

        current_qvalues = self.model.predict_on_batch(current_states)
        new_current_qvalues = self.model.predict_on_batch(new_current_states)
        new_future_qvalues = self.target_model.predict_on_batch(new_current_states)

        x, y = [], []

        for index, (current_state, action, reward, _, done) in enumerate(minibatch):
            if not done:
                # Minimum of New Q Values is Used for the Future Q Value
                model_maxq = np.max(new_current_qvalues[index])
                target_model_maxq = np.max(new_future_qvalues[index])
                new_qvalue = reward + self.GAMMA * np.min([model_maxq, target_model_maxq])
            else:
                new_qvalue = reward

            current_qvalue = current_qvalues[index]
            current_qvalue[action] = new_qvalue

            x.append(current_state)
            y.append(current_qvalue)

        self.model.train_on_batch(np.array(x)/255, np.array(y))
        self.target_update_counter += 1
        
        if self.target_update_counter > self.UPDATE_FREQ:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
