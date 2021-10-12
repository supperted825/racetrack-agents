import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from collections import deque
import os
import csv
import random
import numpy as np
import datetime

from .models import get_model


class DQNAgent(object):
    """Double DQN Agent"""

    def __init__(self, opt=None):

        # Configs & Hyperparameters
        self.name = "{}_{}_{} Actions".format(opt.agent, opt.arch, opt.num_actions)
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
        self.writer = tf.summary.create_file_writer(logdir=f"logs/{self.name}-{time}")

        # self.logdir = f"logs/{self.name}-{time}"
        # os.mkdir(self.logdir)

        # with open(self.logdir + '/log.csv', 'w+', newline ='') as file:
        #     write = csv.writer(file)
        #     write.writerow(['Step', 'Avg Reward', 'Min Reward', 'Max Reward', 'Epsilon'])


    def write_log(self, step, **logs):
        """Write Episode Information to Tensorboard"""
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=step)
                self.writer.flush()
        
        # line = [step] +  [value for value in logs.values()]
        # with open(self.logdir + '/log.csv', 'a', newline ='') as file:
        #     write = csv.writer(file)
        #     write.writerow(line)


    def create_model(self, opt):

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
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        model.summary()

        return model


    def update_replay(self, item):
        self.replay_memory.append(item)


    def get_qvalues(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


    def train(self, terminal_state):

        # Don't Train Unless Sufficient Data
        if len(self.replay_memory) < self.MIN_REPLAY_SIZE:
            return
        
        # Sample Batch of Data for Updating Model
        minibatch = random.sample(self.replay_memory, self.batch_size)

        current_states = np.array([item[0] for item in minibatch])/255
        current_qvalues = self.model.predict(current_states)

        new_current_states = np.array([item[3] for item in minibatch])/255
        future_qvalues = self.target_model.predict(new_current_states)

        x, y = [], []

        for index, (current_state, action, reward, _, done) in enumerate(minibatch):

            # Q Value is Reward if Terminal, otherwise we use G
            if not done:
                new_qvalue = reward + self.GAMMA * np.max(future_qvalues[index])
            else:
                new_qvalue = reward

            # Update Target Q Value for this State
            current_qvalue = current_qvalues[index]
            current_qvalue[action] = new_qvalue

            x.append(current_state)
            y.append(current_qvalue)

        self.model.fit(
            np.array(x)/255, np.array(y),
            batch_size=self.batch_size, verbose=0,
            shuffle=False if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > self.UPDATE_FREQ:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class CDQNAgent(DQNAgent):
    """Double DQN Agent With Clipping"""

    def train(self, terminal_state):
        """Modified Training Sequence with Clipped Update Rule"""

        if len(self.replay_memory) < self.MIN_REPLAY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, self.batch_size)

        current_states = np.array([item[0] for item in minibatch])/255
        current_qvalues = self.model.predict(current_states)

        new_current_states = np.array([item[3] for item in minibatch])/255
        new_current_qvalues = self.model.predict(new_current_states)
        new_future_qvalues = self.target_model.predict(new_current_states)

        x, y = [], []

        for index, (current_state, action, reward, _, done) in enumerate(minibatch):
            if not done:
                model_maxq = np.max(new_current_qvalues[index])
                target_model_maxq = np.max(new_future_qvalues[index])
                new_qvalue = reward + self.GAMMA * np.min([model_maxq, target_model_maxq])
            else:
                new_qvalue = reward

            current_qvalue = current_qvalues[index]
            current_qvalue[action] = new_qvalue

            x.append(current_state)
            y.append(current_qvalue)

        self.model.fit(
            np.array(x)/255, np.array(y),
            batch_size=self.batch_size, verbose=0,
            shuffle=False if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > self.UPDATE_FREQ:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0