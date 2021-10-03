import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from collections import deque
import os
import random
import numpy as np
import datetime

from .models import get_model


REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 500
MODEL_NAME = "DQN_DoubleConv256"
MINIBATCH_SIZE = 64
UPDATE_TARGET_FREQ = 50
DISCOUNT = 0.99


class DQNAgent(object):
    """Double DQN Agent"""

    def __init__(self, opt=None):

        # Configs
        self.name = "{}_{}".format(opt.agent, opt.arch)
        self.lr = opt.lr if opt.lr else 5e-4

        # Main Model to be Trained
        self.model = self.create_model(opt.arch)

        # Target Model is Used for Prediction at Every Step
        self.target_model = self.create_model(opt.arch)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

        # Logging
        time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        self.writer = tf.summary.create_file_writer(logdir=f"logs/{self.name}-{time}")

        # self.logdir = f"logs/{self.name}-{time}"
        # os.mkdir(self.logdir)

        # with open(self.logdir + '/log.csv', 'w+', newline ='') as file:
        #     write = csv.writer(file)
        #     write.writerow(['Step','Avg Reward', 'Min Reward', 'Max Reward', 'Epsilon'])


    def write_log(self, step, **logs):
        """Write Episode Information to Tensorboard"""
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=step)
                self.writer.flush()    


    def create_model(self, arch):

        # Retrieve Model Backbone from Model File
        model = get_model(arch)

        # Add Final Output Layers for DQN Agent & Compile with Loss
        model.add(Dense(64))
        model.add(Dense(9, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))

        # Visualise Model in Console
        model.summary()

        return model


    def update_replay(self, item):
        self.replay_memory.append(item)


    def get_qvalues(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


    def train(self, terminal_state):

        # Don't Train Unless Sufficient Data
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Sample Batch of Data for Updating Model
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([item[0] for item in minibatch])/255
        current_qvalues = self.model.predict(current_states)

        new_current_states = np.array([item[3] for item in minibatch])/255
        future_qvalues = self.target_model.predict(new_current_states)

        x, y = [], []

        for index, (current_state, action, reward, _, done) in enumerate(minibatch):

            # Q Value is Reward if Terminal, otherwise we use G
            if not done:
                new_qvalue = reward + DISCOUNT * np.max(future_qvalues[index])
            else:
                new_qvalue = reward

            # Update Target Q Value for this State
            current_qvalue = current_qvalues[index]
            current_qvalue[action] = new_qvalue

            x.append(current_state)
            y.append(current_qvalue)

        self.model.fit(
            np.array(x)/255, np.array(y),
            batch_size=MINIBATCH_SIZE, verbose=0,
            shuffle=False, callbacks=[self.tensorboard]
            if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_FREQ:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class CDQNAgent(DQNAgent):
    """Double DQN Agent With Clipping"""

    def train(self, terminal_state):
        """Modified Training Sequence with Clipped Update Rule"""

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

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
                new_qvalue = reward + DISCOUNT * np.min([model_maxq, target_model_maxq])
            else:
                new_qvalue = reward

            current_qvalue = current_qvalues[index]
            current_qvalue[action] = new_qvalue

            x.append(current_state)
            y.append(current_qvalue)

        self.model.fit(
            np.array(x)/255, np.array(y),
            batch_size=MINIBATCH_SIZE, verbose=0,
            shuffle=False if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_FREQ:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0