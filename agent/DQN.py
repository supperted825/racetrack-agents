import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
from keras.callbacks import TensorBoard

from collections import deque
import random
import numpy as np
import datetime
import os

from .models import get_model


REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 500
MODEL_NAME = "DQN_DoubleConv256"
MINIBATCH_SIZE = 16
UPDATE_TARGET_FREQ = 5
DISCOUNT = 0.99


class ModifiedTensorBoard(TensorBoard):
    """Custom Tensorboard, modified from PythonProgramming.net by Mohammed AL-Ma'amari."""
    
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided, we train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class DQNAgent(object):
    """Double DQN Agent"""

    def __init__(self, name=MODEL_NAME, opt=None):

        # Configs
        self.name = name

        # Main Model to be Trained
        self.model = self.create_model(opt.arch)

        # Target Model is Used for Prediction at Every Step
        self.target_model = self.create_model(opt.arch)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

        time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        self.tensorboard = ModifiedTensorBoard(self.name, log_dir=f"logs/{self.name}-{time}")


    def create_model(self, arch):

        # Retrieve Model Backbone from Model File
        model = get_model(arch)

        # Add Final Output Layer for DQN Agent & Compile with Loss
        model.add(Dense(9, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        # Visualise Model in Console
        model.summary()

        return model


    def update_replay(self, transition):
        self.replay_memory.append(transition)


    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(
            np.array(X)/255, np.array(y),
            batch_size=MINIBATCH_SIZE, verbose=0,
            steps_per_epoch=(len(X)//MINIBATCH_SIZE),
            shuffle=False, callbacks=[self.tensorboard]
            if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_FREQ:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0