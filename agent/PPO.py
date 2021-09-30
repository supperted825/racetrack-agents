from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import concatenate
from tensorflow.keras.optimizers import Adam

from collections import deque
import random
import numpy as np
import datetime

from .models import get_model, ModifiedTensorBoard


"""PPO Pseudocode"""
# Initial policy parameters and initial function value parameters
#     for episode, do:
#         collect trajectories by running policy in environment
#         compute rewards-to-go
#         compute advantage estimates based on current value function
#         update policy by maximizing PPO clip objective via Adam
#         fit value function by regression with mse error via Adam

class PPOAgent():
    """Proximal Policy Optimisation Agent with Clipping"""

    def __init__(self, opt=None):

            # Configs
            self.name = "{}_{}".format(opt.agent, opt.arch)

            # Main Model to be Trained
            self.model = self.create_actor(opt.arch)

            # Target Model is Used for Prediction at Every Step
            self.target_model = self.create_critic(opt.arch)
            self.target_model.set_weights(self.model.get_weights())

            time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
            self.tensorboard = ModifiedTensorBoard(self.name, log_dir=f"logs/{self.name}-{time}")
    
    def create_actor(self, backbone):

        # Define Model Inputs
        obs = Input(shape=(4,128,128))
        act = Input(shape=(2,))
        adv = Input(shape=(1,))

        # Retrieve CNN Backbone from Model File
        cnn = get_model(backbone)
        feats = cnn(obs)

        # Concat & Add Final Layers for PPO Actor & Compile with Loss
        merge = concatenate([feats, act, adv])
        fc1 = Dense(64, activation='relu')(merge)
        fc2 = Dense(32, activation='relu')(fc1)
        out = Dense(2, activation='tanh')(fc2)

        model = Model(inputs=[obs, act, adv], outputs=out)
        model.compile(optimizer=Adam(learning_rate=0.001),
                    metrics=['accuracy'])

        # Visualise Model in Console
        model.summary()

        return model

    def create_critic(self, arch):

        # Retrieve Model Backbone from Model File
        model = get_model(arch)

        # Add Final Output Layers for PPO Critic & Compile with Loss
        model.add(Dense(64))
        model.add(Dense(32))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        # Visualise Model in Console
        model.summary()

        return model