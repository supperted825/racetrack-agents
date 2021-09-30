from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from collections import deque
import random
import numpy as np
import datetime

from .models import get_model, ModifiedTensorBoard


# Constants for Prediction
ADV_PLACEHOLDER = np.zeros((1, 1))
ACT_PLACEHOLDER = np.zeros((1, 2))

# PPO Parameters
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
ENTROPY_LOSS_RATIO = 0.001

# Memory Replay Parameters
MINIBATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 500


"""PPO Pseudocode"""
# Initial policy parameters and initial function value parameters
#     for episode, do:
#         collect trajectories by running policy in environment
#         compute rewards-to-go
#         compute advantage estimates based on current value function
#         update policy by maximizing PPO clip objective via Adam
#         fit value function by regression with mse error via Adam

class PPOAgent():
    """Proximal Policy Optimisation Agent with Clipping & GAE"""

    def __init__(self, opt=None):

            # Configs
            self.name = "{}_{}".format(opt.agent, opt.arch)

            # Main Model to be Trained
            self.actor = self.create_actor(opt.arch)
            self.critic = self.create_critic(opt.arch)

            self.old_actor = self.create_actor(opt.arch)
            self.old_actor.set_weights(self.actor.get_weights())

            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

            time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
            self.tensorboard = ModifiedTensorBoard(self.name, log_dir=f"logs/{self.name}-{time}")

    def create_actor(self, backbone):

        # Define Model Inputs
        obs = Input(shape=(4,128,128))
        adv = Input(shape=(1,))
        act = Input(shape=(2*2,))

        # Retrieve Model from Model File
        cnn = get_model(backbone)
        feats = cnn(obs)

        # Add Final Layers for PPO Actor & Compile with Loss
        fc1 = Dense(64, activation='relu')(feats)
        fc2 = Dense(64, activation='relu')(fc1)

        # Model Outputs Means and Variances for Each Continuous Action
        mu  = Dense(2, activation='tanh')(fc2)
        sig = Dense(2, activation='softplus')(fc2)
        out = concatenate([mu,sig])

        model = Model(inputs=[obs, adv, act], outputs=out)
        model.compile(
                    loss=self.PPO_loss(adv,act),
                    optimizer=Adam(learning_rate=0.001),
                    metrics=['accuracy'])

        # Visualise Model in Console
        model.summary()

        return model

    def create_critic(self, backbone):

        # Retrieve Model Backbone from Model File
        model = get_model(backbone)

        # Add Final Output Layers for PPO Critic & Compile with Loss
        model.add(Dense(64))
        model.add(Dense(64))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        # Visualise Model in Console
        model.summary()

        return model
    
    def PPO_loss(self, adv, old_pred):
        # Log Probability of Loss: (x-mu)²/2sigma² - log(sqrt(2*PI*sigma²))
        # Entropy of Normal Distribution: sqrt(2*PI*e*sigma²)
        # Reference: https://www.youtube.com/watch?v=WxQfQW48A4A

        def pred2logpdf(y_true, y_pred):
            """Convert Model Output *[(Mean, Variance)] to log PDF"""
            mu = np.array([x[0] for x in y_pred])
            sigma = np.array([x[1] for x in y_pred])
            pdf = 1 / (sigma * np.sqrt(2*np.pi)) * \
                np.exp(-0.5 * np.square((y_true-mu)/sigma))
            log_pdf = np.log(pdf + K.epsilon())
            return log_pdf
        
        def loss(y_true, y_pred):
            """Calculate Clipped Loss According to https://arxiv.org/pdf/1707.06347.pdf"""
            old_log_pdf = pred2logpdf(y_true, old_pred)
            new_log_pdf = pred2logpdf(y_true, y_pred)
            r = np.exp(new_log_pdf - old_log_pdf)

            # Clipped Actor Loss
            loss1 = r * adv
            loss2 = np.clip(r, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * r
            actor_loss = - np.mean(np.min(loss1, loss2))

            # Entropy Bonus
            sigma = np.array([x[1] for x in y_pred])
            variance = np.square(sigma)
            entropy_loss = ENTROPY_LOSS_RATIO * \
                        np.mean((-np.log(2*np.pi*variance)+1)/2)

            return actor_loss + entropy_loss
        
        return loss

    def update_replay(self, item):
        self.replay_memory.append(item)
    
    def act(self, obs, optimal=False):
        model_output = self.actor.predict([[obs],ADV_PLACEHOLDER, ACT_PLACEHOLDER])
        mus  = model_output[0][0]
        sigs = model_output[0][1]
        if optimal:
            action = [mu for mu in mus]
        else:
            action = [random.gauss(mu,sig) for mu, sig in zip(mus,sigs)]
        return action

    def compute_GAE(self):
        pass

    def train(self, terminal_state):
        obs, actions, adv, returns, values = random.sample(self.replay_memory, MINIBATCH_SIZE)
        pass

