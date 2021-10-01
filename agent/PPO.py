from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam

import tensorflow.keras as K
import tensorflow.keras.backend as F

import math
import random
import numpy as np
import datetime

from .models import get_model, ModifiedTensorBoard

# Disable Eager Execution
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Constants for Prediction
ADV_PLACEHOLDER = np.zeros((1, 1))
ACT_PLACEHOLDER = np.zeros((1, 2))

# PPO Parameters
GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
ACTOR_SIGMA = 1.0
ENTROPY_LOSS_RATIO = 0.001
TARGET_UPDATE_ALPHA = 0.9
MINIBATCH_SIZE = 64


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

            self.target_actor = self.create_actor(opt.arch)
            self.target_actor.set_weights(self.actor.get_weights())

            self.replay_memory = []

            time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
            self.tensorboard = ModifiedTensorBoard(self.name, log_dir=f"logs/{self.name}-{time}")


    def create_actor(self, backbone):

        # Define Model Inputs
        obs = Input(shape=(4,128,128))
        adv = Input(shape=(1,))
        act = Input(shape=(2,))

        # Retrieve Model from Model File
        cnn = get_model(backbone)
        feats = cnn(obs)

        # Add Final Layers for PPO Actor & Compile with Loss
        fc1 = Dense(64, activation='relu')(feats)
        fc2 = Dense(64, activation='relu')(fc1)

        # Model Outputs Mean Value for Each Continuous Action
        out  = Dense(2, activation='tanh')(fc2)

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
        # Keras Backend must be used here as values are symbolic only.

        def pred2logpdf(y_true, y_pred):
            """Convert Model Output to log PDF"""
            pdf = 1 / (F.sqrt(2 * np.pi * F.square(ACTOR_SIGMA))) * \
                    F.exp(-0.5 * F.square((y_true-y_pred)/ACTOR_SIGMA))
            log_pdf = F.log(pdf + F.epsilon())
            return log_pdf
        
        def loss(y_true, y_pred):
            """Calculate Clipped Loss According to https://arxiv.org/pdf/1707.06347.pdf"""
            old_log_pdf = pred2logpdf(y_true, old_pred)
            new_log_pdf = pred2logpdf(y_true, y_pred)
            r = F.exp(new_log_pdf - old_log_pdf)

            # Clipped Actor Loss
            loss1 = r * adv
            loss2 = F.clip(r, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * r
            actor_loss = - F.mean(F.minimum(loss1, loss2))

            # Entropy Bonus
            entropy_loss = ENTROPY_LOSS_RATIO * \
                        F.mean((-F.log(2*np.pi*F.square(ACTOR_SIGMA))+1)/2)

            return actor_loss + entropy_loss
        
        return loss


    def update_replay(self, obs, action, reward, done):
        value = self.critic.predict(obs.reshape(1,*obs.shape))[0]
        mask = 0 if done else 1
        self.replay_memory.append((obs, action, reward, mask, value))
    

    def process_episode(self, replay_memory, g=GAE_GAMMA, l=GAE_LAMBDA):
        """Process Espisode Information for Advantages & Returns"""

        # If Last Entry is Terminal State, Use Reward else V(s)
        if replay_memory[-1][3] == 1:
            last_val = replay_memory[-1][2]
        else:
            last_val = replay_memory[-1][4]
        
        last_adv = 0

        # Initialise Output Arrays with Appropriate Shapes
        obss = np.zeros((len(replay_memory), 4, 128, 128))
        acts = np.zeros((len(replay_memory), 2))
        rets = np.zeros((len(replay_memory),))
        advs = np.zeros((len(replay_memory),))

        for idx, (obs, action, reward, mask, value) in enumerate(replay_memory[::-1]):

            # Calculate Advantage & Return with GAE
            delta = reward + g * last_val * mask - value
            adv = delta + g * l * mask * last_adv
            ret = adv + value

            # Append to Output Arrays
            obss[-idx] = obs
            acts[-idx] = action
            rets[-idx] = ret
            advs[-idx] = adv

            last_val, last_adv = value, adv

        return obss, acts, rets, advs


    def act(self, obs, optimal=False):
        """Act on Parameterised Normal Distribution"""
        mus = self.actor.predict([obs.reshape(1,*obs.shape), ADV_PLACEHOLDER, ACT_PLACEHOLDER])[0]
        if optimal:
            action = [mu for mu in mus]
        else:
            action = [random.gauss(mu,ACTOR_SIGMA) for mu in mus]
        return action


    def train(self):
        
        # Calculate & Extract Advantages & Returns for Episode, then Sample
        obss, actions, rets, advs = self.process_episode(self.replay_memory)
        batch_idx = np.random.randint(len(obss), size=MINIBATCH_SIZE)
        obss, actions, rets, advs = obss[batch_idx], actions[batch_idx], rets[batch_idx], advs[batch_idx]

        # Prepare Batch for Fitting
        advs = advs.reshape(-1,1)
        advs = K.utils.normalize(advs)
        olds = self.target_actor.predict_on_batch([obss,
                        np.repeat(ADV_PLACEHOLDER, MINIBATCH_SIZE, axis=0),
                        np.repeat(ACT_PLACEHOLDER, MINIBATCH_SIZE, axis=0)])

        # Train Actor & Critic
        self.actor.fit(x=[obss, advs, olds], y=actions, epochs=1, verbose=0, callbacks=[self.tensorboard])
        self.critic.fit(x=obss, y=rets, epochs=1, verbose=0, callbacks=[self.tensorboard])

        # Update Target Network
        actor_weights = np.array(self.actor.get_weights(), dtype=object)
        target_actor_weights = np.array(self.target_actor.get_weights(), dtype=object)
        new_weights = TARGET_UPDATE_ALPHA * actor_weights \
                        + (1-TARGET_UPDATE_ALPHA) * target_actor_weights
        self.target_actor.set_weights(new_weights)