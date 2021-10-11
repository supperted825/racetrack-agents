from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.backend as F

import os
import csv
import random
import numpy as np
import datetime

from .models import get_model


"""PPO Pseudocode"""
# Initial policy parameters and initial function value parameters
#     for episode, do:
#         collect trajectories by running policy in environment
#         compute returns
#         compute advantage estimates based on current value function
#         update policy by maximizing PPO clip objective via Adam
#         fit value function by regression with mse error via Adam

class PPOAgent():
    """Proximal Policy Optimisation Agent with Clipping & GAE"""

    def __init__(self, opt=None):

            # Configs & Hyperparameters
            self.name = "{}_{}_{} Actions".format(opt.agent, opt.arch, opt.num_actions)
            self.lr = opt.lr
            self.epochs = opt.num_epochs
            self.batch_size = opt.batch_size
            self.num_actions = opt.num_actions
            self.obs_dim = opt.obs_dim
            disable_eager_execution()
            tf.compat.v1.experimental.output_all_intermediates(True)

            # Constants for Prediction
            self.ADV_PLACEHOLDER = np.zeros((1, 1))
            self.ACT_PLACEHOLDER = np.zeros((1, self.num_actions))

            # PPO Hyperparameters
            self.GAE_GAMMA = opt.gae_gamma
            self.GAE_LAMBDA = opt.gae_lambda
            self.PPO_EPSILON = opt.ppo_epsilon
            self.ENTROPY = opt.ppo_entropy
            self.TARGET_ALPHA = opt.target_alpha
            self.ACTOR_SIGMA = opt.actor_sigma

            # Instantiate Models & Replay Buffer
            self.actor = self.create_actor(opt)
            self.critic = self.create_critic(opt)

            self.target_actor = self.create_actor(opt)
            self.target_actor.set_weights(self.actor.get_weights())

            self.replay_memory = []

            # For Manual Logging (TensorBoard doesn't work with Eager Execution Disabled)
            time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
            self.logdir = f"logs/{self.name}-{time}"
            os.mkdir(self.logdir)

            with open(self.logdir + '/log.csv', 'w+', newline ='') as file:
                write = csv.writer(file)
                write.writerow(['Step', 'Avg Reward', 'Min Reward', 'Max Reward'])


    def create_actor(self, opt):

        # Define Model Inputs
        obs = Input(shape=opt.obs_dim)
        adv = Input(shape=(1,))
        act = Input(shape=(self.num_actions,))

        # Retrieve Model from Model File
        cnn = get_model(opt)
        feats = cnn(obs)

        # Add Final Layers for PPO Actor
        fc1 = Dense(64, activation='relu')(feats)
        fc2 = Dense(64, activation='relu')(fc1)

        # Model Outputs Mean Value for Each Continuous Action
        out = Dense(self.num_actions, activation='tanh')(fc2)

        # Compile Model with Custom PPO Loss
        model = Model(inputs=[obs, adv, act], outputs=out)
        model.compile(
                    loss=self.PPO_loss(adv,act),
                    optimizer=Adam(learning_rate=self.lr),
                    metrics=['accuracy'])

        model.summary()

        return model


    def create_critic(self, opt):

        # Retrieve Model Backbone from Model File
        model = get_model(opt)

        # Add Final Output Layers for PPO Critic & Compile with Loss
        model.add(Dense(64))
        model.add(Dense(64))
        model.add(Dense(1))

        # Critic Simply Uses MSE Loss
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr), metrics=['accuracy'])
        model.summary()

        return model
    

    def PPO_loss(self, adv, old_pred):
        # Log Probability of Loss: (x-mu)²/2sigma² - log(sqrt(2*PI*sigma²))
        # Entropy of Normal Distribution: sqrt(2*PI*e*sigma²)
        # Reference: https://www.youtube.com/watch?v=WxQfQW48A4A
        # Keras Backend must be used here as values are symbolic only.

        def pred2logpdf(y_true, y_pred):
            """Convert Model Output to log PDF"""
            pdf = 1 / (F.sqrt(2 * np.pi * F.square(self.ACTOR_SIGMA))) * \
                    F.exp(-0.5 * F.square((y_true-y_pred)/self.ACTOR_SIGMA))
            log_pdf = F.log(pdf + F.epsilon())
            return log_pdf
        
        def loss(y_true, y_pred):
            """Calculate Clipped Loss According to https://arxiv.org/pdf/1707.06347.pdf"""
            old_log_pdf = pred2logpdf(y_true, old_pred)
            new_log_pdf = pred2logpdf(y_true, y_pred)
            r = F.exp(new_log_pdf - old_log_pdf)

            # Clipped Actor Loss
            loss1 = r * adv
            loss2 = F.clip(r, 1 - self.PPO_EPSILON, 1 + self.PPO_EPSILON) * adv
            actor_loss = - F.mean(F.minimum(loss1, loss2))

            # Entropy Bonus
            entropy_loss = self.ENTROPY * \
                        F.mean(-0.5 * (F.log(2*np.pi*F.square(self.ACTOR_SIGMA))+1))

            return actor_loss + entropy_loss
        
        return loss


    def write_log(self, step, **logs):
        """Write Episode Information to CSV File"""
        line = [step] +  [value for value in logs.values()]
        with open(self.logdir + '/log.csv', 'a', newline ='') as file:
            write = csv.writer(file)
            write.writerow(line)


    def update_replay(self, obs, action, reward, done):
        """Record Stepwise Episode Information with Critic Output"""
        value = self.critic.predict(obs.reshape(1,*obs.shape)/255)[0]
        mask = 0 if done else 1
        self.replay_memory.append((obs, action, reward, mask, value))
    

    def process_episode(self, replay_memory):
        """Process Espisode Information for Advantages & Returns, see https://arxiv.org/pdf/1506.02438.pdf"""

        # If Last Entry is Terminal State, Use Reward else V(s)
        if replay_memory[-1][3] == 1:
            last_val = replay_memory[-1][2]
        else:
            last_val = replay_memory[-1][4]
        
        last_adv = 0
        g = self.GAE_GAMMA
        l = self.GAE_LAMBDA

        # Initialise Output Arrays with Appropriate Shapes
        obss = np.zeros((len(replay_memory), *self.obs_dim))
        acts = np.zeros((len(replay_memory), self.num_actions))
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
        mus = self.target_actor.predict([obs.reshape(1,*obs.shape)/255, self.ADV_PLACEHOLDER, self.ACT_PLACEHOLDER])[0]
        if optimal:
            action = [mu for mu in mus]
        else:
            action = [random.gauss(mu,self.ACTOR_SIGMA) for mu in mus]
        return action


    def train(self):
        
        # Calculate & Extract Advantages & Returns for Episode, then Sample
        obss, actions, rets, advs = self.process_episode(self.replay_memory)
        batch_idx = np.random.randint(len(obss), size=self.batch_size)
        obss, actions, rets, advs = obss[batch_idx], actions[batch_idx], rets[batch_idx], advs[batch_idx]

        # Prepare Batch for Fitting
        advs = advs.reshape(-1,1)
        advs = K.utils.normalize(advs)
        olds = self.target_actor.predict_on_batch([obss/255,
                        np.repeat(self.ADV_PLACEHOLDER, self.batch_size, axis=0),
                        np.repeat(self.ACT_PLACEHOLDER, self.batch_size, axis=0)])

        # Train Actor & Critic
        self.actor.fit(x=[obss/255, advs, olds], y=actions, epochs=self.epochs, verbose=0)
        self.critic.fit(x=obss/255, y=rets, epochs=self.epochs, verbose=0)

        # Soft-Update Target Network
        actor_weights = np.array(self.actor.get_weights(), dtype=object)
        target_actor_weights = np.array(self.target_actor.get_weights(), dtype=object)
        new_weights = self.TARGET_ALPHA * actor_weights \
                        + (1-self.TARGET_ALPHA) * target_actor_weights
        self.target_actor.set_weights(new_weights)