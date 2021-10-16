from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.op_def_library import value_to_attr_value
from tensorflow.python.framework.ops import disable_eager_execution

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.backend as F

import tensorflow_probability as tfp
tfd = tfp.distributions

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
    """Proximal Policy Optimisation Agent with Clipping"""

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

            # PPO Hyperparameters
            self.GAE_GAMMA = opt.gae_gamma
            self.GAE_LAMBDA = opt.gae_lambda
            self.PPO_EPSILON = opt.ppo_epsilon
            self.ENTROPY = opt.ppo_entropy
            self.TARGET_ALPHA = opt.target_alpha
            self.TARGET_KL = opt.target_kl
            self.ACTOR_SIGMA = opt.actor_sigma

            # Placeholder Variables for Prediction & Training
            self.ADV_PLACEHOLDER = np.zeros((1, 1))
            self.cov_var = np.full((self.num_actions,), fill_value=self.ACTOR_SIGMA, dtype='float32')
            self.cov_mat = np.diag(self.cov_var)

            # Instantiate Models & Replay Buffer
            self.actor = self.create_actor(opt)
            self.critic = self.create_critic(opt)

            self.target_actor = self.create_actor(opt)
            self.target_actor.set_weights(self.actor.get_weights())

            # Experience Replay Buffer
            self.replay_memory = {"obss" : [], "acts" : [], "rews":[], "mask" : []}

            # For Manual Logging (TensorBoard doesn't work with Eager Execution Disabled)
            time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
            self.logdir = f"logs/{self.name}-{time}"
            os.mkdir(self.logdir)

            with open(self.logdir + '/log.csv', 'w+', newline ='') as file:
                write = csv.writer(file)
                write.writerow(['Step', 'Avg Reward', 'Min Reward', 'Max Reward'])
            
            with open(self.logdir + '/opt.txt', 'w+', newline ='') as file:
                args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
                for k, v in sorted(args.items()):
                    file.write('  %s: %s\n' % (str(k), str(v)))


    def create_actor(self, opt):

        # Define Model Inputs
        obs = Input(shape=opt.obs_dim)
        adv = Input(shape=(1,))

        # Retrieve Model from Model File
        cnn = get_model(opt)
        x = cnn(obs)

        # Add FC Layers for PPO Actor
        for _ in range(opt.fc_layers):
            x = Dense(opt.fc_width, activation='relu')(x)

        # Model Outputs Mean Value for Each Continuous Action
        out = Dense(self.num_actions, activation='tanh')(x)

        # Compile Model with Custom PPO Loss
        model = Model(inputs=[obs, adv], outputs=out)
        model.compile(
                    loss=self.PPO_loss(adv),
                    optimizer=Adam(learning_rate=self.lr))

        model.summary()

        return model


    def create_critic(self, opt):

        # Retrieve Model Backbone from Model File
        model = get_model(opt)

        # Add FC Layers for PPO Critic
        for _ in range(opt.fc_layers):
            model.add(Dense(opt.fc_width))

        # Model Outputs Value Estimate for Each State
        model.add(Dense(1))

        # Critic Simply Uses MSE Loss
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        model.summary()

        return model
    

    def PPO_loss(self, advs):
        """Custom PPO Loss, Wrapped to Feed Advantage in"""
        # Keras Backend must be used here as values are symbolic only.
        # Must involve y_true & y_pred inputs, otherwise model will not train.

        def loss(y_true, y_pred):
            """Calculate Clipped Loss According to https://arxiv.org/pdf/1707.06347.pdf"""

            # Get Old & New Distributions & Log Probabilities
            new_dist = tfd.MultivariateNormalDiag(y_pred, self.cov_mat)
            new_log_probs = new_dist.log_prob(y_pred)

            old_dist = tfd.MultivariateNormalDiag(y_true, self.cov_mat)
            old_log_probs = new_dist.log_prob(y_true)

            # Calculate Ratio Between Old & New Policy
            ratios = F.exp(new_log_probs - old_log_probs)

            # Clipped Actor Loss
            loss1 = advs * ratios
            loss2 = advs * F.clip(ratios, 1 - self.PPO_EPSILON, 1 + self.PPO_EPSILON)
            actor_loss = F.mean(-F.minimum(loss1, loss2))

            # Entropy Bonus
            entropy_loss = - self.ENTROPY * old_dist.entropy()

            return actor_loss + entropy_loss
        
        return loss


    def write_log(self, step, **logs):
        """Write Episode Information to CSV File"""
        line = [step] + [value for value in logs.values()]
        with open(self.logdir + '/log.csv', 'a', newline ='') as file:
            write = csv.writer(file)
            write.writerow(line)


    def clear_memory(self):
        """Reset Agent Replay Memory Buffer"""
        for key in self.replay_memory.keys():
            self.replay_memory[key] = []


    def update_replay(self, obs, act, rew, done):
        """Record Stepwise Episode Information with Critic Output"""
        self.replay_memory["obss"].append(obs)
        self.replay_memory["acts"].append(act)
        self.replay_memory["rews"].append(rew)
        self.replay_memory["mask"].append(0 if done else 1)
    

    def process_episode(self, mem):
        """Process Espisode Information for Value & Advantages"""

        # Get Latest Value Estimations from Critic
        vals = self.critic.predict_on_batch(np.array(mem["obss"])/255)
        vals = vals.flatten()

        # If Last Entry is Terminal State, Use Reward else V(s)
        if mem["mask"][-1] == 1:
            last_val = mem["rews"][-1]
        else:
            last_val = vals[-1]
        
        g = self.GAE_GAMMA
        l = self.GAE_LAMBDA

        # Initialise Return Arrays with Appropriate Shapes
        obss = np.empty((len(vals), *self.obs_dim))
        acts = np.empty((len(vals), self.num_actions))
        advs = np.empty((len(vals),))
        rets = np.empty((len(vals),))

        for idx, value in enumerate(vals[::-1]):
            reward = mem["rews"][idx]
            mask = mem["mask"][idx]

            # Calculate Return & Advantage
            ret = reward + g * last_val * mask - value
            adv = ret - value

            # Append to Output Arrays
            obss[-idx-1] = mem["obss"][idx]
            acts[-idx-1] = mem["acts"][idx]
            rets[-idx-1] = ret
            advs[-idx-1] = adv

            last_val = value if mask == 1 else ret

        return obss, acts, advs, rets


    def act(self, obs, optimal=False):
        """Act on Parameterised Normal Distribution"""
        mus = self.target_actor.predict([obs.reshape(1,*obs.shape)/255, self.ADV_PLACEHOLDER])[0]
        if optimal:
            action = [mu for mu in mus]
        else:
            action = [random.gauss(mu,self.ACTOR_SIGMA) for mu in mus]
        return action

    
    def evaluate(self, obs, acts):
        """Produce Values & Log Probabilties for Model Update"""

        # Pull Outputs from Each Model
        vals = self.critic.predict_on_batch(np.array(obs/255))
        mus = self.target_actor.predict_on_batch([obs/255, np.repeat(self.ADV_PLACEHOLDER, len(vals), axis=0)])

        # Calculate Log Probabilities of Each Action
        dist = tfd.MultivariateNormalDiag(mus, self.cov_mat)
        log_probs = dist.log_prob(acts).eval(session=tf.compat.v1.Session())

        return vals, log_probs

    def train(self):
        
        # Process Returns & Advantages for Buffer Info
        ep_obss, ep_acts, ep_advs, ep_rets = self.process_episode(self.replay_memory)

        # Calculate Initial Log Probs for Early Stopping
        _, ep_log_probs = self.evaluate(ep_obss, ep_acts)

        # Sample Batch of Batch Size from Episode Steps
        batch_idx = np.random.randint(len(ep_obss), size=self.batch_size)

        obss = ep_obss[batch_idx]
        acts = ep_acts[batch_idx]
        advs = ep_advs[batch_idx]
        rets = ep_rets[batch_idx]

        ref_log_probs = ep_log_probs[batch_idx]

        # Normalise Advantages & Reshape as a Single Batch
        advs = (advs - advs.mean()) / (advs.std() + 1e-10)
        advs = np.expand_dims(advs,axis=1)

        for epoch in range(self.epochs):

            # Evaluate Current ("Old") Policy by Getting Values & Log Probs
            _, log_probs = self.evaluate(obss, acts)

            # Stop Training Early if KL Divergence Exceeds Threshold
            log_ratio = np.exp(log_probs - ref_log_probs)
            kl_div = np.mean((np.exp(log_ratio) - 1) - log_ratio)

            if self.TARGET_KL != None and kl_div > self.TARGET_KL:
                print(f"Early stopping at epoch {epoch} due to reaching max kl: {kl_div:.2f}")
                break

            # Train Actor & Critic
            self.actor.fit(x=[obss/255, advs], y=acts, batch_size=self.batch_size, verbose=1)
            self.critic.fit(x=obss/255, y=rets, batch_size=self.batch_size, verbose=0)

            # Soft-Update Target Network
            actor_weights = np.array(self.actor.get_weights(), dtype=object)
            target_actor_weights = np.array(self.target_actor.get_weights(), dtype=object)
            new_weights = self.TARGET_ALPHA * actor_weights \
                            + (1-self.TARGET_ALPHA) * target_actor_weights
            self.target_actor.set_weights(new_weights)