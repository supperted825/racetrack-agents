from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import tensorflow.keras as K

import tensorflow_probability as tfp
tfd = tfp.distributions

import os
import sys
import csv
import random
import numpy as np
import datetime
import logging

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

            # PPO Hyperparameters
            self.GAE_GAMMA = opt.gae_gamma
            self.GAE_LAMBDA = opt.gae_lambda
            self.PPO_EPSILON = opt.ppo_epsilon
            self.ENTROPY = opt.ppo_entropy
            self.TARGET_KL = opt.target_kl
            self.ACTOR_SIGMA = opt.actor_sigma

            # Placeholder Variable for Prediction & Training
            self.ADV_PLACEHOLDER = np.zeros((1, 1))
            
            # Instantiate Models & Replay Buffer
            self.actor = self.create_actor(opt)
            self.critic = self.create_critic(opt)
            
            self.a_optimizer = Adam(learning_rate=self.lr)
            self.c_optimizer = Adam(learning_rate=self.lr)

            # Variables to Track Training Progress & Experience Replay Buffer
            self.total_steps = 0
            self.best = 0
            self.num_updates = 0
            self.kl_div = 0
            self.replay_memory = {
                "obss" : [], "acts" : [], "rews" : [], "mask" : []}

            # For Manual Logging (TensorBoard doesn't work with Eager Execution Disabled)
            time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
            self.logdir = f"logs/{self.name}-{time}"
            os.mkdir(self.logdir)
            
            # Python Logging Gives Easier-to-read Outputs
            logging.basicConfig(filename=self.logdir+'/log.log', format='%(message)s', filemode='w', level = logging.DEBUG)
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

            with open(self.logdir + '/log.csv', 'w+', newline ='') as file:
                write = csv.writer(file)
                write.writerow(['Total Steps', 'Avg Reward', 'Min Reward', 'Max Reward', 'Avg Ep Length'])
            
            with open(self.logdir + '/opt.txt', 'w+', newline ='') as file:
                args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
                for k, v in sorted(args.items()):
                    file.write('  %s: %s\n' % (str(k), str(v)))


    def create_actor(self, opt):
        """Construct Actor Neural Network"""

        # Define Model Inputs
        obs = Input(shape=opt.obs_dim)

        # Retrieve Model from Model File
        cnn = get_model(opt)
        x = cnn(obs)

        # Add FC Layers for PPO Actor
        for _ in range(opt.fc_layers):
            x = Dense(opt.fc_width, activation='relu')(x)

        # Model Outputs Mean Value for Each Continuous Action
        out = Dense(self.num_actions, activation='tanh')(x)

        # Compile Model with Custom PPO Loss
        model = Model(inputs=[obs], outputs=out)

        model.summary()

        return model


    def create_critic(self, opt):
        """Construct Critic with Similar Backbone as Actor"""

        # Retrieve Model Backbone from Model File
        model = get_model(opt)

        # Add FC Layers for PPO Critic
        for _ in range(opt.fc_layers):
            model.add(Dense(opt.fc_width))

        # Model Outputs Value Estimate for Each State
        model.add(Dense(1))

        model.summary()

        return model


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
    

    def process_replay(self, mem):
        """Process Espisode Information for Value & Advantages"""

        # Calculate Values & Log Probs
        vals, prbs = self.evaluate(mem["obss"], mem["acts"])

        # If Last Entry is Terminal State, Use Reward else V(s)
        if mem["mask"][-1] == 1:
            last_val = mem["rews"][-1]
        else:
            last_val = vals[-1]
        
        last_adv = 0
        g = self.GAE_GAMMA
        l = self.GAE_LAMBDA

        # Initialise Return Arrays with Appropriate Shapes
        obss = np.empty((len(mem["obss"]), *self.obs_dim))
        acts = np.empty((len(mem["obss"]), self.num_actions))
        advs = np.empty((len(mem["obss"]),))
        rets = np.empty((len(mem["obss"]),))
        
        for idx, value in enumerate(reversed(vals)):
            reward = mem["rews"][-idx]
            mask = mem["mask"][-idx]

            # Calculate Return & Advantage
            delta = reward + g * last_val * mask - value
            adv = delta + g * l * last_adv * mask
            ret = value + adv

            # Append to Output Arrays
            obss[-idx-1] = mem["obss"][-idx-1]
            acts[-idx-1] = mem["acts"][-idx-1]
            advs[-idx-1] = adv
            rets[-idx-1] = ret

            last_val = value if mask == 1 else ret
            last_adv = adv

        return obss, acts, advs, rets, prbs


    def collect_rollout(self, env, opt):
        """Collect Experiences from Environment for Training"""
        
        # For Logging of Agent Performance
        ep_rewards = []
        ep_lengths = []
        rollout_steps = self.batch_size * 12
        num_steps = 0

        while num_steps < rollout_steps:

            steps, done = 0, False
            obs = env.reset() if not opt.debug == 2 else env.reset().T
            ep_reward = 0

            while not done:

                # Get Action & Step Environment
                action = self.act(obs)
                obs, reward, done, _ = env.step(action)

                # Update Replay Memory
                if opt.debug == 2:
                    obs = obs.T 
                self.update_replay(obs, action, reward, done)

                # Increment Step Counters
                steps += 1
                num_steps += 1
                ep_reward += reward
                if num_steps == rollout_steps:
                    break
        
            ep_lengths.append(steps)
            ep_rewards.append(ep_reward)
        
        self.total_steps += rollout_steps

        # Log Memory Buffer Information & Write to CSV
        avg_ep_len = np.mean(ep_lengths)
        avg_reward = np.mean(ep_rewards)
        min_reward = np.min(ep_rewards)
        max_reward = np.max(ep_rewards)

        # Show Training Progress on Console
        logging.info(40*"-")
        logging.info(f"Total Steps: {self.total_steps}")
        logging.info(f"Average Reward: {avg_reward:.3f}")
        logging.info(f"Average Episode Length: {avg_ep_len}")
        logging.info(f"Num. Model Updates: {self.num_updates}")
        logging.info(f"Previous KL Div: {self.kl_div:.3f}")
        logging.info(40*"-")

        self.write_log(self.total_steps, reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, avg_ep_len=avg_ep_len)

        # Save Model if Average Reward is Greater than a Minimum & Better than Before
        if avg_reward >= np.max([opt.min_reward, self.best]) and opt.save_model:
            self.best = avg_reward
            self.actor.save(f'models/{self.name}_actor_best.model')
            self.critic.save(f'models/{self.name}_critic_best.model')
        
        return ep_rewards, ep_lengths


    def act(self, obs):
        """Get Action from Actor"""
        with tf.device('/cpu:0'):
            action = self.actor(np.expand_dims(obs/255, axis=0))
        return action
    

    def evaluate(self, obss, acts):
        """Produce Values & Log Probabilties for Given Batch of Observations"""

        # Predict Value & Prepare Action Outputs
        vals = self.critic(np.array(obss)/255)
        acts = np.expand_dims(np.array(acts).flatten(), axis=0)

        # Calculate Log Probabilities of Each Action
        dist = tfd.Normal(np.array(acts), self.ACTOR_SIGMA)
        log_probs = dist.log_prob(acts).numpy().squeeze()

        return vals, log_probs


    def compute_entropy(self, acts):
        """Compute Distribution Entropy for Entropy Loss"""
        dist = tfd.Normal(np.array(acts), self.ACTOR_SIGMA)
        entropy = tf.math.reduce_mean(dist.entropy())
        return entropy
    
    
    def learn(self, env, opt):
        """Run Rollout & Training Sequence"""
        
        while self.total_steps < 200 * opt.num_episodes:
            self.collect_rollout(env, opt)
            self.train()    
    

    def train(self):
        """Train Agent by Consuming Collected Memory Buffer"""

        # Process Returns & Advantages for Buffer Info
        buffer_obss, buffer_acts, buffer_advs, buffer_rets, buffer_prbs = self.process_replay(self.replay_memory)

        for idx, batch_idx in enumerate(range(0, len(buffer_obss), self.batch_size)): 

            # Go Through Buffer Batch Size at a Time
            obss = buffer_obss[batch_idx:batch_idx + self.batch_size]
            acts = buffer_acts[batch_idx:batch_idx + self.batch_size]
            advs = buffer_advs[batch_idx:batch_idx + self.batch_size]
            rets = buffer_rets[batch_idx:batch_idx + self.batch_size]
            prbs = buffer_prbs[batch_idx:batch_idx + self.batch_size]
            
            # Normalise Advantages & Reshape with Log Probs as Single Batch
            advs = (advs - advs.mean()) / (advs.std() + 1e-10)
            advs = np.expand_dims(advs, axis=1)
            prbs = np.expand_dims(prbs, axis=1)
            
            # Compute Entropy for Entropy Loss
            entropy = self.compute_entropy(acts)
            entropy = tf.cast(entropy, dtype=tf.float32)

            for epoch in range(self.epochs):
                
                # Cast Constant Inputs to Tensors
                acts = tf.constant(acts, tf.float32)
                advs = tf.constant(advs, tf.float32)
                rets = tf.constant(rets, tf.float32)
                prbs = tf.constant(prbs, tf.float32)
                
                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    
                    # Run Forward Passes on Models
                    a_pred = self.actor([obss/255], training=True)
                    v_pred = self.critic([obss/255], training=True)
                    
                    # Compute Respective Losses
                    c_loss = K.losses.mean_squared_error(rets, v_pred)
                    a_loss, ratios = self.PPO_loss(a_pred, acts, prbs, advs, entropy)

                # Compute Gradients & Apply to Model
                grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
                grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
                self.a_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
                self.c_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))
                
                # logging.info("Actor Loss: {}".format(a_loss))
                
                # Compute KL Divergence for Early Stopping
                self.kl_div = tf.reduce_mean((tf.math.exp(ratios) - 1) - ratios).numpy()

                if self.TARGET_KL != None and self.kl_div > self.TARGET_KL:
                    logging.info(f"Early stopping at epoch {epoch+1} due to reaching max kl: {self.kl_div:.3f}")
                    break
                
                self.num_updates += 1
                
            logging.info(f"Batch {idx+1}: KL Divergence of {self.kl_div:.3f}")
        
        self.clear_memory()


    @tf.function
    def PPO_loss(self, y_pred, acts, old_log_probs, advs, entropy):
        """Clipped PPO Loss for Actor"""
        
        # Get New Distributions & Log Probabilities of Actions
        new_dist = tfd.Normal(y_pred, self.ACTOR_SIGMA)
        new_log_probs = new_dist.log_prob(acts)

        # Calculate Ratio Between Old & New Policy
        ratios = tf.math.exp(new_log_probs - old_log_probs)

        # Clipped Actor Loss
        loss1 = advs * ratios
        loss2 = advs * tf.clip_by_value(ratios, 1 - self.PPO_EPSILON, 1 + self.PPO_EPSILON)
        actor_loss = tf.math.reduce_mean(-tf.math.minimum(loss1, loss2))

        # Entropy Bonus
        entropy_loss = - self.ENTROPY * entropy

        return actor_loss + entropy_loss, ratios