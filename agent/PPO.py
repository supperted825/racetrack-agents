from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal

import tensorflow as tf
import tensorflow.keras as K

import tensorflow_probability as tfp
tfd = tfp.distributions

import os
import sys
import csv
import numpy as np
import datetime
import logging

from .models import get_model


"""PPO Pseudocode"""
# Initial Policy Parameters and Initial Value Function parameters
#     For Num Steps, Do:
#         Collect Trajectories by Running Policy in Environment for n Steps
#         Compute Advantages & Return Estimates with Value Function
#         Update Policy by Maximizing PPO-clip Objective via Adam
#         Update Value Function by Regression with MSE error via Adam


class PPOAgent():
    """Proximal Policy Optimisation Agent with Clipping"""

    def __init__(self, opt=None):

            # Configs & Hyperparameters
            self.name = "{}_{}_{}Actions".format(opt.agent, opt.arch, opt.num_actions)
            self.lr = opt.lr
            self.epochs = opt.num_epochs
            self.batch_size = opt.batch_size
            self.num_actions = opt.num_actions
            self.obs_dim = opt.obs_dim
            self.memory_size = self.batch_size * 12
            self.mode = opt.obs_dim[0]

            # PPO Hyperparameters
            self.GAE_GAMMA = opt.gae_gamma
            self.GAE_LAMBDA = opt.gae_lambda
            self.PPO_EPSILON = opt.ppo_epsilon
            self.ENTROPY = opt.ppo_entropy
            self.TARGET_KL = opt.target_kl
            self.ACTOR_SIGMA = opt.actor_sigma

            # Instantiate Model & Optimizer
            self.policy = self.create_model(opt)
            self.optimizer = Adam(learning_rate=self.lr)

            # Variables to Track Training Progress & Experience Replay Buffer
            self.total_steps = 0
            self.best = 0
            self.num_updates = 0
            self.losses = []
            
            # Initialise Replay Memory Buffer
            self.reset_memory()

            # Manage Logging Properties
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


    def create_model(self, opt):
        """Constructs Policy Network with Shared Backbone & Actor+Critic Heads"""
        
        # Define Model Inputs
        obs = Input(shape=opt.obs_dim)
        
        # Grab Shared Backbone from Models
        self.feature_extractor = get_model(opt)
        
        self.actor_network = Sequential()
        self.critic_network = Sequential()
        
        # Build Hidden Layers for Actor & Critic Output Heads
        for _ in range(opt.fc_layers):
            self.actor_network.add(Dense(opt.fc_width, activation='tanh', kernel_initializer=Orthogonal(0.01)))
            self.critic_network.add(Dense(opt.fc_width, activation='tanh', kernel_initializer=Orthogonal(1)))

        # Add Final Output Layers to Each Head
        self.actor_network.add(Dense(self.num_actions, activation='tanh', kernel_initializer=Orthogonal(0.01)))
        self.critic_network.add(Dense(1, activation='tanh', kernel_initializer=Orthogonal(1)))
        
        # Generate Passes & Compile Model
        feats = self.feature_extractor(obs)
        action_output = self.actor_network(feats)
        value_output = self.critic_network(feats)
        
        model = Model(inputs=[obs], outputs=[action_output,value_output])
        model.summary()
        
        return model


    def write_log(self, step, **logs):
        """Write Episode Information to CSV File"""
        line = [step] + [value for value in logs.values()]
        with open(self.logdir + '/log.csv', 'a', newline ='') as file:
            write = csv.writer(file)
            write.writerow(line)

    
    def callback(self, avg_reward, avg_ep_len):
        """Write Training Information to Console"""
        
        logging.info(40*"-")
        logging.info(f"Total Steps: {self.total_steps}")
        logging.info(f"Average Reward: {avg_reward:.3f}")
        logging.info(f"Average Episode Length: {avg_ep_len:.3f}")
        logging.info(f"Num. Model Updates: {self.num_updates}")
        
        if len(self.losses) > 0:
            logging.info(f"Total Loss: {np.mean(self.losses):.5f}")
            logging.info(f"Actor Loss: {np.mean(self.a_losses):.5f}")
            logging.info(f"Critic Loss: {np.mean(self.c_losses):.5f}")
            logging.info(f"Entropy Loss: {np.mean(self.e_losses):.5f}")
            logging.info(f"Approx KL Div: {np.mean(self.kl_divs):.3f}")
            
        logging.info(40*"-")


    def reset_memory(self):
        """Reset Agent Replay Memory Buffer"""
        self.replay_memory = {
            "obss" : np.zeros((self.memory_size, *self.obs_dim,)),
            "acts" : np.zeros((self.memory_size, self.num_actions,)),
            "rews" : np.zeros(self.memory_size),
            "mask" : np.zeros(self.memory_size)
        }
        self.last_vals = np.zeros((self.memory_size+1))


    def update_replay(self, step, obs, act, rew, done):
        """Record Stepwise Episode Information with Critic Output"""
        self.replay_memory["obss"][step] = obs if self.mode == 2 else obs/255
        self.replay_memory["acts"][step] = act
        self.replay_memory["rews"][step] = rew
        self.replay_memory["mask"][step] = 0 if done else 1
    

    def process_replay(self, mem):
        """Process Espisode Information for Value & Advantages"""

        # Calculate Values & Log Probs
        vals, prbs = self.evaluate(mem["obss"], mem["acts"])

        last_adv = 0
        g = self.GAE_GAMMA
        l = self.GAE_LAMBDA

        # Initialise Return Arrays with Appropriate Shapes
        advs = np.empty((self.memory_size,))
        rets = np.empty((self.memory_size,))
        
        for idx in reversed(range(len(vals))):
            
            # Prepare Variables for this Step
            reward = mem["rews"][idx]
            mask = mem["mask"][idx]
            value = vals[idx]
            
            # Retrieve Next Step Value
            if mask == 0:
                last_val = self.last_vals[idx+1]
            else:
                last_val = vals[idx+1]
            
            # Calculate Return & Advantage
            delta = reward + g * last_val * mask - value
            adv = delta + g * l * last_adv * mask
            ret = value + adv

            # Append to Output Arrays
            advs[idx] = adv
            rets[idx] = ret

            last_adv = adv
        
        return mem["obss"], mem["acts"], advs, rets, prbs


    def collect_rollout(self, env, opt):
        """Collect Experiences from Environment for Training"""
        
        # For Logging of Agent Performance
        ep_rewards = []
        ep_lengths = []
        num_steps = 0

        while num_steps != self.memory_size - 1:

            steps, done = 0, False
            last_obs = env.reset() if not opt.debug == 2 else env.reset().T
            ep_reward = 0

            while True:

                # Get Action & Step Environment
                action = self.act(last_obs)
                new_obs, reward, done, _ = env.step(action)

                if opt.debug == 2:
                    obs = obs.T
                
                # Break Early if Rollout has Been Filled, Mark Step as End
                if done or num_steps == self.memory_size - 1:
                    done = True
                    self.update_replay(num_steps, last_obs, action, reward, done)
                    break
                
                self.update_replay(num_steps, last_obs, action, reward, done)
                last_obs = new_obs
                
                # Increment Step Counters
                steps += 1
                num_steps += 1
                ep_reward += reward
            
            # Calculate Last Value for Finished Episode
            new_obs = np.expand_dims(new_obs, axis=0)
            self.last_vals[num_steps] = self.critic_network(self.feature_extractor(new_obs, training=False)).numpy()

            ep_lengths.append(steps+1)
            ep_rewards.append(ep_reward)
        
        self.total_steps += num_steps + 1

        # Log Memory Buffer Information & Write to CSV
        avg_ep_len = np.mean(ep_lengths)
        avg_reward = np.mean(ep_rewards)
        min_reward = np.min(ep_rewards)
        max_reward = np.max(ep_rewards)

        # Show Training Progress on Console
        self.callback(avg_reward, avg_ep_len)

        self.write_log(self.total_steps, reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, avg_ep_len=avg_ep_len)

        # Save Model if Average Reward is Greater than a Minimum & Better than Before
        if avg_reward >= np.max([opt.min_reward, self.best]) and opt.save_model:
            self.best = avg_reward
            self.policy.save(f'models/{self.name}_best.model')
        
        return ep_rewards, ep_lengths


    def act(self, obs, optimal=False):
        """Get Action from Actor Network"""
        
        obs = obs if self.mode == 2 else obs/255
        with tf.device('/cpu:0'):
            obs = np.expand_dims(obs, axis=0)
            feats = self.feature_extractor(obs, training=False)
            action = self.actor_network(feats)

        return action.numpy()
    

    def evaluate(self, obss, acts):
        """Produce Values & Log Probabilties for Given Batch of Observations"""

        # Predict Value & Prepare Action Outputs
        obss = np.array(obss)
        feats = self.feature_extractor(obss, training=False)
        vals = self.critic_network(feats).numpy()
        acts = np.expand_dims(np.array(acts).flatten(), axis=0)

        # Calculate Log Probabilities of Each Action
        dist = tfd.Normal(acts, self.ACTOR_SIGMA)
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
        
        # Generate Indices for Sampling
        ind = np.random.permutation(self.memory_size)
        
        # Train Logging
        self.losses = []
        self.e_losses = []
        self.a_losses = []
        self.c_losses = []
        self.kl_divs = []

        for epoch in range(self.epochs):
            
            for batch_idx in range(0, len(buffer_obss), self.batch_size): 

                # Go Through Buffer Batch Size at a Time
                obss = buffer_obss[ind[batch_idx:batch_idx + self.batch_size]]
                acts = buffer_acts[ind[batch_idx:batch_idx + self.batch_size]]
                advs = buffer_advs[ind[batch_idx:batch_idx + self.batch_size]]
                rets = buffer_rets[ind[batch_idx:batch_idx + self.batch_size]]
                prbs = buffer_prbs[ind[batch_idx:batch_idx + self.batch_size]]
                
                # Normalise Advantages
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                
                # Reshape Advantages with Log Probs as Single Batch
                advs = np.expand_dims(advs, axis=1)
                prbs = np.expand_dims(prbs, axis=1)
                
                # Compute Entropy for Entropy Loss
                entropy = self.compute_entropy(acts)
                entropy = tf.cast(entropy, dtype=tf.float32)

                # Cast Constant Inputs to Tensors
                acts = tf.constant(acts, tf.float32)
                advs = tf.constant(advs, tf.float32)
                rets = tf.constant(rets, tf.float32)
                prbs = tf.constant(prbs, tf.float32)
                
                with tf.GradientTape() as tape:
                    
                    # Run Forward Passes on Models
                    a_pred, v_pred = self.policy([obss], training=True)
                    
                    # Compute Respective Losses
                    c_loss = self.critic_loss(v_pred, rets)
                    a_loss, new_log_probs = self.PPO_loss(a_pred, acts, prbs, advs)
                    
                    # Entropy Bonus
                    e_loss = - self.ENTROPY * entropy
                    
                    tot_loss = 0.5 * c_loss + a_loss + e_loss
                
                # Compute Gradients & Apply to Model
                gradients = tape.gradient(tot_loss, self.policy.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
                
                # Compute KL Divergence for Early Stopping
                kl_div = 0.5 * tf.reduce_mean(tf.square(new_log_probs - prbs))

                if self.TARGET_KL != None and self.kl_div > self.TARGET_KL:
                    logging.info(f"Early stopping at epoch {epoch+1} due to reaching max kl: {self.kl_div:.3f}")
                    break
                
                # Logging
                self.losses.append(tot_loss.numpy())
                self.e_losses.append(e_loss.numpy())
                self.a_losses.append(a_loss.numpy())
                self.c_losses.append(c_loss.numpy())
                self.kl_divs.append(kl_div)
                
                self.num_updates += 1
        
        self.reset_memory()


    @tf.function
    def PPO_loss(self, y_pred, acts, old_log_probs, advs):
        """PPO-Clip Loss for Actor"""
        
        # Get New Distributions & Log Probabilities of Actions
        new_dist = tfd.Normal(y_pred, self.ACTOR_SIGMA)
        new_log_probs = new_dist.log_prob(acts)

        # Calculate Ratio Between Old & New Policy
        ratios = tf.math.exp(new_log_probs - old_log_probs)

        # Clipped Actor Loss
        loss1 = advs * ratios
        loss2 = advs * tf.clip_by_value(ratios, 1 - self.PPO_EPSILON, 1 + self.PPO_EPSILON)
        actor_loss = tf.math.reduce_mean(-tf.math.minimum(loss1, loss2))
        
        return actor_loss, new_log_probs
    
    
    @tf.function
    def critic_loss(self, y_pred, rets):
        """Mean-Squared-Error for Critic"""
        critic_loss = tf.math.squared_difference(rets, y_pred)
        critic_loss = tf.math.reduce_mean(critic_loss)
        return critic_loss