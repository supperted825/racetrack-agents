from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import PolynomialDecay, ExponentialDecay
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
            self.memory_size = 12 * opt.batch_size
            self.target_steps = 200 * opt.num_episodes
            self.mode = opt.obs_dim[0]

            # PPO Hyperparameters
            self.GAE_GAMMA = opt.gae_gamma
            self.GAE_LAMBDA = opt.gae_lambda
            self.PPO_EPSILON = opt.ppo_epsilon
            self.TARGET_KL = opt.target_kl
            self.ENTROPY = 0.001

            # Instantiate Model & Optimizer
            self.policy = PolicyModel(opt)
            lr_schedule = PolynomialDecay(self.lr, self.target_steps // self.batch_size * self.epochs, end_learning_rate=0)
            self.optimizer = Adam(learning_rate=lr_schedule if opt.lr_decay else self.lr)

            # Variables to Track Training Progress & Experience Replay Buffer
            self.total_steps = 0
            self.best = 0
            self.num_updates = 0
            self.losses = []
            
            # Initialise Replay Memory Buffer
            self.reset_memory()

            # Manage Logging Properties
            time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
            self.logdir = f"{opt.exp_dir}/log_{time}"
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
                    
            # Load Last Model if Resume is Specified
            if opt.resume:
                weights2load = K.models.load_model(f'{opt.exp_dir}/last_best.model').get_weights()
                self.policy.set_weights(weights2load)
                logging.info("Loaded Weights from Last Best Model!")


    def write_log(self, step, **logs):
        """Write Episode Information to CSV File"""
        line = [step] + [value for value in logs.values()]
        with open(self.logdir + '/log.csv', 'a', newline ='') as file:
            write = csv.writer(file)
            write.writerow(line)

    
    def callback(self, avg_reward):
        """Write Training Information to Console"""
        
        self.avg_ep_len = self.memory_size / self.episode_counter
        
        logging.info(40*"-")
        logging.info(f"Total Steps: {self.total_steps}")
        logging.info(f"Average Reward: {avg_reward:.3f}")
        logging.info(f"Average Episode Length: {self.avg_ep_len:.3f}")
        logging.info(f"Num. Model Updates: {self.num_updates}")
        
        if len(self.losses) > 0:
            logging.info(f"Total Loss: {np.mean(self.losses):.5f}")
            logging.info(f"Actor Loss: {np.mean(self.a_losses):.5f}")
            logging.info(f"Critic Loss: {np.mean(self.c_losses):.5f}")
            logging.info(f"Entropy Loss: {np.mean(self.e_losses):.5f}")
            logging.info(f"Approx KL Div: {np.mean(self.kl_divs):.3f}")
            logging.info(f"Log Std: {np.exp(self.policy.log_std.numpy()).squeeze():.3f}")
            
        logging.info(40*"-")


    def learn(self, env, opt):
        """Run Rollout & Training Sequence"""
        
        self.last_obs = env.reset()
        while self.total_steps < self.target_steps:
            self.collect_rollout(env, opt)
            self.train()
        self.policy.save(f'{opt.exp_dir}/model_last.model')
    

    def reset_memory(self):
        """Reset Agent Replay Memory Buffer"""
        self.replay_memory = {
            "obss" : np.zeros((self.memory_size, *self.obs_dim,)),
            "acts" : np.zeros((self.memory_size, self.num_actions,)),
            "rews" : np.zeros(self.memory_size),
            "vals" : np.zeros(self.memory_size),
            "prbs" : np.zeros(self.memory_size),
            "mask" : np.zeros(self.memory_size)
        }
        self.last_vals = np.zeros((self.memory_size+1))


    def update_replay(self, step, obs, act, rew, val, prb, done):
        """Record Stepwise Episode Information with Critic Output"""
        self.replay_memory["obss"][step] = obs/255
        self.replay_memory["acts"][step] = act
        self.replay_memory["rews"][step] = rew
        self.replay_memory["vals"][step] = val
        self.replay_memory["prbs"][step] = prb
        self.replay_memory["mask"][step] = 0 if done else 1
    

    def collect_rollout(self, env, opt):
        """Collect Experiences from Environment for Training"""
        
        # For Logging of Agent Performance
        ep_rewards = []
        num_steps = 0
        self.episode_counter = 0

        while num_steps != self.memory_size - 1:

            steps, done = 0, False
            self.last_obs = env.reset() if not opt.debug == 2 else env.reset().T
            ep_reward = 0

            while True:

                # Get Action & Step Environment
                action, value, logp = self.policy.act(self.last_obs)
                new_obs, reward, done, _ = env.step(action)

                if opt.debug == 2:
                    new_obs = new_obs.T
                
                # Break Early if Rollout has Been Filled, Mark Step as End
                if done or num_steps == self.memory_size - 1:
                    self.update_replay(num_steps, self.last_obs, action, reward, value, logp, done)
                    self.episode_counter += 1
                    break
                
                self.update_replay(num_steps, self.last_obs, action, reward, value, logp, done)
                self.last_obs = new_obs
                
                # Increment Step Counters
                steps += 1
                num_steps += 1
                ep_reward += reward
            
            # Calculate Last Value for Finished Episode
            new_obs = np.expand_dims(new_obs, axis=0)
            self.last_vals[num_steps] = self.policy.critic_network(self.policy.feature_extractor(new_obs, training=False)).numpy()

            ep_rewards.append(ep_reward)
        
        self.total_steps += num_steps + 1

        # Log Memory Buffer Information & Write to CSV
        avg_reward = np.mean(ep_rewards)
        min_reward = np.min(ep_rewards)
        max_reward = np.max(ep_rewards)

        # Show Training Progress on Console
        self.callback(avg_reward)

        self.write_log(self.total_steps, reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, avg_ep_len=self.avg_ep_len)

        # Save Model if Average Reward is Greater than a Minimum & Better than Before
        if avg_reward >= np.max([opt.min_reward, self.best]) and opt.save_model:
            self.best = avg_reward
            self.policy.save(f'{opt.exp_dir}/last_best.model')
        
        if self.best > 120 and self.TARGET_KL == None:
            logging.info("Decaying PPO Clip & Learning Rate!")
            self.PPO_EPSILON = 0.1
            #self.optimizer.learning_rate.assign(self.lr/10)
            #self.TARGET_KL = 0.01


    def process_replay(self, mem):
        """Process Espisode Information for Value & Advantages"""

        # Calculate Values & Log Probs
        vals = mem["vals"].flatten()
        prbs = mem["prbs"].flatten()

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
            if idx == len(vals) -1:
                last_val = self.last_vals[idx+1]
            else:
                last_val = vals[idx+1].squeeze()
            
            # Calculate Return & Advantage
            delta = reward + g * last_val * mask - value
            adv = delta + g * l * last_adv * mask
            ret = value + adv

            # Append to Output Arrays
            advs[idx] = adv
            rets[idx] = ret

            last_adv = adv
        
        return mem["obss"], mem["acts"], advs, rets, prbs
    

    def train(self):
        """Train Agent by Consuming Collected Memory Buffer"""

        # Process Returns & Advantages for Buffer Info
        buffer_obss, buffer_acts, buffer_advs, buffer_rets, buffer_prbs = self.process_replay(self.replay_memory)
        
        # Generate Indices for Sampling
        ind = np.random.permutation(self.memory_size)
        
        # Train Logging
        self.losses = []
        self.a_losses = []
        self.c_losses = []
        self.e_losses = []
        self.kl_divs = []

        for epoch in range(self.epochs):
            
            for batch_idx in range(0, self.memory_size, self.batch_size): 

                # Go Through Buffer Batch Size at a Time
                obss = buffer_obss[ind[batch_idx:batch_idx + self.batch_size]]
                acts = buffer_acts[ind[batch_idx:batch_idx + self.batch_size]]
                advs = buffer_advs[ind[batch_idx:batch_idx + self.batch_size]]
                rets = buffer_rets[ind[batch_idx:batch_idx + self.batch_size]]
                prbs = buffer_prbs[ind[batch_idx:batch_idx + self.batch_size]]
                
                # Normalise Advantages
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                
                # Cast Constant Inputs to Tensors
                acts = tf.constant(acts, tf.float32)
                advs = tf.constant(advs, tf.float32)
                rets = tf.constant(rets, tf.float32)
                prbs = tf.constant(prbs, tf.float32)
                
                with tf.GradientTape() as tape:
                    
                    # Run Forward Passes on Models & Get New Log Probs
                    a_pred, v_pred = self.policy(obss)
                    new_log_probs = self.policy.logp(a_pred, acts)

                    # Calculate Ratio Between Old & New Policy
                    ratios = tf.math.exp(new_log_probs - prbs)
                    
                    # Clipped Actor Loss
                    loss1 = advs * ratios
                    loss2 = advs * tf.clip_by_value(ratios, 1 - self.PPO_EPSILON, 1 + self.PPO_EPSILON)
                    a_loss = tf.math.reduce_mean(-tf.math.minimum(loss1, loss2))
                    
                    # Entropy Loss
                    entropy = self.policy.entropy()
                    e_loss = -tf.math.reduce_mean(entropy)
                    
                    # Value Loss
                    c_loss = tf.math.reduce_mean(tf.square(v_pred - rets))
                    
                    tot_loss = 0.5 * c_loss + a_loss + self.ENTROPY * e_loss
                
                # Compute KL Divergence for Early Stopping Before Backprop
                kl_div = 0.5 * tf.reduce_mean(tf.square(new_log_probs - prbs))

                if self.TARGET_KL != None and kl_div > self.TARGET_KL:
                    logging.info(f"Early stopping at epoch {epoch+1} due to reaching max kl: {kl_div:.3f}")
                    break
                
                # Compute Gradients & Apply to Model
                gradients = tape.gradient(tot_loss, self.policy.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
                
                # learning_rate = (1 - self.total_steps / self.target_steps + 5e-8) * self.lr
                # self.optimizer.lr.assign(learning_rate)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
                
                # Logging
                self.losses.append(tot_loss.numpy())
                self.a_losses.append(a_loss.numpy())
                self.c_losses.append(c_loss.numpy())
                self.e_losses.append(e_loss.numpy())
                self.kl_divs.append(kl_div)
                
                self.num_updates += 1
        
        self.reset_memory()
    

class PolicyModel(Model):
    """Actor Critic Policy Model for PPO"""
    
    
    def __init__(self, opt):
        """Pass Model Parameters from Opt & Initialise Learnable Log Std Param"""
        super().__init__('PolicyModel')
        self.build_model(opt)
        self.log_std = tf.Variable(initial_value=0*np.ones(opt.num_actions, dtype=np.float32))
        
    
    def build_model(self, opt):
        """Build Model Layers & Architecture"""
        
        self.feature_extractor = get_model(opt)
        
        # Retrieve Post-Feature Extractor Dimensions
        for layer in self.feature_extractor.layers:
            feature_output_dim = layer.output_shape

        # Define Actor & Critic Networks
        self.actor_network = Sequential()
        self.critic_network = Sequential()
        
        for _ in range(opt.fc_layers):
            self.actor_network.add(Dense(opt.fc_width, activation='tanh', kernel_initializer=Orthogonal(np.sqrt(2))))
            self.critic_network.add(Dense(opt.fc_width, activation='tanh', kernel_initializer=Orthogonal(np.sqrt(2))))

        self.actor_network.add(Dense(opt.num_actions, activation='tanh', kernel_initializer=Orthogonal(0.01)))
        self.critic_network.add(Dense(1, activation='tanh',kernel_initializer=Orthogonal(1)))
        
        self.actor_network.build(feature_output_dim)
        self.critic_network.build(feature_output_dim)

    
    def call(self, inputs):
        """Run Forward Pass on Actor Network Only"""
        feats = self.feature_extractor(inputs)
        action_output = self.actor_network(feats)
        value_output = self.critic_network(feats)
        return action_output, tf.squeeze(value_output)
    
        
    def act(self, obss):
        """Get Actions, Values & Log Probs During Experience Collection"""
        
        obss = np.expand_dims(obss, axis=0) / 255     
        
        # Run Forward Passes
        feats = self.feature_extractor(obss)
        a_pred = self.actor_network(feats)
        v_pred = self.critic_network(feats)
        
        # Calcualte Log Probabilities
        std = tf.exp(self.log_std)
        action = a_pred + tf.random.normal(tf.shape(a_pred)) * std
        action = tf.clip_by_value(action, -1, 1)
        logp_t = self.logp(action, a_pred)
        
        return action.numpy(), v_pred.numpy().squeeze(), logp_t.numpy().squeeze()
        
        
    def logp(self, x, mu):
        """Return Log Probability of Action Given Distribution Parameters"""
        pre_sum = -0.5 * (((x - mu) / (tf.exp(self.log_std) + 1e-8))**2 + 2 * self.log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis= -1)
        
    
    def entropy(self):
        """Return Entropy of Policy Distribution"""
        entropy = tf.reduce_sum(self.log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return entropy