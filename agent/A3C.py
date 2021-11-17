from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import PolynomialDecay

import tensorflow_probability as tfp
import tensorflow as tf
import os
import sys
import csv
import numpy as np
import datetime
import logging
import threading

from threading import Thread
from multiprocessing import cpu_count
from racetrack_env import RaceTrackEnv
from .models import get_model


"""A3C Pseudocode"""
# Initial Policy Parameters and Initial Value Function parameters
# Initialise GLobal Policy
#     For Num Steps, Do:
#         Collect Trajectories by Running Policy in Environment for n Steps with Local Policy
#         Compute Advantages & Return Estimates with Value Function
#         Update Global Policy by Minimizing losses via RMSProp
#         Update Value Function by with MSE error via RMSprop


class ActorCritic(Model):
    """Actor-Critic Network Architecture"""
    
    def __init__(self, opt):
        super().__init__('ActorCriticModel')
        self.create_model(opt)
        self.num_actions = opt.num_actions
    
    
    def create_model(self, opt):
        """Build Model Layers & Architecture"""
        self.feature_extractor = get_model(opt)
        
        # Retrieve Post-Feature Extractor Dimensions
        for layer in self.feature_extractor.layers:
            feature_output_dim = layer.output_shape

        # Define Actor & Critic Networks
        self.actor_network  = Sequential()
        self.var_network    = Sequential()
        self.critic_network = Sequential()

        # Build Hidden Layers for Actor & Critic Output Heads
        for _ in range(opt.fc_layers):
            self.actor_network.add(Dense(opt.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
            self.var_network.add(Dense(opt.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
            self.critic_network.add(Dense(opt.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
        
        # Add Final Output Layers to Each Head
        self.actor_network.add(Dense(opt.num_actions, activation='tanh', kernel_initializer = 'glorot_uniform'))
        self.var_network.add(Dense(opt.num_actions, activation='softmax', kernel_initializer = 'glorot_uniform'))
        self.critic_network.add(Dense(1, kernel_initializer = 'glorot_uniform'))

        # Generate Passes & Compile Model
        self.actor_network.build(feature_output_dim)
        self.var_network.build(feature_output_dim)
        self.critic_network.build(feature_output_dim)


    def call(self, inputs):
        """Run Forward Pass"""
        feats = self.feature_extractor(inputs)
        action_output = self.actor_network(feats)
        var_output    = self.var_network(feats)
        value_output  = self.critic_network(feats)
        return action_output, var_output, tf.squeeze(value_output)


    def act(self, obs):
        """Get Action from Actor NN"""
        obs       = np.expand_dims(obs, axis=0)

        # Run Forward Passes
        feats   = self.feature_extractor(obs)
        mu      = self.actor_network(feats)
        var     = self.var_network(feats)

        # Calculate Action
        probability_density_func = tfp.distributions.Normal(mu, var)
        action = probability_density_func.sample(self.num_actions)
        action = tf.clip_by_value(action, -1, 1)

        return action.numpy()


class A3CAgent():
    """Master Agent for A3C"""
    
    def __init__(self, opt=None):
        # Instantiate Global Model & Parameters
        self.policy_global = ActorCritic(opt)
        self.num_workers = cpu_count() if opt.num_workers == None else opt.num_workers  
          

    def learn(self, env, opt):
        """Run Training Sequence"""
        workers = []

        for i in range(self.num_workers):
            env = RaceTrackEnv(opt)
            workers.append(A3C_worker(env, self.policy_global, i, opt))  

        for k, worker in enumerate(workers):
            print(f"Starting worker {k}")
            worker.start()

        for worker in workers:
            worker.join()


class A3C_worker(Thread):
    """Worker Agent for A3C"""

    # Set-up Global Variables Across Worker Threads
    global_best = 0
    global_episode = 0
    global_total_reward = []
    global_total_loss = []
    global_actor_loss = []
    global_critic_loss = []
    global_entropy_loss = []
    save_lock = threading.Lock()


    def __init__(self, env, policy_global, worker_idx, opt):

        # Configs & Hyperparameters
        Thread.__init__(self)
        self.name = "{}_{}_{} Actions".format(opt.agent, opt.arch, opt.num_actions)
        self.env = env
        self.lr = opt.lr
        self.num_episodes = opt.num_episodes
        self.num_actions = opt.num_actions
        self.obs_dim = opt.obs_dim
        self.update_global_freq = opt.update_global_freq
        self.worker_idx = worker_idx
        self.save_model = opt.save_model
        self.min_reward = opt.min_reward
        self.exp_dir = opt.exp_dir
        self.log_freq = opt.log_freq
        self.eval_freq = opt.eval_freq
        self.rmsprop_epsilon = opt.rmsprop_epsilon

        # Get Global Model     
        self.policy_global = policy_global

        # A3C Hyperparameters
        self.A3C_GAMMA    = opt.a3c_gamma
        self.ACTOR_COEF   = 1.0
        self.CRITIC_COEF  = 0.5
        self.ENTROPY_COEF = 0.01

        # Instantiate Local Model & Optimizer 
        self.policy_local  = ActorCritic(opt)
        lr_schedule = PolynomialDecay(self.lr, self.num_episodes * 200, end_learning_rate=0)
        self.optimizer = RMSprop(learning_rate=lr_schedule if opt.lr_decay else self.lr, epsilon=self.rmsprop_epsilon)
        
        # Manage Logging Properties
        time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        self.logdir = f"{opt.exp_dir}/log_{time}"
        try:
            os.makedirs(self.logdir)
        except OSError:
            pass

        # Python Logging Gives Easier-to-read Outputs
        logging.basicConfig(filename=self.logdir+'/log.log', format='%(message)s', filemode='w', level = logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        with open(self.logdir + '/log.csv', 'w+', newline ='') as file:
            write = csv.writer(file)
            write.writerow(['Total Steps', 'Avg Reward', 'Max Reward', 'Min Reward', 'Avg Actor Loss', 'Avg Critic Loss'])
            
        with open(self.logdir + '/opt.txt', 'w+', newline ='') as file:
            args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
            for k, v in sorted(args.items()):
                file.write('  %s: %s\n' % (str(k), str(v)))


    def write_log(self):
        """Record to Log File"""
        
        logging.info(40*"-")
        logging.info(f"Worker: {self.worker_idx}")
        logging.info(f"Global Steps: {A3C_worker.global_episode}")
        logging.info(f"Rewards: {self.ep_reward}")
        logging.info(f"Episode Steps: {self.ep_lengths}")

        if len(self.a_losses) > 0 or len(self.c_losses) > 0:
            logging.info(f"Total Loss: {np.mean(self.total_losses):.5f}")
            logging.info(f"Actor Loss: {np.mean(self.a_losses):.5f}")
            logging.info(f"Critic Loss: {np.mean(self.c_losses):.5f}")
            
        logging.info(40*"-")


    def write_csv(self, step, g_mean_rews, g_max_rews, g_min_rews, g_a_loss, g_c_loss):
        """Record to CSV File"""

        line = [step, g_mean_rews, g_max_rews, g_min_rews, g_a_loss, g_c_loss]
        with open(self.logdir + '/log.csv', 'a', newline ='') as file:
            write = csv.writer(file)
            write.writerow(line)


    def replay_memory(self, step, obs, action, reward):
        """Memory Record for A3C Worker to Update Global Network"""

        self.obss_batch[step] = obs
        self.acts_batch[step] = action
        self.rews_batch[step] = reward

        return self.obss_batch, self.acts_batch, self.rews_batch


    def reset_memory(self):
        """Reset Worker Agent Replay Memory Buffer"""

        # Initialize with proper shape
        self.obss_batch = np.zeros((self.update_global_freq, *self.obs_dim))
        self.acts_batch = np.zeros((self.update_global_freq, self.num_actions))
        self.rews_batch = np.zeros(self.update_global_freq)


    def td_target(self, rewards, next_v, done):
        """Calculate TD-target"""
        td_targets = np.zeros_like(rewards)

        if done:
            ret = 0
        else:
            ret = next_v
        
        for i in reversed(range(0, len(rewards))):
            ret = rewards[i] + self.A3C_GAMMA * ret
            td_targets[i] = ret

        return td_targets


    def run(self):
        """Run A3C Worker"""

        # Losses Logging
        self.total_losses = []
        self.a_losses     = []
        self.c_losses     = []
        self.e_losses     = []

        while A3C_worker.global_episode < self.num_episodes:
            
            self.ep_reward = 0
            self.ep_lengths = 0
            update_counter, done = 0, False
            obs = self.env.reset()
            self.reset_memory()

            while not done:

                # Get Action & Step Environment
                action = self.policy_local.act(obs)

                next_obs, reward, done, _ = self.env.step(action.reshape(self.num_actions, ))

                # Update Replay Memory
                obss, acts, rews = self.replay_memory(update_counter, obs, action, reward)

                self.ep_reward += reward

                # Update Global with Local Gradient if Worker Reach Global Update Frequency
                if self.update_global_freq <= len(rews) or done:         

                    next_obss   = np.expand_dims(next_obs, axis=0) 

                    next_v     = self.policy_local.critic_network(self.policy_local.feature_extractor(next_obss), training=False).numpy()

                    td_targets = self.td_target(rews, next_v, done)
                    td_targets = np.array(td_targets)

                    values     = self.policy_local.critic_network(self.policy_local.feature_extractor(obss), training=False).numpy()

                    advs       = td_targets - values
                    advs       = np.array(advs)
                    
                    # Normalise Advantages
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                    # Cast Constant Inputs to Tensors
                    acts       = tf.constant(acts, tf.float32)
                    advs       = tf.constant(advs, tf.float32)
                    td_targets = tf.constant(td_targets, tf.float32)

                    # Calculate Gradient wrt to Local Actor Model
                    with tf.GradientTape() as tape:
                        
                        mu, var, v_pred = self.policy_local(obss, training=True) 

                        # Compute Actor Loss & Critic Loss
                        a_loss = self.actor_loss(mu, var, acts, advs)
                        c_loss = self.critic_loss(v_pred, td_targets)

                        total_loss = self.ACTOR_COEF * a_loss + self.CRITIC_COEF * c_loss

                        # Compute Gradients & Apply to Global Model
                        grad  = tape.gradient(total_loss,self.policy_local.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO) 
                        grad, _ = tf.clip_by_global_norm(grad, 0.5)
                        self.optimizer.apply_gradients(zip(grad, self.policy_global.trainable_variables))

                        # Update Local Model with New Global Weights
                        self.policy_local.set_weights(self.policy_global.get_weights())

                    # Losses Logging
                    self.total_losses.append(total_loss.numpy())
                    self.a_losses.append(a_loss.numpy())
                    self.c_losses.append(c_loss.numpy())

                    # Reset Variables After Global Updates
                    update_counter = 1 
                    self.reset_memory()

                # Increment All Counter
                self.ep_lengths += 1 
                update_counter += 1 
                obs = next_obs 

            # Value for Log and CSV
            A3C_worker.global_episode += 1
            A3C_worker.global_total_reward.append(self.ep_reward)
            A3C_worker.global_total_loss.append(np.mean(self.total_losses))
            A3C_worker.global_actor_loss.append(np.mean(self.a_losses))
            A3C_worker.global_critic_loss.append(np.mean(self.c_losses))

            # Calculate Value Across all Workers for CSV
            avg_global_reward = round(np.mean(A3C_worker.global_total_reward[-self.log_freq:]),3)
            max_global_reward = round(np.max(A3C_worker.global_total_reward[-self.log_freq:]),3)
            min_global_reward = round(np.min(A3C_worker.global_total_reward[-self.log_freq:]),3)
            avg_actor_loss    = round(np.mean(A3C_worker.global_actor_loss[-self.log_freq:]),3)
            avg_critic_loss   = round(np.mean(A3C_worker.global_critic_loss[-self.log_freq:]),3)

            # Write to Log and CSV
            if A3C_worker.global_episode % self.log_freq == 0:
                self.write_log()
                self.write_csv(A3C_worker.global_episode, avg_global_reward, max_global_reward, min_global_reward, \
                               avg_actor_loss, avg_critic_loss)

            # Save Model if Episode Reward is Greater than a Minimum & Better than Before
            if self.ep_reward >= np.max([self.min_reward, A3C_worker.global_best]) and self.save_model:
                with A3C_worker.save_lock:
                    logging.info(f"Saving Model! Worker: {self.worker_idx}, Episode Score: {self.ep_reward}")
                    self.policy_local.save(f'{self.exp_dir}/R{self.ep_reward:.0f}.model')
                    A3C_worker.global_best = self.ep_reward

            # Save Model for Every 500 Episodes to Check Training Progress
            if A3C_worker.global_episode % 500 == 0:
                global_episode = A3C_worker.global_episode
                with A3C_worker.save_lock:
                    logging.info(f"Saving Every 500th Model! Worker: {self.worker_idx}, Episode Score: {self.ep_reward}")
                    self.policy_local.save(f'{self.exp_dir}/checkpoint_{global_episode}.model')


    def actor_loss(self, mu, var, action, adv):
        """Calculate Actor Loss"""

        probability_density_func = tfp.distributions.Normal(mu, var)
        entropy         = probability_density_func.entropy()
        log_prob        = probability_density_func.log_prob(action) 
        expected_value  = tf.multiply(log_prob, adv)
        ev_with_entropy = expected_value + entropy * self.ENTROPY_COEF
        actor_loss      = tf.reduce_sum(-ev_with_entropy)

        return actor_loss


    def critic_loss(self, v_pred, td_targets):
        """Mean-Squared-Error for Critic"""

        mse = tf.keras.losses.MeanSquaredError()
        critic_loss = mse(td_targets, v_pred)
        
        return critic_loss
