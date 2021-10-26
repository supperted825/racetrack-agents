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
import threading
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


class PPOMultiAgent():
    """Proximal Policy Optimisation Agent with Clipping, with Workers"""

    def __init__(self, opt=None):

            # Configs & Hyperparameters
            self.name = "{}_{}_{}Actions".format(opt.agent, opt.arch, opt.num_actions)
            self.lr = opt.lr
            self.epochs = opt.num_epochs
            self.batch_size = opt.batch_size
            self.num_actions = opt.num_actions
            self.obs_dim = opt.obs_dim
            
            # Training Parameters
            self.num_workers = 1
            self.memory_size = 32
            self.target_steps = 200 * opt.num_episodes
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
            lr_schedule = PolynomialDecay(self.lr, self.target_steps // self.batch_size * self.epochs, end_learning_rate=0)
            self.optimizer = Adam(learning_rate=lr_schedule if opt.lr_decay else self.lr)

            # Variables to Track Training Progress & Experience Replay Buffer
            self.total_steps = self.num_updates = self.best = 0
            self.min_reward = self.max_reward = None
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
            self.actor_network.add(Dense(opt.fc_width, activation='tanh', kernel_initializer=Orthogonal(np.sqrt(2))))
            self.critic_network.add(Dense(opt.fc_width, activation='tanh', kernel_initializer=Orthogonal(np.sqrt(2))))

        # Add Final Output Layers to Each Head
        self.actor_network.add(Dense(self.num_actions, activation='tanh', kernel_initializer=Orthogonal(0.01)))
        self.critic_network.add(Dense(1, kernel_initializer=Orthogonal(1)))
        
        # Generate Passes & Compile Model
        feats = self.feature_extractor(obs)
        action_output = self.actor_network(feats)
        value_output = self.critic_network(feats)
        
        model = Model(inputs=[obs], outputs=[action_output,value_output])
        
        return model


    def write_log(self, step, **logs):
        """Write Episode Information to CSV File"""
        line = [step] + [value for value in logs.values()]
        with open(self.logdir + '/log.csv', 'a', newline ='') as file:
            write = csv.writer(file)
            write.writerow(line)


    def env_callback(self, opt):
        """Write Rollout Information to Console"""
        
        avg_ep_len = self.memory_size / self.num_eps
        avg_reward = self.tot_rew / self.num_eps
        
        logging.info(f"Total Steps: {self.total_steps}")
        logging.info(f"Num. Model Updates: {self.num_updates}")
        logging.info(f"Average Reward: {avg_reward}")
        logging.info(f"Average Ep Length: {avg_ep_len}")
        
        self.write_log(self.total_steps, reward_avg=avg_reward, reward_min=self.min_reward, reward_max=self.max_reward, avg_ep_len=avg_ep_len)
        
        # Save Model if Average Reward is Greater than a Minimum & Better than Before
        if avg_reward >= np.max([opt.min_reward, self.best]) and opt.save_model:
            self.best = avg_reward
            self.policy.save(f'{opt.exp_dir}/last_best.model')

    
    def train_callback(self):
        """Write Training Information to Console"""
        
        logging.info(40*"-")
        
        if len(self.losses) > 0:
            logging.info(f"Total Loss: {np.mean(self.losses):.5f}")
            logging.info(f"Actor Loss: {np.mean(self.a_losses):.5f}")
            logging.info(f"Critic Loss: {np.mean(self.c_losses):.5f}")
            logging.info(f"Entropy Loss: {np.mean(self.e_losses):.5f}")
            logging.info(f"Approx KL Div: {np.mean(self.kl_divs):.3f}")
            
        logging.info(40*"-")


    def reset_memory(self):
        """Create Empty Consolidated Memory Buffer"""
        self.replay_memory = {
            "obss" : np.zeros((self.memory_size, *self.obs_dim,)),
            "acts" : np.zeros((self.memory_size, self.num_actions,)),
            "advs" : np.zeros(self.memory_size),
            "rets" : np.zeros(self.memory_size),
            "prbs" : np.zeros(self.memory_size)
        }


    def create_buffer(self, steps):
        """Create Empty Replay Memory Buffers for Workers"""
        replay_memory = {
            "obss" : np.zeros((steps, *self.obs_dim,)),
            "acts" : np.zeros((steps, self.num_actions,)),
            "rews" : np.zeros(steps),
            "mask" : np.zeros(steps)
        }

        last_vals = np.zeros((steps))
        return replay_memory, last_vals


    def update_buffer(self, mem, step, obs, act, rew, done):
        """Record Stepwise Episode Information with Critic Output"""
        mem["obss"][step-1] = obs if self.mode == 2 else obs/255
        mem["acts"][step-1] = act
        mem["rews"][step-1] = rew
        mem["mask"][step-1] = 0 if done else 1
    

    def process_replay(self, worker, mem, last_vals):
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
            
            # Retrieve Next Step Value, Last Val if Terminal else Val
            if mask == 0 and last_vals[idx] != 0:
                last_val = last_vals[idx]
                print(last_val)
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
            
        out_dict = {
            "obss" : mem["obss"],
            "acts" : mem["acts"],
            "advs" : advs,
            "rews" : mem["rews"],
            "rets" : rets,
            "prbs" : prbs
        }

        for key in self.replay_memory.keys():
            self.replay_memory[key][worker*self.work_steps:worker*self.work_steps+self.work_steps] = out_dict[key]


    def collect_rollout(self, env, total_steps, buffer, last_vals, opt):
        """Collect Experiences from Environment for Training"""
        
        num_steps = 0
        env = env(opt)

        while num_steps != total_steps -1:
            done = False
            last_obs = env.reset() if not opt.debug == 2 else env.reset().T
            ep_reward = 0

            while True:

                # Get Action & Step Environment
                action = self.act(last_obs)
                new_obs, reward, done, _ = env.step(action)

                if opt.debug == 2:
                    obs = obs.T
                    
                # Increment Step Counters
                num_steps += 1
                ep_reward += reward
                
                # Break Early if Rollout has Been Filled, Mark Step as End
                if done or num_steps == self.memory_size - 1:
                    done = True
                    self.update_buffer(buffer, num_steps, last_obs, action, reward, done)
                    break
                
                self.update_buffer(buffer, num_steps, last_obs, action, reward, done)
                last_obs = new_obs
                
            # Calculate Last Value for Finished Episode
            new_obs = np.expand_dims(new_obs, axis=0)
            print(num_steps)
            last_vals[num_steps] = self.critic_network(self.feature_extractor(new_obs, training=False)).numpy()
            self.num_eps += 1
            self.tot_rew += ep_reward
            
            # Replace Highest or Lowest Reward Variables
            if self.min_reward == None or ep_reward < self.min_reward:
                self.min_reward = ep_reward
            elif self.max_reward == None or ep_reward > self.max_reward:
                self.max_reward = ep_reward
        
        self.total_steps += num_steps + 1


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

    
    def learn(self, env, opt):
        """Run Rollout & Training Sequence"""
        
        self.work_steps = self.memory_size // self.num_workers
        print("Work Steps", self.work_steps)
        
        while self.total_steps < self.target_steps:
            
            # Initialise Worker & Tracking Variables
            worker = [None] * self.num_workers
            buffer_arr, last_val_arr = [], []
            self.tot_rew = self.num_eps = 0
            self.min_reward, self.max_reward = None, None
            
            logging.info("Collecting Rollouts...")
            
            # Start Workers to Collect Rollout
            for i in range(self.num_workers):
                buffer, last_vals = self.create_buffer(self.work_steps)
                buffer_arr.append(buffer)
                last_val_arr.append(last_vals)
                worker[i] = threading.Thread(target=self.collect_rollout, args=(env, self.work_steps, buffer, last_vals, opt))
                worker[i].start()

            for i in range(self.num_workers):
                worker[i].join()
                self.process_replay(i, buffer_arr[i], last_val_arr[i])
            
            # Display Performance & Run Training
            self.env_callback(opt)
            self.train()
            
        self.policy.save(f'{opt.exp_dir}/model_last.model')
    

    def train(self):
        """Train Agent by Consuming Collected Memory Buffer"""

        # Retrieve Arrays from Consolidated Memory Buffer
        buffer_obss = self.replay_memory["obss"]
        buffer_acts = self.replay_memory["acts"]
        buffer_advs = self.replay_memory["advs"]
        buffer_rets = self.replay_memory["rets"]
        buffer_prbs = self.replay_memory["prbs"]
        
        # Generate Indices for Sampling
        ind = np.random.permutation(self.memory_size)
        
        # Train Logging
        self.losses = []
        self.e_losses = []
        self.a_losses = []
        self.c_losses = []
        self.rewards = []
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
                    a_loss, entropy, new_log_probs = self.ppo_loss(a_pred, acts, prbs, advs)
                    
                    # Entropy Bonus
                    e_loss = - self.ENTROPY * entropy
                    
                    tot_loss = 0.5 * c_loss + a_loss + e_loss
                
                # Compute KL Divergence for Early Stopping Before Backprop
                kl_div = 0.5 * tf.reduce_mean(tf.square(new_log_probs - prbs))

                if self.TARGET_KL != None and kl_div > self.TARGET_KL:
                    logging.info(f"Early stopping at epoch {epoch+1} due to reaching max kl: {kl_div:.3f}")
                    break
                
                # Compute Gradients & Apply to Model
                gradients = tape.gradient(tot_loss, self.policy.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
                
                # Logging
                self.losses.append(tot_loss.numpy())
                self.e_losses.append(e_loss.numpy())
                self.a_losses.append(a_loss.numpy())
                self.c_losses.append(c_loss.numpy())
                self.kl_divs.append(kl_div)

                # Show Training Progress on Console
                self.train_callback()
                
                if self.best > 120 and self.TARGET_KL == None:
                    self.TARGET_KL = 0.01
                
                self.num_updates += 1
        
        self.reset_memory()


    @tf.function
    def ppo_loss(self, y_pred, acts, old_log_probs, advs):
        """PPO-Clip Loss for Actor"""
        
        # Get New Distributions & Log Probabilities of Actions
        new_dist = tfd.Normal(y_pred, self.ACTOR_SIGMA)
        new_log_probs = new_dist.log_prob(acts)
        
        # Entropy of New Distribution
        entropy = new_dist.entropy()

        # Calculate Ratio Between Old & New Policy
        ratios = tf.math.exp(new_log_probs - old_log_probs)

        # Clipped Actor Loss
        loss1 = advs * ratios
        loss2 = advs * tf.clip_by_value(ratios, 1 - self.PPO_EPSILON, 1 + self.PPO_EPSILON)
        actor_loss = tf.math.reduce_mean(-tf.math.minimum(loss1, loss2))
        
        return actor_loss, entropy, new_log_probs
    
    
    @tf.function
    def critic_loss(self, y_pred, rets):
        """Mean-Squared-Error for Critic"""
        critic_loss = tf.math.squared_difference(rets, y_pred)
        critic_loss = tf.math.reduce_mean(critic_loss)
        return critic_loss