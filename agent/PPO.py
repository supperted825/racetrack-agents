from tensorflow.keras.models import Model, Sequential
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

            # Instantiate Model & Optimizer
            self.policy = self.create_model(opt)
            self.optimizer = Adam(learning_rate=self.lr)

            # Variables to Track Training Progress & Experience Replay Buffer
            self.total_steps = 0
            self.best = 0
            self.num_updates = 0
            self.kl_div = 0
            self.replay_memory = {
                "obss" : [], "acts" : [], "rews" : [], "mask" : []}

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
            self.actor_network.add(Dense(opt.fc_width, activation='relu'))
            self.critic_network.add(Dense(opt.fc_width, activation='relu'))

        # Add Final Output Layers to Each Head
        self.actor_network.add(Dense(self.num_actions, activation='tanh'))
        self.critic_network.add(Dense(1))
        
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
        logging.info(f"Average Episode Length: {avg_ep_len:.3f}")
        logging.info(f"Num. Model Updates: {self.num_updates}")
        logging.info(f"Previous KL Div: {self.kl_div:.3f}")
        logging.info(40*"-")

        self.write_log(self.total_steps, reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, avg_ep_len=avg_ep_len)

        # Save Model if Average Reward is Greater than a Minimum & Better than Before
        if avg_reward >= np.max([opt.min_reward, self.best]) and opt.save_model:
            self.best = avg_reward
            self.policy.save(f'models/{self.name}_best.model')
        
        return ep_rewards, ep_lengths


    def act(self, obs, optimal=False):
        """Get Action from Actor Network"""
        
        with tf.device('/cpu:0'):
            obs = np.expand_dims(obs/255, axis=0)
            feats = self.feature_extractor(obs, training=False)
            action = self.actor_network(feats)
            
            if not optimal:
                dist = tfd.Normal(action, self.ACTOR_SIGMA)
                action = dist.sample()

        return action.numpy()
    

    def evaluate(self, obss, acts):
        """Produce Values & Log Probabilties for Given Batch of Observations"""

        # Predict Value & Prepare Action Outputs
        obss = np.array(obss)/255
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

        for epoch in range(self.epochs):
            
            logging.info(f"Epoch {epoch}: KL Divergence of {self.kl_div:.3f}")
            
            for batch_idx in range(0, len(buffer_obss), self.batch_size): 

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

                # Cast Constant Inputs to Tensors
                acts = tf.constant(acts, tf.float32)
                advs = tf.constant(advs, tf.float32)
                rets = tf.constant(rets, tf.float32)
                prbs = tf.constant(prbs, tf.float32)
                
                with tf.GradientTape() as tape:
                    
                    # Run Forward Passes on Models
                    a_pred, v_pred = self.policy([obss/255], training=True)
                    
                    # Compute Respective Losses
                    c_loss = self.critic_loss(v_pred, rets)
                    a_loss, ratios = self.PPO_loss(a_pred, acts, prbs, advs, entropy)
                    
                    tot_loss = 0.5 * c_loss + a_loss

                # Compute Gradients & Apply to Model
                gradients = tape.gradient(tot_loss, self.policy.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
                
                # Compute KL Divergence for Early Stopping
                self.kl_div = tf.reduce_mean((tf.math.exp(ratios) - 1) - ratios).numpy()

                if self.TARGET_KL != None and self.kl_div > self.TARGET_KL:
                    logging.info(f"Early stopping at epoch {epoch+1} due to reaching max kl: {self.kl_div:.3f}")
                    break
                
                self.num_updates += 1
        
        self.clear_memory()


    @tf.function
    def PPO_loss(self, y_pred, acts, old_log_probs, advs, entropy):
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

        # Entropy Bonus
        entropy_loss = - self.ENTROPY * entropy

        return actor_loss + entropy_loss, ratios
    
    
    @tf.function
    def critic_loss(self, y_pred, rets):
        """Mean-Squared-Error for Critic"""
        critic_loss = tf.math.squared_difference(rets, y_pred)
        critic_loss = tf.math.reduce_mean(critic_loss)
        return critic_loss
        
    